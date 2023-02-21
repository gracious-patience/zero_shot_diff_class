from dally.src.diffusion import ContinuousDiffusion, DiscreteDiffusion, beta_linear_t
from tqdm import tqdm
import torch
import numpy as np
import torch
from einops import repeat
from einops import rearrange
from dally.src.helpers import default, right_pad_dims_to


class Sampler:
    def __init__(
        self,
        unet,
        noise_scheduler,
        dynamic_thresholding=False,
        dynamic_thresholding_percentile=0.9,
        clamp=True,
    ):
        self.unet = unet
        self.noise_scheduler = noise_scheduler
        self.dynamic_thresholding = dynamic_thresholding
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile
        self.clamp = clamp

    def p_mean_variance(
        self,
        x,
        t,
        *,
        text_embeds=None,
        text_mask=None,
        vec=None,
        lowres_cond_img=None,
        lowres_noise_times=None,
        learned_variance=False,
        cond_scale=1.0,
        model_output=None,
        t_next=None,
        labels=None
    ):
        pred = default(
            model_output,
            lambda: self.unet.forward_with_cond_scale(
                x,
                self.noise_scheduler.get_condition(t),
                text_embeds=text_embeds,
                text_mask=text_mask,
                cond_scale=cond_scale,
                lowres_cond_img=lowres_cond_img,
                lowres_noise_times=self.noise_scheduler.get_condition(
                    lowres_noise_times
                ),
                vec=vec,
                labels=labels,
            ),
        )

        if learned_variance:
            pred, var_interp_frac_unnormalized = pred.chunk(2, dim=1)

        x_recon = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        if self.clamp:
            if self.dynamic_thresholding:
                s = torch.quantile(
                    rearrange(x_recon, "b ... -> b (...)").abs(),
                    self.dynamic_thresholding_percentile,
                    dim=-1,
                )
                s.clamp_(min=1.0)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))
                x_recon = x_recon.clamp(-s, s) / s
            else:
                x_recon = x_recon.clamp(-1, 1)

        (
            model_mean,
            posterior_variance,
            posterior_log_variance,
        ) = self.noise_scheduler.q_posterior(x_start=x_recon, x_t=x, t=t, t_next=t_next)

        if learned_variance:
            posterior_log_variance = (
                self.noise_scheduler.get_learned_posterior_log_variance(
                    var_interp_frac_unnormalized, x_t=x, t=t
                )
            )
            posterior_variance = posterior_log_variance.exp()

        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        *,
        text_embeds=None,
        text_mask=None,
        t_next=None,
        cond_scale=1.0,
        vec=None,
        lowres_cond_img=None,
        lowres_noise_times=None,
        learned_variance=False,
        is_last=False,
        labels=None
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            t=t,
            text_embeds=text_embeds,
            text_mask=text_mask,
            cond_scale=cond_scale,
            lowres_cond_img=lowres_cond_img,
            lowres_noise_times=lowres_noise_times,
            learned_variance=learned_variance,
            t_next=t_next,
            vec=vec,
            labels=labels,
        )
        noise = torch.randn_like(x)
        is_last_sampling_timestep = (
            torch.zeros(b, device=device)
            if not is_last
            else torch.ones(b, device=device)
        )
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(
            b, *((1,) * (len(x.shape) - 1))
        )
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(
        self,
        shape,
        *,
        learned_variance=False,
        lowres_cond_img=None,
        lowres_noise_times=None,
        text_embeds=None,
        text_mask=None,
        vec=None,
        labels=None,
        num_timesteps=None,
        cond_scale=1,
        noise=None,
        progress=False,
        start_t=0.999,
        ode_steps=5
    ):
        device = self.noise_scheduler._device.device
        batch = shape[0]
        img = default(noise, torch.randn(shape, device=device))
        if ode_steps > 0:
            timesteps = self.noise_scheduler.get_sampling_timesteps_ode(
                batch, num_timesteps=num_timesteps, p=ode_steps
            )
        else:
            timesteps = self.noise_scheduler.get_sampling_timesteps(
                batch, num_timesteps=num_timesteps, start_t=start_t
            )

        timesteps = (
            tqdm(timesteps, desc="sampling loop time step", total=len(timesteps))
            if progress
            else timesteps
        )
        for i, (times, times_next) in enumerate(timesteps):
            is_last = i == len(timesteps) - 1
            img = self.p_sample(
                img,
                times,
                text_embeds=text_embeds,
                text_mask=text_mask,
                cond_scale=cond_scale,
                t_next=times_next,
                lowres_cond_img=lowres_cond_img,
                lowres_noise_times=lowres_noise_times,
                learned_variance=learned_variance,
                vec=vec,
                is_last=is_last,
                labels=labels,
            )

        if self.clamp:
            img.clamp_(-1.0, 1.0)
        return img

    @torch.no_grad()
    def ddim_step(
        self,
        x,
        t,
        t_next,
        *,
        text_embeds=None,
        text_mask=None,
        vec=None,
        lowres_cond_img=None,
        lowres_noise_times=None,
        learned_variance=False,
        labels=None,
        cond_scale=1.0,
        model_output=None
    ):
        pred = default(
            model_output,
            lambda: self.unet.forward_with_cond_scale(
                x,
                self.noise_scheduler.get_condition(t),
                text_embeds=text_embeds,
                text_mask=text_mask,
                cond_scale=cond_scale,
                lowres_cond_img=lowres_cond_img,
                lowres_noise_times=self.noise_scheduler.get_condition(
                    lowres_noise_times
                ),
                vec=vec,
                labels=labels,
            ),
        )
        x_recon = self.noise_scheduler.predict_start_from_noise(x, t, pred)
        if self.clamp:
            if self.dynamic_thresholding:
                s = torch.quantile(
                    rearrange(x_recon, "b ... -> b (...)").abs(),
                    self.dynamic_thresholding_percentile,
                    dim=-1,
                )
                s.clamp_(min=1.0)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))
                x_recon = x_recon.clamp(-s, s) / s
            else:
                x_recon = x_recon.clamp(-1, 1)

        return self.noise_scheduler.ddim_step(x_recon, x, t, t_next=t_next)

    @torch.no_grad()
    def ddim_sample(
        self,
        shape,
        *,
        learned_variance=False,
        num_timesteps=None,
        lowres_cond_img=None,
        lowres_noise_times=None,
        text_embeds=None,
        text_mask=None,
        vec=None,
        cond_scale=1,
        noise=None,
        progress=False
    ):
        device = self.noise_scheduler._device.device
        batch = shape[0]
        img = default(noise, torch.randn(shape, device=device))
        timesteps = self.noise_scheduler.get_sampling_timesteps(
            batch, num_timesteps=num_timesteps
        )
        timesteps = (
            tqdm(timesteps, desc="sampling loop time step", total=len(timesteps))
            if progress
            else timesteps
        )

        unet_kwargs = {
            "text_embeds": text_embeds,
            "lowres_noise_times": lowres_noise_times,
            "lowres_cond_img": lowres_cond_img,
            "text_mask": text_mask,
            "vec": vec,
            "cond_scale": cond_scale,
        }
        for times, times_next in timesteps:
            img = self.ddim_step(img, times, times_next, **unet_kwargs)
        return img.clamp_(-1, 1)

    #@torch.no_grad()
    def ode_rhs(
        self,
        x,
        t,
        *,
        text_embeds=None,
        text_mask=None,
        vec=None,
        labels=None,
        lowres_cond_img=None,
        lowres_noise_times=None,
        learned_variance=False,
        cond_scale=1.0,
        model_output=None
    ):
        pred = default(
            model_output,
            lambda: self.unet.forward_with_cond_scale(
                x,
                self.noise_scheduler.get_condition(t),
                text_embeds=text_embeds,
                text_mask=text_mask,
                cond_scale=cond_scale,
                lowres_cond_img=lowres_cond_img,
                lowres_noise_times=self.noise_scheduler.get_condition(
                    lowres_noise_times
                ),
                vec=vec,
                labels=labels,
            ),
        )
        x_recon = self.noise_scheduler.predict_start_from_noise(x, t, pred)
        if self.clamp:
            if self.dynamic_thresholding:
                s = torch.quantile(
                    rearrange(x_recon, "b ... -> b (...)").abs(),
                    self.dynamic_thresholding_percentile,
                    dim=-1,
                )
                s.clamp_(min=1.0)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))
                x_recon = x_recon.clamp(-s, s) / s
            else:
                x_recon = x_recon.clamp(-1, 1)

        return self.noise_scheduler.ode_rhs(x_recon, x, t)

    @torch.no_grad()
    def heun_solve(
        self,
        shape,
        *,
        learned_variance=False,
        num_timesteps=None,
        lowres_cond_img=None,
        lowres_noise_times=None,
        text_embeds=None,
        text_mask=None,
        vec=None,
        labels=None,
        cond_scale=1,
        noise=None,
        progress=False,
        ode_steps=7,
        reverse_flow=False,
        start_t = 0.999
    ):

        device = self.noise_scheduler._device.device

        batch = shape[0]
        img = default(noise, torch.randn(shape, device=device))
        if ode_steps == -1:
            timesteps = self.noise_scheduler.get_sampling_timesteps(
                batch, num_timesteps=num_timesteps, start_t=start_t
            )
        else:
            timesteps = self.noise_scheduler.get_sampling_timesteps_ode(
                batch, num_timesteps=num_timesteps, p=ode_steps
            )

        timesteps = (
            tqdm(timesteps, desc="sampling loop time step", total=len(timesteps))
            if progress
            else timesteps
        )

        unet_kwargs = {
            "text_embeds": text_embeds,
            "lowres_noise_times": lowres_noise_times,
            "lowres_cond_img": lowres_cond_img,
            "text_mask": text_mask,
            "vec": vec,
            "cond_scale": cond_scale,
            "labels": labels,
        }
        for (times, times_next) in timesteps:
            if reverse_flow:
                times, times_next = 1 - times_next, 1 - times
                sign = -1
            else:
                sign = 1
            rhs = -sign * self.ode_rhs(img, times, **unet_kwargs)
            dt = right_pad_dims_to(img, times_next - times)
            img_mid = img + dt * rhs
            rhs2 = -sign * self.ode_rhs(img_mid, times_next, **unet_kwargs)
            img = img + 0.5 * dt * (rhs + rhs2)

        if self.clamp:
            img.clamp_(-1, 1)
        return img

    @torch.no_grad()
    def karras_stoch_solve(
        self,
        shape,
        *,
        learned_variance=False,
        num_timesteps=None,
        lowres_cond_img=None,
        lowres_noise_times=None,
        text_embeds=None,
        text_mask=None,
        vec=None,
        cond_scale=1,
        noise=None,
        progress=False,
        ode_steps=7,
        t_min_churn=0.05,
        t_max_churn=1.0,
        S_churn=80,
        S_noise=1.007
    ):

        device = self.noise_scheduler._device.device

        batch = shape[0]
        img = default(noise, torch.randn(shape, device=device))
        timesteps = self.noise_scheduler.get_sampling_timesteps_ode(
            batch, num_timesteps=num_timesteps, p=ode_steps
        )

        timesteps = (
            tqdm(timesteps, desc="sampling loop time step", total=len(timesteps))
            if progress
            else timesteps
        )

        unet_kwargs = {
            "text_embeds": text_embeds,
            "lowres_noise_times": lowres_noise_times,
            "lowres_cond_img": lowres_cond_img,
            "text_mask": text_mask,
            "vec": vec,
            "cond_scale": cond_scale,
        }
        for i, (times, times_next) in enumerate(timesteps):

            gamma = min(S_churn // len(timesteps), np.sqrt(2) - 1)
            gamma_flag = ((times < t_max_churn) & (times > t_min_churn)).float() * gamma

            t_hat = times * (1 + gamma_flag)

            img = self.p_sample(
                img,
                times,
                text_embeds=text_embeds,
                text_mask=text_mask,
            )

            rhs = -self.ode_rhs(img, times, **unet_kwargs)
            dt = right_pad_dims_to(img, times_next - times)
            img_mid = img + dt * rhs
            rhs2 = -self.ode_rhs(img_mid, times_next, **unet_kwargs)
            img = img + 0.5 * dt * (rhs + rhs2)

        if self.clamp:
            img.clamp_(-1, 1)
        return img
