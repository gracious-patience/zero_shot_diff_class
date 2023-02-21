import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from dally.src.helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    extract,
    unnormalize_zero_to_one,
    default,
    log,
    right_pad_dims_to,
    maybe,
)
import math
from einops import repeat
from torch.special import expm1
import numpy as np


class DiscreteDiffusion(nn.Module):
    pass


class ContinuousDiffusion(nn.Module):
    pass


class GaussianDiffusion(DiscreteDiffusion):
    def __init__(self, *, beta_schedule, timesteps):
        super().__init__()

        self.continuous = False

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise NotImplementedError()

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # register buffer helper function to cast double back to float

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32), persistent=False
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer(
            "posterior_log_variance_clipped", log(posterior_variance, eps=1e-20)
        )
        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        self.register_buffer("_device", torch.zeros(1))

    def get_times(self, batch_size, noise_level):
        device = self.betas.device
        return torch.full(
            (batch_size,),
            int(self.num_timesteps * noise_level),
            device=device,
            dtype=torch.long,
        )

    def sample_random_times(self, batch_size):
        device = self.betas.device
        return torch.randint(
            0, self.num_timesteps, (batch_size,), device=device, dtype=torch.long
        )

    def get_learned_posterior_log_variance(self, var_interp_frac_unnormalized, x_t, t):
        # if learned variance, posterior variance and posterior log variance are predicted by the network
        # by an interpolation of the max and min log beta values
        # eq 15 - https://arxiv.org/abs/2102.09672
        min_log = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        max_log = extract(torch.log(self.betas), t, x_t.shape)
        var_interp_frac = unnormalize_zero_to_one(var_interp_frac_unnormalized)

        posterior_log_variance = (
            var_interp_frac * max_log + (1 - var_interp_frac) * min_log
        )
        return posterior_log_variance

    def q_posterior(self, x_start, x_t, t, **kwargs):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def ddim_mean_pred(self, x_start, noise, t, **kwargs):
        mean_pred = (
            x_start * torch.sqrt(extract(self.alphas_cumprod_prev, t, x_start.shape))
            + torch.sqrt(1 - extract(self.alphas_cumprod_prev, t, x_start.shape))
            * noise
        )
        return mean_pred

    def get_condition(self, times):
        return times

    def get_sampling_timesteps(self, batch):
        time_transitions = []
        for i in reversed(range(self.num_timesteps)):
            time_transitions.append(
                (
                    torch.full((batch,), i, device=self.betas.device, dtype=torch.long),
                    None,
                )
            )
        return time_transitions

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )


class BetaLinear:
    MAX_LOG_SNR = 9.21
    MIN_LOG_SNR = -10

    @staticmethod
    def t_to_log_snr(t):
        return beta_linear_log_snr(t).clamp(
            min=BetaLinear.MIN_LOG_SNR, max=BetaLinear.MAX_LOG_SNR
        )

    @staticmethod
    def ode_coeff(t):
        return beta_ode_coeff(t)

    @staticmethod
    def log_snr_to_t(log_snr):
        return beta_linear_t(log_snr)


class AlphaCosine:
    MAX_LOG_SNR = 8.769
    MIN_LOG_SNR = -12.928

    @staticmethod
    def t_to_log_snr(t):
        return alpha_cosine_log_snr(t).clamp(
            min=AlphaCosine.MIN_LOG_SNR, max=AlphaCosine.MAX_LOG_SNR
        )

    @staticmethod
    def ode_coeff(t):
        return cosine_ode_coeff(t)

    @staticmethod
    def log_snr_to_t(log_snr):
        return alpha_cosine_t(log_snr).clamp(max=0.999)


def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t**2)))


def beta_linear_t(log_snr):
    return torch.sqrt(0.1 * torch.log1p(torch.exp(-log_snr)) - 1e-5)


def beta_ode_coeff(t):
    return 10 * t


def cosine_ode_coeff(t, s: float = 0.008):
    return torch.tan((t + s) / (1 + s) * math.pi * 0.5) * math.pi / 2 / (1 + s)


def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log(
        (torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1
    )  # not sure if this accounts for beta being clipped to 0.999 in discrete version


def alpha_cosine_t(log_snr, s: float = 0.008):
    return torch.arccos(torch.sqrt(torch.sigmoid(log_snr))) * 2 / math.pi * (1 + s) - s


def fancy_linear_log_snr(t):
    t = 2 - 2 * t
    return -torch.log(expm1((t - 0.5 * torch.log(expm1(1e-4 + 2 * t)))))


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


class GaussianDiffusionContinuousTimes(ContinuousDiffusion):
    def __init__(self, *, beta_schedule, timesteps, elucidate=False):
        super().__init__()
        if beta_schedule == "linear":
            self.schedule = BetaLinear
        elif beta_schedule == "cosine":
            self.schedule = AlphaCosine
        else:
            raise ValueError(f"invalid noise schedule {beta_schedule}")

        self.continuous = True
        self.num_timesteps = timesteps
        self.elucidate = elucidate
        self.register_buffer("_device", torch.zeros(1))

    def get_times(self, batch_size, noise_level):
        return torch.full(
            (batch_size,), noise_level, device=self._device.device, dtype=torch.float
        )

    def sample_random_times(self, batch_size, max_thres=0.999):
        if not self.elucidate:
            return (
                torch.zeros((batch_size,), device=self._device.device)
                .float()
                .uniform_(0, max_thres)
            )
        mean = 0.5 * (self.schedule.MAX_LOG_SNR + self.schedule.MIN_LOG_SNR)
        spread = np.abs((self.schedule.MAX_LOG_SNR - self.schedule.MIN_LOG_SNR))
        sigma = spread / 6
        log_snr = (
            torch.zeros((batch_size,), device=self._device.device)
            .float()
            .normal_(mean, sigma)
        )
        log_snr.clamp_(min=self.schedule.MIN_LOG_SNR, max=self.schedule.MAX_LOG_SNR)
        t = self.schedule.log_snr_to_t(log_snr).clamp_(0, max_thres)
        print(t)
        return t

    def get_condition(self, times):
        return maybe(self.schedule.t_to_log_snr)(times)

    def get_sampling_timesteps(self, batch, num_timesteps=None, start_t=0.999):
        num_timesteps = default(num_timesteps, self.num_timesteps)
        times = torch.linspace(
            start_t, 0.0, num_timesteps + 1, device=self._device.device
        )
        times = repeat(times, "t -> b t", b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def get_sampling_timesteps_with_initial(self, batch, num_timesteps=None, start_t=0.999):
        num_timesteps = default(num_timesteps, self.num_timesteps)
        times = torch.linspace(
            start_t, 0.0, num_timesteps + 1, device=self._device.device
        )
        times = repeat(times, "t -> b t", b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def get_sampling_timesteps_ode(self, batch, num_timesteps=None, p=7):
        num_timesteps = default(num_timesteps, self.num_timesteps)
        sigmas = torch.linspace(
            np.exp(-0.5 * self.schedule.MIN_LOG_SNR / p),
            np.exp(-0.5 * self.schedule.MAX_LOG_SNR / p),
            num_timesteps + 1,
            device=self._device.device,
        ).pow(p)
        log_snrs = torch.log(sigmas.pow(-2))
        times = self.schedule.log_snr_to_t(log_snrs)
        times = repeat(times, "t -> b t", b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times

    def q_posterior(self, x_start, x_t, t, *, t_next=None):
        t_next = default(t_next, lambda: (t - 1.0 / self.num_timesteps).clamp(min=0.0))

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.schedule.t_to_log_snr(t)
        log_snr_next = self.schedule.t_to_log_snr(t_next)
        log_snr, log_snr_next = map(
            partial(right_pad_dims_to, x_t), (log_snr, log_snr_next)
        )

        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        # c - as defined near eq 33
        c = -expm1(log_snr - log_snr_next)
        posterior_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        # following (eq. 33)
        posterior_variance = (sigma_next**2) * c
        posterior_log_variance_clipped = log(posterior_variance, eps=1e-20)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        log_snr = self.schedule.t_to_log_snr(t)
        log_snr_padded_dim = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr_padded_dim)
        return alpha * x_start + sigma * noise, log_snr

    def predict_start_from_noise(self, x_t, t, noise):
        log_snr = self.schedule.t_to_log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        return (x_t - sigma * noise) / alpha.clamp(min=1e-8)

    def ddim_step(self, x_pred, x_t, t, t_next=None):
        t_next = default(t_next, lambda: (t - 1.0 / self.num_timesteps).clamp(min=0.0))

        """ https://openreview.net/attachment?id=2LdBqxc1Yv&name=supplementary_material """
        log_snr = self.schedule.t_to_log_snr(t)
        log_snr_next = self.schedule.t_to_log_snr(t_next)
        log_snr, log_snr_next = map(
            partial(right_pad_dims_to, x_t), (log_snr, log_snr_next)
        )
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

        c = -expm1(0.5 * (log_snr - log_snr_next))
        x_s = x_t * (sigma_next / sigma) + c * x_pred * alpha_next
        return x_s

    def _ode_rhs(self, x_pred, x_t, t):
        # t : 1->0 | noise -> data
        log_snr = self.schedule.t_to_log_snr(t)
        log_snr = right_pad_dims_to(x_t, log_snr)
        alpha, sigma = log_snr_to_alpha_sigma(log_snr)

        # x_t = a x_pred + sigma z
        # no division since it cancels out with * sigma^2
        score = -(x_t - alpha * x_pred) / sigma.pow(2).clamp(1e-8)

        # dgamma = right_pad_dims_to(x_t, self.dlog_snr(t))
        rhs = x_t + score
        return rhs

    def ode_rhs(self, x_pred, x_t, t):
        t = right_pad_dims_to(x_t, t)
        return self.schedule.ode_coeff(t) * self._ode_rhs(x_pred, x_t, t)
