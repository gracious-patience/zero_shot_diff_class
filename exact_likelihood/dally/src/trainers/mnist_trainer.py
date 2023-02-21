from dally.utils import ddp
from regex import P
from dally.bert.utils.fs import nirvana
from dally.bert.utils.distributed.mpi import mpicomm
import logging
import torch
from dally.utils.configs import instantiate_from_config
from dally.src.diffusion import GaussianDiffusion
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import io
import time
from dally.src.augment.augment import AugmentPipe
import os
import numpy as np
from dally.src.helpers import max_grad_norm
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import make_grid
import torchvision.transforms as T
import torch_fidelity
from dally.src.helpers import default
from torch.utils.data import DistributedSampler

from dally.src.guided_diffusion.script_util import create_gaussian_diffusion


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x):
        super().__init__()

        self.x = (x.add(1).div(2).cpu() * 255).type(torch.uint8)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i]


MAX_LENGTH = 128

logger = logging.getLogger(__name__)


def get_transform():
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(16),
            T.RandomHorizontalFlip(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return transform


def rank():
    return ddp.worker_local_idx()


class Trainer:
    def __init__(self, config):
        self.config = config
        assert ddp.num_workers() in [1, 2, 4, 8]
        self._unet = instantiate_from_config(config.diffusion.unet).to(rank())
        if config.trainer.augment:
            self.pipe = AugmentPipe(
                xflip=1, rotate90=1, xint=1, xint_max=0.5, scale=1, aniso=1
            )
            self.equiv_loss = config.trainer.equivariance_loss
        else:
            self.pipe = None
            self.equiv_loss = 0

        self._ddpm = instantiate_from_config(
            config.diffusion.generator,
            {
                "unet": self._unet,
                "aug_pipe": self.pipe,
                "equiv_loss_weight": self.equiv_loss,
            },
        ).to(rank())

        logger.info(f"TOTAL PARAMS: {sum(p.numel() for p in self._ddpm.parameters())}")

        self.emas = []
        for beta in config.trainer.betas_ema:
            config.diffusion.ema.params.beta = beta
            ema = instantiate_from_config(config.diffusion.ema, {"model": self._unet})
            self.emas.append(ema)

        self._model = DDP(self._ddpm)

        self._amp = config.trainer.amp
        self._debug = config.trainer.debug

        self._build_optim()
        self._build_schedulers()

        self._report_every = config.trainer.report_metrics_every
        self._eval_every = config.trainer.eval_every
        self._save_every = config.trainer.save_every
        self._eval_after = config.trainer.get("eval_after", 0)

        self._eval_algo = config.trainer.get("eval_algo", "ddpm")
        self._eval_num_timesteps = config.trainer.get("eval_num_timesteps", 1000)
        self._final_eval_algo = config.trainer.get("final_eval_algo", "ddpm")
        self._final_eval_num_timesteps = config.trainer.get(
            "final_eval_num_timesteps", 1000
        )

        self.step = torch.zeros(1, device=rank(), dtype=torch.long)
        self._best_step = torch.zeros((1,), device=rank(), dtype=torch.long)
        self._best_fid = 100000 * torch.ones((1,), device=rank())
        self.max_steps = config.trainer.max_training_steps
        self.valid_steps = 0

        self._set_dirs_and_resume()
        # measure imgs/sec when we restart
        self._local_step = 0

        if ddp.dist_utils.is_chief():
            self.writer = SummaryWriter(log_dir=self.logdir)

        if not self._amp:
            self._dtype = torch.float32
        else:
            if config.trainer.amp_dtype == "float16":
                self._dtype = torch.float16
            elif config.trainer.amp_dtype == "bfloat16":
                self._dtype = torch.bfloat16
            else:
                raise ValueError(f"Unknown amp dtype: {config.trainer.amp_dtype}")
        self._instantiate_data()

        if self._debug:
            self._activations = {}
            for name, module in self._model.module.unet.named_children():
                module.register_forward_hook(self._get_activation(name))

    def _get_activation(self, name):
        # the hook signature
        def hook(model, input, output):
            self._activations[name] = output.detach()

        return hook

    def _set_dirs_and_resume(self, ckpt_path=None):
        ddp.barrier()
        if nirvana.ndl is not None:
            # We are in nirvana
            self.checkpoint_dir = nirvana.checkpoint_dir()
            self.logdir = nirvana.logs_dir()
        else:
            checkpoint_dir = default(self.config.trainer.checkpoint_dir, "./states")
            logdir = default(self.config.trainer.log_dir, "./logs")
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(logdir, exist_ok=True)
            self.checkpoint_dir = checkpoint_dir
            self.logdir = logdir

        state_dict = None
        if ddp.is_chief():
            if ckpt_path:
                self.ckpt_path = ckpt_path
            else:
                paths = list(
                    filter(
                        lambda s: s.endswith(".pth"), os.listdir(self.checkpoint_dir)
                    )
                )
                self.ckpt_path = None
                if len(paths) > 0:
                    self.ckpt_path = os.path.join(
                        self.checkpoint_dir, sorted(paths)[-1]
                    )
                    logger.info("CHECKPOINT FOUND ON CHIEF")

            if self.ckpt_path:
                with open(self.ckpt_path, "rb") as rb:
                    state_dict = rb.read()

        state_dict = self.bcast_model_file(state_dict)
        if state_dict is None:
            logger.info(f"Could not load checkpoint on {ddp.worker_idx()}")
        else:
            state_dict = io.BytesIO(state_dict)
            state_dict = torch.load(state_dict, map_location=f"cuda:{rank()}")

            # RESUME EMA AND ONLINE MODEL
            for i in range(len(self.emas)):
                ema_state_dict = state_dict[f"ema_{i}"]
                self.emas[i].load_state_dict(ema_state_dict)
            self._model.module.unet.load_state_dict(
                self.emas[0].online_model.state_dict()
            )

            for i in range(len(self.emas)):
                self.emas[i].online_model = self._model.module.unet

            self.step = state_dict["step"]
            self._best_fid = state_dict["_best_fid"]
            self._best_step = state_dict["_best_step"]
            self._optimizer.load_state_dict(state_dict["optimizer"])
            self._scheduler.load_state_dict(state_dict["scheduler"])
            self._warmup.load_state_dict(state_dict["warmup"])
            self._scaler.load_state_dict(state_dict["scaler"])

        ddp.barrier()

    def bcast_model_file(self, model_file):
        _BYTES_PER_CHUNK = 2000000000
        num_chunks = None
        if ddp.is_chief():
            if model_file is not None:
                num_chunks = (
                    len(model_file) + _BYTES_PER_CHUNK - 1
                ) // _BYTES_PER_CHUNK
            logger.info(f"Total model file chunks: {num_chunks}")

        num_chunks = mpicomm.bcast(num_chunks, root=0)
        if num_chunks is None:
            return None
        chunk_data = [None for _ in range(num_chunks)]
        if ddp.is_chief():
            chunk_data = [
                model_file[
                    chunk_id * _BYTES_PER_CHUNK : (chunk_id + 1) * _BYTES_PER_CHUNK
                ]
                for chunk_id in range(num_chunks)
            ]
        for i in range(num_chunks):
            chunk_data[i] = mpicomm.bcast(chunk_data[i], root=0)
        return b"".join(chunk_data)

    def _save_checkpoint(self):
        ddp.barrier()
        if ddp.dist_utils.is_chief():
            logger.info("Saving checkpoint.")
            logger.info(
                f"Best fid: {self._best_fid.item()} on step {self._best_step.item()}."
            )
            state_dict = {}
            for i in range(len(self.emas)):
                state_dict[f"ema_{i}"] = self.emas[i].state_dict()
            state_dict["step"] = self.step
            state_dict["_best_fid"] = self._best_fid
            state_dict["_best_step"] = self._best_step
            state_dict["optimizer"] = self._optimizer.state_dict()
            state_dict["scheduler"] = self._scheduler.state_dict()
            state_dict["warmup"] = self._warmup.state_dict()
            state_dict["scaler"] = self._scaler.state_dict()
            fname = os.path.join(
                self.checkpoint_dir, f"ckpt_{self.step.item():09d}.pth"
            )
            torch.save(state_dict, fname)
            if nirvana.ndl:
                nirvana.ndl.snapshot.dump_snapshot()
        ddp.barrier()

    def _build_optim(self):
        logger.info("Building optimizers.")
        self.batch_size = self.config.trainer.batch_size
        self._optimizer = instantiate_from_config(
            self.config.optimizer, {"params": self._model.module.unet.parameters()}
        )
        self._max_grad_norm = self.config.trainer.max_grad_norm
        self._scaler = GradScaler(enabled=self._amp)

    def _build_schedulers(self):
        self._scheduler = instantiate_from_config(
            self.config.scheduler, {"optimizer": self._optimizer}
        )
        self._warmup = instantiate_from_config(
            self.config.warmup, {"optimizer": self._optimizer}
        )

    def _sample(
        self,
        ema_index=0,
        text_embeds=None,
        num_samples=1250,
        chunk=1,
        num_timesteps=1000,
        eval_algo="ddpm",
        progress=False,
    ):

        SIZE = num_samples

        ddp.barrier()
        if ema_index == -1:
            unet = self.emas[0].online_model
        else:
            unet = self.emas[ema_index].ema_model

        unet.eval()
        if self.pipe:
            cond = torch.zeros(SIZE // chunk, self.config.augment_dim).to(rank())
        else:
            cond = None

        if self.config.trainer.conditional:
            labels = torch.randint(0, 10, size=(SIZE // chunk,), device=rank())
        else:
            labels = None

        sampler = instantiate_from_config(
            self.config.diffusion.sampler,
            {"unet": unet, "noise_scheduler": self._ddpm.noise_scheduler},
        )
        # diffusion = create_gaussian_diffusion(timestep_respacing=respacing)
        # sample_fn = diffusion.ddim_sample_loop if mode == "ddim" else diffusion.p_sample_loop

        sample_fn = sampler.p_sample_loop if eval_algo == "ddpm" else sampler.heun_solve

        outputs = []
        for i in range(chunk):
            with autocast(enabled=self._amp, dtype=self._dtype):
                output = (
                    sample_fn(
                        shape=(SIZE // chunk, 1, 16, 16),
                        num_timesteps=num_timesteps,
                        noise=torch.randn(
                            SIZE // chunk,
                            1,
                            16,
                            16,
                            generator=torch.Generator().manual_seed(100 * rank() + i),
                        ).to(rank()),
                        progress=progress,
                        vec=cond,
                        labels=labels,
                    )
                    .cpu()
                    .numpy()
                )

            outputs.append(output)

        outputs = np.concatenate(outputs, 0)
        ddp.barrier()

        outputs = mpicomm.gather(outputs, 0)

        if ddp.is_chief():
            outputs = torch.from_numpy(np.concatenate(outputs)).float()
            image = make_grid(outputs[:64], nrow=8).add(1).div(2)
        else:
            image = None

        unet.train()
        ddp.barrier()
        return image, outputs

    def _instantiate_data(self, skip_rows=0):
        self._reset_train(skip_rows=skip_rows)

    def _reset_train(self, skip_rows):
        logger.info("Building train dataset.")

        if nirvana.ndl is not None:
            root = nirvana.data_dir()
        else:
            root = "./cache"

        self._train_dataset = instantiate_from_config(
            self.config.data.train_dataset,
            {"root": root, "download": True, "transform": get_transform()},
        )
        self._train_sampler = DistributedSampler(
            self._train_dataset,
            shuffle=True,
            drop_last=True,
            num_replicas=ddp.num_workers(),
            rank=rank(),
        )
        self._train_loader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=1,
            sampler=self._train_sampler,
        )

    def _np2torch(self, x):
        return torch.from_numpy(x).to(rank())

    def _get_lr(self):
        for param_group in self._optimizer.param_groups:
            return param_group["lr"]

    def _train_step(self, batch):
        metrics = {}

        time_start = time.time()

        self._optimizer.zero_grad()

        image, labels = batch
        image = image.to(rank())

        if self.pipe:
            [image_aug], embeds = self.pipe([image])
            embeds = embeds.to(rank())
            embeds_orig = torch.zeros_like(embeds)

            image = torch.cat([image, image_aug], 0)
            cond = torch.cat([embeds_orig, embeds], 0)
        else:
            cond = None

        if self.config.trainer.conditional:
            labels = labels.to(rank())
        else:
            labels = None

        time_start_forward = time.time()

        with autocast(enabled=self._amp, dtype=self._dtype):
            loss_ = self._model(image=image, vec=cond, labels=labels)
            if self.pipe:
                loss = loss_["loss"]
                loss_equi = loss_["loss_equi"]
                metrics["loss_equi"] = loss_equi.item()
                loss = loss + self.equiv_loss * loss_equi
            else:
                loss = loss_

        # torch.cuda.synchronize()

        assert torch.isnan(loss).sum() == 0, "NaN loss encountered"

        self._scaler.scale(loss).backward()

        # CLIP GRAD NORMS
        self._scaler.unscale_(self._optimizer)

        grad_norm = max_grad_norm(self._model.module.unet.parameters())
        nn.utils.clip_grad_norm_(
            self._model.module.unet.parameters(), self._max_grad_norm
        )
        if ddp.is_chief():
            if self._debug:
                logger.info("Logging params.")
                for name, param in self._model.module.unet.named_parameters():
                    if param.grad is not None:
                        metrics[f"gradients/{name}"] = torch.norm(param.view(-1))

                        if self.step % 250 == 0:
                            if ddp.is_chief():
                                self.writer.add_histogram(
                                    f"params/values/{name}", param.view(-1), self.step
                                )
                                self.writer.add_histogram(
                                    f"params/gradients/{name}",
                                    param.grad.view(-1),
                                    self.step,
                                )
                if self.step % 250 == 0:
                    for k, v in self._activations.items():
                        self.writer.add_histogram(
                            f"activations/{k}", v.reshape(-1), self.step
                        )

        self._scaler.step(self._optimizer)
        self._scaler.update()

        with self._warmup.dampening():
            self._scheduler.step()

        # torch.cuda.synchronize()
        time_end_forward = time.time()

        time_start_ema = time.time()
        for ema in self.emas:
            ema.update()
        # torch.cuda.synchronize()
        time_end_ema = time.time()

        self.step += 1
        self._local_step += 1
        steps_per_sec = (
            self._local_step * self.batch_size * ddp.dist_utils.num_workers()
        ) / (time.time() - self.start)

        time_end = time.time()

        metrics.update(
            {
                "loss": loss.item(),
                "imgs/sec": steps_per_sec,
                "lr": self._get_lr(),
                "time/step": time_end - time_start,
                "time/ema": time_end_ema - time_start_ema,
                "time/forward": time_end_forward - time_start_forward,
                "grad_norm": grad_norm,
            }
        )
        return metrics

    def _report_metrics(self, metrics):
        if ddp.dist_utils.is_chief():
            s = f" | ".join(
                map(
                    lambda x: f"{x[0]}: {x[1]:.3f}"
                    if x[1] > 1e-3
                    else f"{x[0]}: {x[1]:.2e}",
                    filter(lambda x: "grad" not in x[0], metrics.items()),
                )
            )
            epoch = self.step.item() // len(self._train_loader)
            done = (self.step.item() % len(self._train_loader)) / len(
                self._train_loader
            )
            s += f" | step: {self.step.item()}"

            metrics.update({"epoch": epoch, "done": done})
            # metrics.update({"scale" : self._scaler.get_scale()})
            logging.info(s)
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, global_step=self.step.item())

    @property
    def step_for_report(self):
        return self.step % self._report_every == 0

    @property
    def step_for_eval(self):
        return self.step % self._eval_every == 0 and self.step > self._eval_after

    @property
    def step_for_checkpoint(self):
        return self.step % self._save_every == 0

    @torch.no_grad()
    def eval(
        self, ema_index=0, chunk=1, num_samples=1250, progress=False, **sample_kwargs
    ):
        ddp.barrier()
        image, output = self._sample(
            chunk=chunk,
            ema_index=ema_index,
            num_samples=num_samples,
            text_embeds=None,
            progress=progress,
            **sample_kwargs,
        )
        metrics_dict = None
        if ddp.dist_utils.is_chief():
            self.writer.add_image("samples", image, global_step=self.step.item())
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=MyDataset(output),
                input2="mnist-train",
                cuda=True,
                isc=False,
                fid=True,
                kid=False,
                verbose=False,
                datasets_root="./cache" if not nirvana.ndl else nirvana.data_dir(),
                feature_extractor_weights_path=None
                if not nirvana.ndl
                else os.path.join(
                    nirvana.data_dir(),
                    "fidelity_cache",
                    "weights-inception-2015-12-05-6726825d.pth",
                ),
                datasets_download=False,
            )
            fid = metrics_dict["frechet_inception_distance"]
            if fid < self._best_fid:
                self._best_fid = torch.FloatTensor(
                    [
                        fid,
                    ]
                ).to(rank())
                self._best_step = self.step.clone()
        del output
        ddp.barrier()
        return metrics_dict

    def _final_eval(self):
        ddp.barrier()
        best_ckpt = None
        if ddp.is_chief():
            best_ckpt = os.path.join(
                self.checkpoint_dir, f"ckpt_{self._best_step.item():09d}.pth"
            )

        best_ckpt = mpicomm.bcast(best_ckpt, root=0)
        self._set_dirs_and_resume(best_ckpt)
        samples_per_worker = 50000 // ddp.num_workers()
        safe_size = 1250
        chunks = samples_per_worker // safe_size

        for i in range(len(self.emas)):
            metrics = self.eval(
                ema_index=i,
                chunk=chunks,
                num_samples=samples_per_worker,
                num_timesteps=self._final_eval_num_timesteps,
                eval_algo=self._final_eval_algo,
                progress=False,
            )
            if ddp.is_chief():
                logger.info(f"-------EMA MODEL {i}---")
                logger.info("-------BEST FID-------")
                self._report_metrics(metrics)
                logger.info("----------------------")

    def train(self):
        self.start = time.time()
        self.step += 1
        self._local_step += 1
        while self.step < self.max_steps:
            self._model.train()
            self._train_sampler.set_epoch(self.step.item())  # kek for randomness
            for batch in self._train_loader:
                metrics = self._train_step(batch)
                if self.step_for_report:
                    self._report_metrics(metrics)

                if self.step_for_eval:
                    metrics = self.eval(
                        num_timesteps=self._eval_num_timesteps,
                        num_samples=6250,
                        chunk=5,
                        progress=True,
                        eval_algo=self._eval_algo,
                    )
                    self._report_metrics(metrics)

                if self.step_for_checkpoint:
                    self._save_checkpoint()

        self._save_checkpoint()
        self._final_eval()

    def profile(self):
        self.start = time.time()
        self.step += 1
        self._local_step += 1
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.logdir),
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for step, batch in enumerate(self._train_loader):
                if step >= (1 + 1 + 3) * 2:
                    break
                metrics = self._train_step(batch)
                prof.step()
