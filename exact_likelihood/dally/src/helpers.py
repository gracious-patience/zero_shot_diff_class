import math
from typing import List
from tqdm import tqdm
from inspect import isfunction
from functools import partial, wraps
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from torch.special import expm1
import torchvision.transforms as T
import cv2
from torch._six import inf
import yt
import base64

from einops import rearrange

# from resize_right import resize

NAT = 1.0 / math.log(2.0)

# helper functions


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)

    return inner


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cast_tuple(val, length=1):
    if isinstance(val, list):
        val = tuple(val)

    return val if isinstance(val, tuple) else ((val,) * length)


def module_device(module):
    return next(module.parameters()).device


@contextmanager
def null_context(*args, **kwargs):
    yield


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def pad_tuple_to_length(t, length, fillvalue=None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))


# tensor helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def l2norm(t):
    return F.normalize(t, dim=-1)


def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))


def masked_mean(t, *, dim, mask=None):
    if not exists(mask):
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


def resize_image_to(image, target_image_size):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    scale_factor = target_image_size / orig_image_size
    return F.interpolate(image, mode="bicubic", scale_factor=scale_factor)


def resize_image_to_size(image, target_image_size):
    return F.interpolate(image, mode="bicubic", size=target_image_size)


# image normalization functions
# ddpms expect images to be in the range of -1 to 1


def normalize_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5


# classifier free guidance functions


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# gaussian diffusion helper functions


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def meanflat(x):
    return x.mean(dim=tuple(range(1, len(x.shape))))


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (
        1.0 + torch.tanh(((2.0 / math.pi) ** 0.5) * (x + 0.044715 * (x**3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres=0.999):
    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus)
    log_one_minus_cdf_min = log(1.0 - cdf_min)
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(
        x < -thres,
        log_cdf_plus,
        torch.where(x > thres, log_one_minus_cdf_min, log(cdf_delta)),
    )

    return log_probs


def cosine_beta_schedule(timesteps, s=0.008, thres=0.999):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, thres)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def decode_img_from_bytes(
    image_bytes, dtype=np.uint8, normalize=True, resize_to=64, use_base64=False
):

    if isinstance(image_bytes, yt.yson.yson_types.YsonStringProxy):
        image_bytes = image_bytes._bytes
    image_bytes = yt.yson.get_bytes(image_bytes)

    if use_base64:
        image_bytes = base64.b64decode(image_bytes)

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=dtype), cv2.IMREAD_COLOR)

    h, w, = (
        image.shape[0],
        image.shape[1],
    )
    # if h > 256:
    #     crop_x = np.random.randint(0, h - 256)
    #     crop_y = np.random.randint(0, h - 256)
    #     image = image[crop_x : crop_x + 256, crop_y : crop_y + 256]

    # image = cv2.resize(
    #     image, (resize_to, resize_to), interpolation=cv2.INTER_CUBIC
    # ).clip(0, 255)
    image = image.clip(0, 255)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = np.transpose(image, [2, 0, 1])
    if normalize:
        image = (image - 0.5) / 0.5

    return image


def decode_nparray_from_bytes(arr_bytes, dtype=np.float32):
    if isinstance(arr_bytes, yt.yson.yson_types.YsonStringProxy):
        arr_bytes = arr_bytes._bytes

    arr_bytes = yt.yson.get_bytes(arr_bytes)
    arr = np.frombuffer(arr_bytes, dtype=dtype)
    return arr


def max_grad_norm(
    parameters, norm_type: float = 2.0, error_if_nonfinite: bool = False
) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm
