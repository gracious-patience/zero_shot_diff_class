from atexit import register
import math

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.special import expm1
import torchvision.transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many
from einops_exts.torch import EinopsToAndFrom
from functools import partial
from dally.src.helpers import exists, default, cast_tuple, prob_mask_like, masked_mean
from dally.src.guided_diffusion.unet import AttentionBlock as GDTransformerBlock

# from dally.src.guided_diffusion.nn import GroupNorm32 as GroupNorm
from torch.nn import GroupNorm

NAT = 1.0 / math.log(2.0)


def get_gelu(compressed=False):
    return torch.nn.GELU()


def get_silu(compressed=False):
    return torch.nn.SiLU()


def cos_sin(x):
    return torch.cat([torch.cos(x), torch.sin(x)], -1)


class Scale(nn.Module):
    """scaling skip connection by 1 / sqrt(2), purportedly speeds up convergence in a number of papers"""

    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class Periodic(nn.Module):
    def __init__(self, n_features: int, dim: int) -> None:
        super().__init__()
        self.coefficients = nn.Linear(n_features, dim // 2)

    def forward(self, x):
        assert x.ndim == 2
        return cos_sin(2 * torch.pi * self.coefficients(x))


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class GlobalContext(nn.Module):
    """basically a superior form of squeeze-excitation that is attention-esque"""

    def __init__(self, *, dim_in, dim_out):
        super().__init__()
        self.to_k = nn.Conv2d(dim_in, 1, 1)
        hidden_dim = max(3, dim_out // 2)

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim_out, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        context = self.to_k(x)
        x, context = rearrange_many((x, context), "b n ... -> b n (...)")
        out = einsum("b i n, b c n -> b c i", context.softmax(dim=-1), x)
        out = rearrange(out, "... -> ... 1")
        return self.net(out)


class Always:
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


# attention pooling


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()

        dim_head = dim // heads

        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), nn.LayerNorm(dim)
        )

    def forward(self, x, latents, mask=None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        # attention

        sim = einsum("... i d, ... j d  -> ... i j", q, k)

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_latents=64,
        num_latents_mean_pooled=4,  # number of latents derived from mean pooled representation of the sequence
        max_seq_len=512,
        ff_mult=4,
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, mask=None):
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device=device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])

        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(
                x,
                dim=1,
                mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool),
            )
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents

        return latents


# attention


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head=64,
        heads=8,
        causal=False,
    ):
        super().__init__()

        self.scale = 1.0 / math.sqrt(math.sqrt(dim_head))
        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )

    def forward(self, x, mask=None, attn_bias=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        # q = q

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), "d -> b 1 d", b=b)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # calculate query / key similarities

        sim = einsum("b h i d, b j d -> b h i j", q * self.scale, k * self.scale)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.half_dim = self.dim // 2
        self.sz = math.log(10000) / (self.half_dim - 1)

        self.register_buffer("emb", torch.exp(torch.arange(self.half_dim) * -self.sz))

    def forward(self, x):
        emb = rearrange(x, "i -> i 1") * rearrange(self.emb, "j -> 1 j")
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, norm=True):
        super().__init__()
        self.groupnorm = GroupNorm(groups, dim, eps=1e-5) if norm else nn.Identity()
        self.activation = nn.SiLU()
        self.project = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x, scale_shift=None):
        x = self.groupnorm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.activation(x)
        return self.project(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim=None,
        time_cond_dim=None,
        groups=8,
        linear_attn=False,
        skip_connection_scale=2**-0.5,
        use_gca=False,
        squeeze_excite=False,
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            attn_klass = CrossAttention if not linear_attn else LinearCrossAttention

            self.cross_attn = EinopsToAndFrom(
                "b c h w", "b (h w) c", attn_klass(dim=dim_out, context_dim=cond_dim)
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)

        self.gca = (
            GlobalContext(dim_in=dim_out, dim_out=dim_out) if use_gca else Always(1)
        )

        self.res_conv = (
            nn.Conv2d(dim, dim_out, 1)
            if dim != dim_out
            else Scale(skip_connection_scale)
        )

    def forward(self, x, cond=None, time_emb=None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x)

        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context=cond) + h

        h = self.block2(h, scale_shift=scale_shift)

        h = h * self.gca(h)

        return h + self.res_conv(x)


class CrossAttention(nn.Module):
    def __init__(
        self, dim, *, context_dim=None, dim_head=64, heads=8, norm_context=False
    ):
        super().__init__()

        self.scale = 1.0 / math.sqrt(math.sqrt(dim_head))
        # self.register_buffer("scale", torch.FloatTensor([dim_head**-0.5]))
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm(dim)
        )

    def forward(self, x, context, mask=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(
            self.null_kv.unbind(dim=-2), "d -> b h 1 d", h=self.heads, b=b
        )

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # q = q * self.scale

        sim = einsum("b h i d, b h j d -> b h i j", q * self.scale, k * self.scale)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class LinearCrossAttention(CrossAttention):
    def forward(self, x, context, mask=None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> (b h) n d", h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(
            self.null_kv.unbind(dim=-2), "d -> (b h) 1 d", h=self.heads, b=b
        )

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # masking

        max_neg_value = -torch.finfo(x.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, "b n -> b n 1")
            k = k.masked_fill(~mask, max_neg_value)
            v = v.masked_fill(~mask, 0.0)

        # linear attention

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum("b n d, b n e -> b d e", k, v)
        out = einsum("b n d, b d e -> b n e", q, context)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out(out)


def FeedForward(dim, mult=2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        get_gelu(),
        LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, dim, bias=False),
    )


def ChanFeedForward(
    dim, mult=2
):  # in paper, it seems for self attention layers they did feedforwards with twice channel width
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        ChanLayerNorm(dim),
        nn.Conv2d(dim, hidden_dim, 1, bias=False),
        get_gelu(),
        ChanLayerNorm(hidden_dim),
        nn.Conv2d(hidden_dim, dim, 1, bias=False),
    )


class ChanLayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(1, 1, dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransformerBlock(nn.Module):
    def __init__(self, dim, *, heads=8, dim_head=32, ff_mult=2, use_flash_attn=False):
        super().__init__()
        attn_ = (
            FlashMHA(dim=dim, heads=heads, dim_head=dim_head)
            if use_flash_attn
            else Attention(dim=dim, heads=heads, dim_head=dim_head)
        )

        if use_flash_attn:
            attn_ = nn.Sequential(LayerNorm(dim), attn_, LayerNorm(dim))

        self.attn = EinopsToAndFrom("b c h w", "b (h w) c", attn_)
        self.ff = ChanFeedForward(dim=dim, mult=ff_mult)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class LinearAttention(nn.Module):
    def __init__(self, dim, dim_head=32, heads=8, dropout=0.05):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.SiLU()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias=False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim),
        )

        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias=False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim),
        )

        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(dim, inner_dim, 1, bias=False),
            nn.Conv2d(inner_dim, inner_dim, 3, bias=False, padding=1, groups=inner_dim),
        )

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias=False), ChanLayerNorm(dim)
        )

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = map(lambda fn: fn(fmap), (self.to_q, self.to_k, self.to_v))
        q, k, v = rearrange_many((q, k, v), "b (h c) x y -> (b h) (x y) c", h=h)

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum("b n d, b n e -> b d e", k, v)
        out = einsum("b n d, b d e -> b n e", q, context)
        out = rearrange(out, "(b h) (x y) d -> b (h d) x y", h=h, x=x, y=y)

        out = self.nonlin(out)
        return self.to_out(out)


class LinearAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, *, heads=8, dim_head=32, ff_mult=2):
        super().__init__()
        self.attn = LinearAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.ff = ChanFeedForward(dim=dim, mult=ff_mult)

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class CrossEmbedLayer(nn.Module):
    def __init__(self, dim_in, kernel_sizes, dim_out=None, stride=2):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(
                    dim_in,
                    dim_scale,
                    kernel,
                    stride=stride,
                    padding=(kernel - stride) // 2,
                )
            )

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim=1)


class Unet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_embed_dim=1024,
        text_embed_dim=4096,
        num_resnet_blocks=1,
        cond_dim=None,
        num_image_tokens=4,
        num_time_tokens=2,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        channels_out=None,
        attn_dim_head=64,
        attn_heads=8,
        ff_mult=2.0,
        lowres_cond=False,  # for cascading diffusion - https://cascaded-diffusion.github.io/
        layer_attns=True,
        attend_at_middle=True,  # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        layer_cross_attns=True,
        cond_on_text=True,
        max_text_len=128,
        init_dim=None,
        init_conv_kernel_size=7,
        resnet_groups=8,
        init_cross_embed_kernel_sizes=(3, 7, 15),
        cross_embed_downsample=False,
        cross_embed_downsample_kernel_sizes=(2, 4),
        attn_pool_text=True,
        attn_pool_num_latents=32,
        use_flash_attn=False,
        continuous_time=False,
        use_linear_attn=False,
        use_linear_cross_attn=False,
        use_guided_diffusion_attn=False,
        fp16=False,
    ):
        super().__init__()
        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        self._locals.pop("self", None)
        self._locals.pop("__class__", None)

        # for eventual cascading diffusion

        self.lowres_cond = lowres_cond

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        init_channels = (
            channels if not lowres_cond else channels * 2
        )  # in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        init_dim = default(init_dim, dim)

        self.init_conv = CrossEmbedLayer(
            init_channels,
            dim_out=init_dim,
            kernel_sizes=init_cross_embed_kernel_sizes,
            stride=1,
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time, image embeddings, and optional text encoding

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4

        if not continuous_time:
            self.to_time_hiddens = nn.Sequential(
                SinusoidalPosEmb(dim), nn.Linear(dim, time_cond_dim), get_silu()
            )
        else:
            self.to_time_hiddens = nn.Sequential(
                Rearrange("... -> ... 1"),
                Periodic(1, dim=time_cond_dim),
                nn.SiLU(),
                nn.LayerNorm(time_cond_dim),
                nn.Linear(time_cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.LayerNorm(time_cond_dim),
            )

        self.to_lowres_time_hiddens = None
        if lowres_cond:
            self.to_lowres_time_hiddens = nn.Sequential(
                SinusoidalPosEmb(dim), nn.Linear(dim, time_cond_dim), get_silu()
            )
            time_cond_dim *= 2

        # project to time tokens as well as time hiddens

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange("b (r d) -> b r d", r=num_time_tokens),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(time_cond_dim, time_cond_dim))
        self.norm_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None

        if cond_on_text:
            assert exists(
                text_embed_dim
            ), "text_embed_dim must be given to the unet if cond_on_text is True"
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)

        # finer control over whether to condition on text encodings

        self.cond_on_text = cond_on_text

        # attention pooling

        self.attn_pool = (
            PerceiverResampler(
                dim=cond_dim,
                depth=2,
                dim_head=attn_dim_head,
                heads=attn_heads,
                num_latents=attn_pool_num_latents,
            )
            if attn_pool_text
            else None
        )

        # for classifier free guidance

        self.register_buffer(
            "null_image_embed", torch.zeros(1, num_image_tokens, cond_dim)
        )
        self.max_text_len = max_text_len
        self.register_buffer("null_text_embed", torch.zeros(1, max_text_len, cond_dim))

        # attention related params

        attn_kwargs = dict(heads=attn_heads, dim_head=attn_dim_head)

        num_layers = len(in_out)

        # resnet block klass

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = cast_tuple(resnet_groups, num_layers)

        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)

        assert all(
            [
                layers == num_layers
                for layers in list(
                    map(len, (resnet_groups, layer_attns, layer_cross_attns))
                )
            ]
        )

        # downsample klass

        downsample_klass = Downsample
        if cross_embed_downsample:
            downsample_klass = partial(
                CrossEmbedLayer, kernel_sizes=cross_embed_downsample_kernel_sizes
            )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        transformer_block_klass = (
            LinearAttentionTransformerBlock if use_linear_attn else TransformerBlock
        )
        if use_guided_diffusion_attn:
            transformer_block_klass = GDTransformerBlock

        layer_params = [
            num_resnet_blocks,
            resnet_groups,
            layer_attns,
            layer_cross_attns,
        ]
        reversed_layer_params = list(map(reversed, layer_params))

        for ind, (
            (dim_in, dim_out),
            layer_num_resnet_blocks,
            groups,
            layer_attn,
            layer_cross_attn,
        ) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)
            layer_cond_dim = cond_dim if layer_cross_attn else None

            transformer_config = {
                "dim": dim_out,
                "heads": attn_heads,
                "dim_head": attn_dim_head,
                "ff_mult": ff_mult,
                "fp16": fp16,
            }

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_in,
                            dim_out,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups,
                        ),
                        nn.ModuleList(
                            [
                                ResnetBlock(
                                    dim_out,
                                    dim_out,
                                    groups=groups,
                                    time_cond_dim=time_cond_dim,
                                )
                                for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        transformer_block_klass(**transformer_config)
                        if layer_attn
                        else nn.Identity(),
                        downsample_klass(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
        )

        attn_class = GDTransformerBlock
        attn_mid = attn_class(mid_dim, **attn_kwargs)

        self.mid_attn = attn_mid if attend_at_middle else None
        self.mid_block2 = ResnetBlock(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
        )

        for ind, (
            (dim_in, dim_out),
            layer_num_resnet_blocks,
            groups,
            layer_attn,
            layer_cross_attn,
        ) in enumerate(zip(reversed(in_out[1:]), *reversed_layer_params)):
            layer_cond_dim = cond_dim if layer_cross_attn else None

            transformer_config = {
                "dim": dim_in,
                "heads": attn_heads,
                "dim_head": attn_dim_head,
                "ff_mult": ff_mult,
            }

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                            dim_out * 2,
                            dim_in,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups,
                        ),
                        nn.ModuleList(
                            [
                                ResnetBlock(
                                    dim_in,
                                    dim_in,
                                    groups=groups,
                                    time_cond_dim=time_cond_dim,
                                )
                                for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        transformer_block_klass(**transformer_config)
                        if layer_attn
                        else nn.Identity(),
                        Upsample(dim_in),
                    ]
                )
            )

        # self.final_res_block = ResnetBlock(dim, dim, groups=resnet_groups[0], skip_connection_scale=1.0, time_cond_dim=time_cond_dim)
        self.final_conv = nn.Sequential(
            GroupNorm(resnet_groups[0], dim),
            nn.SiLU(),
            nn.Conv2d(dim, self.channels_out, 1),
        )

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        *,
        lowres_cond_img=None,
        lowres_noise_times=None,
        text_embeds=None,
        text_mask=None,
        cond_drop_prob=0.0,
    ):

        batch_size, device = x.shape[0], x.device

        # add low resolution conditioning, if present

        assert not (
            self.lowres_cond and not exists(lowres_cond_img)
        ), "low resolution conditioning image must be present"
        assert not (
            self.lowres_cond and not exists(lowres_noise_times)
        ), "low resolution conditioning noise time must be present"

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim=1)

        # initial convolution

        x = self.init_conv(x)

        # time conditioning

        time_hiddens = self.to_time_hiddens(time)

        # add the time conditioning for the noised lowres conditioning, if needed

        if exists(self.to_lowres_time_hiddens):
            lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
            time_hiddens = torch.cat((time_hiddens, lowres_time_hiddens), dim=-1)

        # derive time tokens

        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # conditional dropout

        text_keep_mask = prob_mask_like(
            (batch_size,), 1 - cond_drop_prob, device=device
        )

        text_keep_mask = rearrange(text_keep_mask, "b -> b 1 1")

        # take care of text encodings (optional)

        text_tokens = None

        if exists(text_embeds) and self.cond_on_text:
            text_tokens = self.text_to_cond(text_embeds)

            text_tokens = text_tokens[:, : self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            if exists(text_mask):
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder), value=False)

                text_mask = rearrange(text_mask, "b n -> b n 1")
                text_keep_mask = text_mask & text_keep_mask

            null_text_embed = self.null_text_embed.to(
                text_tokens.dtype
            )  # for some reason pytorch AMP not working

            text_tokens = torch.where(text_keep_mask, text_tokens, null_text_embed)

            if exists(self.attn_pool):
                text_tokens = self.attn_pool(text_tokens)

        # main conditioning tokens (c)

        c = (
            time_tokens
            if not exists(text_tokens)
            else torch.cat((time_tokens, text_tokens), dim=-2)
        )

        # normalize conditioning tokens

        c = self.norm_cond(c)

        # go through the layers of the unet, down and up

        hiddens = []

        for init_block, resnet_blocks, attn_block, downsample in self.downs:
            x = init_block(x, c, t)
            for resnet_block in resnet_blocks:
                x = resnet_block(x, c, t)
            x = attn_block(x)
            hiddens.append(x)
            x = downsample(x)

        x = self.mid_block1(x, c, t)
        if exists(self.mid_attn):
            x = self.mid_attn(x)
        x = self.mid_block2(x, c, t)

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = init_block(x, c, t)
            for resnet_block in resnet_blocks:
                x = resnet_block(x, c, t)
            x = attn_block(x)
            x = upsample(x)

        # x = self.final_res_block(x, None, t)
        return self.final_conv(x)


# predefined unets, with configs lining up with hyperparameters in appendix of paper


class BaseUnet64(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                dim=512,
                dim_mults=(1, 2, 3, 4),
                num_resnet_blocks=3,
                layer_attns=(False, True, True, True),
                layer_cross_attns=(False, True, True, True),
                attn_heads=8,
                ff_mult=2.0,
            )
        )
        super().__init__(*args, **kwargs)


class SmallUnet64(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                dim=384,
                dim_mults=(1, 2, 3, 4),
                num_resnet_blocks=3,
                layer_attns=(False, True, True, True),
                layer_cross_attns=(False, True, True, True),
                attn_heads=4,
                ff_mult=2.0,
            )
        )
        super().__init__(*args, **kwargs)


class SRUnet256(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                dim=128,
                dim_mults=(1, 2, 4, 8),
                num_resnet_blocks=(2, 4, 4, 4),
                layer_attns=(False, False, False, True),
                layer_cross_attns=(False, False, False, True),
                attn_heads=8,
                ff_mult=2.0,
            )
        )
        super().__init__(*args, **kwargs)


class SmallSRUnet256(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                dim=96,
                dim_mults=(1, 2, 4, 8),
                num_resnet_blocks=(2, 4, 8, 8),
                layer_attns=(False, False, False, True),
                layer_cross_attns=(False, False, False, True),
                attn_heads=8,
                ff_mult=2.0,
            )
        )
        super().__init__(*args, **kwargs)


class SRUnet1024(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                dim=128,
                dim_mults=(1, 2, 4, 8),
                num_resnet_blocks=(2, 4, 8, 8),
                layer_attns=False,
                layer_cross_attns=(False, False, False, True),
                attn_heads=8,
                ff_mult=2.0,
            )
        )
        super().__init__(*args, **kwargs)


class BaseUnet32(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            dict(
                dim_mults=(1, 2, 2),
                num_resnet_blocks=2,
                layer_attns=(False, True, True),
                layer_cross_attns=False,
                ff_mult=2.0,
            )
        )
        super().__init__(*args, **kwargs)
