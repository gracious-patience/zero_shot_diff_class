from abc import abstractmethod

import math
from turtle import forward
from cv2 import CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

# from dally.src.unets import PerceiverResampler
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torch import nn, einsum
from einops_exts import rearrange_many
from einops import rearrange, repeat, reduce
import os
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    LayerNorm32,
    timestep_embedding,
    LearnedSinusoidalPosEmb,
    SinusoidalPosEmb,
    FusedSoftmaxMaybeApex,
)
from dally.src.helpers import exists, default, cast_tuple, prob_mask_like, masked_mean

if os.environ.get("APEX", ""):
    try:
        from apex.normalization import FusedLayerNorm as LayerNorm32
    except ImportError:
        print("Apex not installed, please install first.")


def FeedForward(dim, mult=2):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm32(dim),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.GELU(),
        LayerNorm32(hidden_dim),
        nn.Linear(hidden_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=-1, heads=-1):
        super().__init__()
        if dim_head == -1:
            self.heads = heads
            assert dim % heads == 0
            self.dim_head = dim // heads
        else:
            assert (
                dim % dim_head == 0
            ), f"q,k,v channels {dim} is not divisible by dim_head {dim_head}"
            self.heads = dim // dim_head
            self.dim_head = dim_head

        self.scale = self.dim_head**-0.5
        inner_dim = self.dim_head * self.heads

        self.norm = LayerNorm32(dim)
        self.norm_latents = LayerNorm32(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False), LayerNorm32(dim)
        )
        self.softmax_fn = FusedSoftmaxMaybeApex(scale=self.scale)

    def forward(self, x, latents, mask=None):
        x = self.norm(x)
        latents = self.norm_latents(latents)

        b, h = x.shape[0], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = th.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        # q = q * self.scale

        # attention

        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        if exists(mask):
            max_neg_value = -th.finfo(sim.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, "b j -> b 1 1 j")
            sim = sim.masked_fill(~mask, max_neg_value)

        # attn = th.softmax(sim.float(), dim=-1).type(sim.dtype)
        attn = self.softmax_fn(sim)
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
        max_seq_len=128,
        ff_mult=2,
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.latents = nn.Parameter(th.randn(num_latents, dim))

        self.to_latents_from_mean_pooled_seq = None

        if num_latents_mean_pooled > 0:
            self.to_latents_from_mean_pooled_seq = nn.Sequential(
                LayerNorm32(dim),
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
        pos_emb = self.pos_emb(th.arange(n, device=device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, "n d -> b n d", b=x.shape[0])

        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = masked_mean(
                x,
                dim=1,
                mask=th.ones(x.shape[:2], device=device, dtype=th.bool)
                if mask is None
                else mask,
            )
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = th.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents

        return latents


def prob_mask_like(shape, prob, device):
    # always have at least one zero (so we always drop something and DDP is happy)
    if prob == 1:
        return th.ones(shape, device=device, dtype=th.bool)
    elif prob == 0:
        return th.zeros(shape, device=device, dtype=th.bool)
    else:
        proba = th.zeros(shape, device=device).float().uniform_(0, 1)
        idx = np.random.randint(shape[0])
        proba[idx] = 1
        return proba < prob


def attention_mask_func(attention_scores, attention_mask):
    return attention_scores.masked_fill(attention_mask, -10000.0)


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32, undoing convert_module_to_f16().
    """
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()


class TimeAndTextEncoder(nn.Module):
    def __init__(
        self,
        model_channels,
        cond_on_text=False,
        cond_on_vec=False,
        cond_on_discrete=False,
        lowres_time_cond=False,
        dim_text=None,
        dim_vec=None,
        num_classes=None,
        learned_sinusoidal_pos_emb=False,
        attn_pool_depth=2,
        attn_pool_dim_head=64,
        attn_pool_heads=8,
        attn_pool_num_latents=64,
        num_time_tokens=4,
        max_seq_len=128,
    ):
        super().__init__()

        self.model_channels = model_channels
        self.cond_on_text = cond_on_text
        self.cond_on_vec = cond_on_vec
        self.cond_on_discrete = cond_on_discrete
        self.lowres_time_cond = lowres_time_cond
        self.dim_text = dim_text
        self.learned_sinusoidal_pos_emb = learned_sinusoidal_pos_emb
        self.attn_pool_depth = attn_pool_depth
        self.attn_pool_dim_head = attn_pool_dim_head
        self.attn_pool_heads = attn_pool_heads
        self.attn_pool_num_latents = attn_pool_num_latents
        self.num_time_tokens = num_time_tokens
        self.max_seq_len = max_seq_len

        time_cond_dim = model_channels * 4
        cond_dim = model_channels * 4

        self.time_cond_dim = time_cond_dim
        self.cond_dim = cond_dim

        if learned_sinusoidal_pos_emb:
            timestep_embed = LearnedSinusoidalPosEmb(self.model_channels)
            time_embed_dim_initial = model_channels + 1
        else:
            timestep_embed = SinusoidalPosEmb(self.model_channels)
            time_embed_dim_initial = model_channels

        self.to_time_hiddens = nn.Sequential(
            timestep_embed, nn.Linear(time_embed_dim_initial, time_cond_dim), nn.SiLU()
        )
        self.to_time_cond = nn.Sequential(nn.Linear(time_cond_dim, time_cond_dim))

        if cond_on_text:
            self.to_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange("b (r d) -> b r d", r=num_time_tokens),
            )

        if lowres_time_cond:
            if learned_sinusoidal_pos_emb:
                lowres_timestep_embed = LearnedSinusoidalPosEmb(self.model_channels)
            else:
                lowres_timestep_embed = SinusoidalPosEmb(self.model_channels)

            self.lowres_to_time_hiddens = nn.Sequential(
                lowres_timestep_embed,
                nn.Linear(time_embed_dim_initial, time_cond_dim),
                nn.SiLU(),
            )
            self.lowres_to_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )
            if cond_on_text:
                self.lowres_to_time_tokens = nn.Sequential(
                    nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                    Rearrange("b (r d) -> b r d", r=num_time_tokens),
                )

            self.time_merger = nn.Linear(2 * time_cond_dim, time_cond_dim)

        if cond_on_vec:
            self.vec_to_cond = nn.Sequential(
                nn.Linear(dim_vec, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
            )
            self.null_vec_embed = nn.Parameter(th.randn(1, dim_vec))

        if cond_on_discrete:
            self.class_to_cond = nn.Sequential(
                nn.Embedding(num_classes, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
            )

        if cond_on_text:
            self.text_to_cond = nn.Linear(dim_text, cond_dim)

            self.attn_pool = PerceiverResampler(
                dim=cond_dim,
                depth=attn_pool_depth,
                dim_head=attn_pool_dim_head,
                heads=attn_pool_heads,
                num_latents=attn_pool_num_latents,
                max_seq_len=max_seq_len,
            )

            self.to_text_non_attn_cond = nn.Sequential(
                LayerNorm32(cond_dim),
                nn.Linear(cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, time_cond_dim),
            )

            self.null_text_embed = nn.Parameter(th.randn(1, 1, cond_dim))
            self.null_text_hidden = nn.Parameter(th.randn(1, time_cond_dim))

            self.c_norm = LayerNorm32(cond_dim)

    def forward(
        self,
        timesteps,
        lowres_timesteps=None,
        text_embeds=None,
        text_mask=None,
        vec=None,
        labels=None,
        cond_drop_prob=0.1,
    ):

        device = timesteps.device
        batch_size = timesteps.shape[0]

        time_hiddens = self.to_time_hiddens(timesteps)
        t = self.to_time_cond(time_hiddens)

        if self.lowres_time_cond:
            lowres_time_hiddens = self.lowres_to_time_hiddens(lowres_timesteps)
            extra_t = self.lowres_to_time_cond(lowres_time_hiddens)
            t = self.time_merger(th.cat([t, extra_t], dim=1))

        if self.cond_on_text:
            text_keep_mask = prob_mask_like(
                (batch_size,), 1 - cond_drop_prob, device=device
            )

            text_keep_mask_embed = rearrange(text_keep_mask, "b -> b 1 1")
            text_keep_mask_hidden = rearrange(text_keep_mask, "b -> b 1")

            text_tokens = self.text_to_cond(text_embeds)

            text_tokens = text_tokens[:, : self.max_seq_len]
            if exists(text_mask):
                text_mask = text_mask[:, : self.max_seq_len]

            text_tokens_len = text_tokens.shape[1]

            # remainder = self.max_seq_len - text_tokens_len

            # if remainder > 0:
            #     text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            null_text_embed = self.null_text_embed.repeat(1, text_tokens_len, 1)

            null_text_embed = null_text_embed.to(text_tokens.dtype)

            if exists(text_mask):
                text_keep_mask_embed = (
                    rearrange(text_mask, "b n -> b n 1") & text_keep_mask_embed
                )

            text_tokens = th.where(text_keep_mask_embed, text_tokens, null_text_embed)
            text_tokens = self.attn_pool(text_tokens, mask=text_mask)
            mean_pooled_text_tokens = text_tokens.mean(dim=-2)

            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)
            null_text_hidden = self.null_text_hidden.to(t.dtype)

            text_hiddens = th.where(
                text_keep_mask_hidden, text_hiddens, null_text_hidden
            )

            t = t + text_hiddens

            time_tokens = self.to_time_tokens(time_hiddens)
            if self.lowres_time_cond:
                lowres_time_tokens = self.lowres_to_time_tokens(lowres_time_hiddens)
                time_tokens = th.cat((time_tokens, lowres_time_tokens), dim=-2)

            c = th.cat((text_tokens, time_tokens), dim=-2)
            c = self.c_norm(c)

        if self.cond_on_vec:
            if vec is None:
                vec = self.null_vec_embed.to(t.dtype).repeat(batch_size, 1)

            vec = th.where(
                text_keep_mask_hidden, vec.to(t.dtype), self.null_vec_embed.to(t.dtype)
            )
            cond = self.vec_to_cond(vec)
            t = t + cond

        if self.cond_on_discrete:
            cond = self.class_to_cond(labels)

            t = t + cond

        if self.cond_on_text:
            return t, c
        else:
            return t, None


class TimeAndTextEncoderWithPosEmbed(nn.Module):
    def __init__(
        self,
        model_channels,
        cond_on_text=False,
        cond_on_vec=False,
        cond_on_discrete=False,
        lowres_time_cond=False,
        dim_text=None,
        dim_vec=None,
        num_classes=None,
        learned_sinusoidal_pos_emb=False,
        attn_pool_depth=2,
        attn_pool_dim_head=64,
        attn_pool_heads=8,
        attn_pool_num_latents=64,
        num_time_tokens=4,
        max_seq_len=128,
    ):
        super().__init__()

        self.model_channels = model_channels
        self.cond_on_text = cond_on_text
        self.cond_on_vec = cond_on_vec
        self.cond_on_discrete = cond_on_discrete
        self.lowres_time_cond = lowres_time_cond
        self.dim_text = dim_text
        self.learned_sinusoidal_pos_emb = learned_sinusoidal_pos_emb
        self.attn_pool_depth = attn_pool_depth
        self.attn_pool_dim_head = attn_pool_dim_head
        self.attn_pool_heads = attn_pool_heads
        self.attn_pool_num_latents = attn_pool_num_latents
        self.num_time_tokens = num_time_tokens
        self.max_seq_len = max_seq_len

        time_cond_dim = model_channels * 4
        cond_dim = model_channels * 4

        self.time_cond_dim = time_cond_dim
        self.cond_dim = cond_dim

        if learned_sinusoidal_pos_emb:
            timestep_embed = LearnedSinusoidalPosEmb(self.model_channels)
            time_embed_dim_initial = model_channels + 1
        else:
            timestep_embed = SinusoidalPosEmb(self.model_channels)
            time_embed_dim_initial = model_channels

        self.to_time_hiddens = nn.Sequential(
            timestep_embed, nn.Linear(time_embed_dim_initial, time_cond_dim), nn.SiLU()
        )
        self.to_time_cond = nn.Sequential(nn.Linear(time_cond_dim, time_cond_dim))

        if cond_on_text:
            self.to_time_tokens = nn.Sequential(
                nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                Rearrange("b (r d) -> b r d", r=num_time_tokens),
            )

        if lowres_time_cond:
            if learned_sinusoidal_pos_emb:
                lowres_timestep_embed = LearnedSinusoidalPosEmb(self.model_channels)
            else:
                lowres_timestep_embed = SinusoidalPosEmb(self.model_channels)

            self.lowres_to_time_hiddens = nn.Sequential(
                lowres_timestep_embed,
                nn.Linear(time_embed_dim_initial, time_cond_dim),
                nn.SiLU(),
            )
            self.lowres_to_time_cond = nn.Sequential(
                nn.Linear(time_cond_dim, time_cond_dim)
            )
            if cond_on_text:
                self.lowres_to_time_tokens = nn.Sequential(
                    nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
                    Rearrange("b (r d) -> b r d", r=num_time_tokens),
                )

            self.time_merger = nn.Linear(2 * time_cond_dim, time_cond_dim)

        if cond_on_vec:
            self.vec_to_cond = nn.Sequential(
                nn.Linear(dim_vec, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim)
            )
            self.null_vec_embed = nn.Parameter(th.randn(1, dim_vec))

        # This was Valya's code
        # if cond_on_discrete:
        #     self.class_to_cond = nn.Sequential(
        #         nn.Embedding(num_classes, cond_dim),
        #         nn.SiLU(),
        #         nn.Linear(cond_dim, cond_dim),
        #     )

        # Here is mine
        if cond_on_discrete:

            def SinCosEmbed(label) -> th.Tensor:
                embed =  th.Tensor([th.sin( th.tensor(1/(10000**(2*(i//2)/cond_dim))) ) if i%2==0 else 
                th.cos(th.tensor(1/(10000**(2*(i//2)/cond_dim)))) for i in range(cond_dim) ]).cuda()
                # ls_for_mult = label.reshape((label.shape[0], 1)).type(th.FloatTensor).clone().cuda()
                # res = label.reshape((label.shape[0], 1)).type(th.FloatTensor).cuda() @ embed.reshape((1, cond_dim))
                # print(res.requires_grad, label.reshape((label.shape[0], 1)).type(th.FloatTensor).cuda().requires_grad)
                
                a = label.reshape((label.shape[0], 1)).type(th.FloatTensor).cuda()
                # a.requires_grad = True
                # print(a.requires_grad)
                return a @ embed.reshape((1, cond_dim))


            self.class_to_cond = SinCosEmbed

        if cond_on_text:
            self.text_to_cond = nn.Linear(dim_text, cond_dim)

            self.attn_pool = PerceiverResampler(
                dim=cond_dim,
                depth=attn_pool_depth,
                dim_head=attn_pool_dim_head,
                heads=attn_pool_heads,
                num_latents=attn_pool_num_latents,
                max_seq_len=max_seq_len,
            )

            self.to_text_non_attn_cond = nn.Sequential(
                LayerNorm32(cond_dim),
                nn.Linear(cond_dim, time_cond_dim),
                nn.SiLU(),
                nn.Linear(time_cond_dim, time_cond_dim),
            )

            self.null_text_embed = nn.Parameter(th.randn(1, 1, cond_dim))
            self.null_text_hidden = nn.Parameter(th.randn(1, time_cond_dim))

            self.c_norm = LayerNorm32(cond_dim)

    def forward(
        self,
        timesteps,
        lowres_timesteps=None,
        text_embeds=None,
        text_mask=None,
        vec=None,
        labels=None,
        cond_drop_prob=0.1,
    ):

        device = timesteps.device
        batch_size = timesteps.shape[0]

        time_hiddens = self.to_time_hiddens(timesteps)
        t = self.to_time_cond(time_hiddens)

        if self.lowres_time_cond:
            lowres_time_hiddens = self.lowres_to_time_hiddens(lowres_timesteps)
            extra_t = self.lowres_to_time_cond(lowres_time_hiddens)
            t = self.time_merger(th.cat([t, extra_t], dim=1))

        if self.cond_on_text:
            text_keep_mask = prob_mask_like(
                (batch_size,), 1 - cond_drop_prob, device=device
            )

            text_keep_mask_embed = rearrange(text_keep_mask, "b -> b 1 1")
            text_keep_mask_hidden = rearrange(text_keep_mask, "b -> b 1")

            text_tokens = self.text_to_cond(text_embeds)

            text_tokens = text_tokens[:, : self.max_seq_len]
            if exists(text_mask):
                text_mask = text_mask[:, : self.max_seq_len]

            text_tokens_len = text_tokens.shape[1]

            # remainder = self.max_seq_len - text_tokens_len

            # if remainder > 0:
            #     text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            null_text_embed = self.null_text_embed.repeat(1, text_tokens_len, 1)

            null_text_embed = null_text_embed.to(text_tokens.dtype)

            if exists(text_mask):
                text_keep_mask_embed = (
                    rearrange(text_mask, "b n -> b n 1") & text_keep_mask_embed
                )

            text_tokens = th.where(text_keep_mask_embed, text_tokens, null_text_embed)
            text_tokens = self.attn_pool(text_tokens, mask=text_mask)
            mean_pooled_text_tokens = text_tokens.mean(dim=-2)

            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)
            null_text_hidden = self.null_text_hidden.to(t.dtype)

            text_hiddens = th.where(
                text_keep_mask_hidden, text_hiddens, null_text_hidden
            )

            t = t + text_hiddens

            time_tokens = self.to_time_tokens(time_hiddens)
            if self.lowres_time_cond:
                lowres_time_tokens = self.lowres_to_time_tokens(lowres_time_hiddens)
                time_tokens = th.cat((time_tokens, lowres_time_tokens), dim=-2)

            c = th.cat((text_tokens, time_tokens), dim=-2)
            c = self.c_norm(c)

        if self.cond_on_vec:
            if vec is None:
                vec = self.null_vec_embed.to(t.dtype).repeat(batch_size, 1)

            vec = th.where(
                text_keep_mask_hidden, vec.to(t.dtype), self.null_vec_embed.to(t.dtype)
            )
            cond = self.vec_to_cond(vec)
            t = t + cond

        if self.cond_on_discrete:
            cond = self.class_to_cond(labels)
            # print("label embedding = ", cond)
            t = t + cond

        if self.cond_on_text:
            return t, c
        else:
            return t, None

class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :]  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TextConditionalBlock(nn.Module):
    """
    Any module where forward() takes text conditionong as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepMaybeTextEmbedSequential(
    nn.Sequential, TimestepBlock, TextConditionalBlock
):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, t, c=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, t)
            elif isinstance(layer, TextConditionalBlock):
                x = layer(x, c=c)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        else:
            if self.out_channels == channels:
                self.skip_connection = nn.Identity()
            else:
                self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        # if use_conv:
        # #     self.skip_connection = conv_nd(
        # #         dims, channels, self.out_channels, 3, padding=1
        # #     )
        # # else:
        # self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(TextConditionalBlock):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self, dim, heads=1, num_head_channels=-1, cross_attn_dim=None, **kwargs
    ):
        super().__init__()
        self.channels = dim
        if num_head_channels == -1:
            self.num_heads = heads
        else:
            assert (
                dim % num_head_channels == 0
            ), f"q,k,v channels {dim} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = dim // num_head_channels

        self.num_head_channels = dim // self.num_heads

        self.norm = normalization(dim)
        self.qkv = conv_nd(1, dim, dim * 3, 1)

        if cross_attn_dim:
            self.cond_kv = nn.Sequential(conv_nd(1, cross_attn_dim, dim * 2, 1))
        else:
            self.cond_kv = None

        self.attention = QKVAttention(self.num_heads, self.num_head_channels)
        self.proj_out = zero_module(conv_nd(1, dim, dim, 1))

    def forward(self, x, c=None):
        b, chan, *spatial = x.shape
        x = x.reshape(b, chan, -1)
        if c is not None and self.cond_kv is not None:
            c = self.cond_kv(c)
        else:
            c = None

        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, cond_kv=c)
        h = self.proj_out(h)
        return (x + h).reshape(b, chan, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads, n_channels):
        super().__init__()
        self.n_heads = n_heads
        self.scale = 1 / math.sqrt(n_channels)
        self.softmax_fn = FusedSoftmaxMaybeApex(scale=self.scale)

    def forward(self, qkv, cond_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.

        :param kv_cond: an [N x (2 * H * C) x L] tensor of Ks and Vs
        """
        bs, width, length = qkv.shape

        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        if cond_kv is not None:
            k_cond, v_cond = cond_kv.chunk(2, dim=1)
            k = th.cat((k, k_cond), dim=-1)
            v = th.cat((v, v_cond), dim=-1)

        weight = th.einsum(
            "bct,bcs->bts",
            q.reshape(bs * self.n_heads, ch, -1),
            k.reshape(bs * self.n_heads, ch, -1),
        )  # More stable with f16 than dividing afterwards
        # weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        weight = self.softmax_fn(weight)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, -1))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        num_classes=None,
        learned_sinusoidal_pos_emb=True,  # T AND TEXT EMBEDDING PARAMS
        cond_on_text=False,
        cond_on_vec=False,
        cond_on_discrete=False,
        lowres_time_cond=False,
        dim_text=None,
        dim_vec=None,
        attn_pool_depth=2,
        attn_pool_dim_head=64,
        attn_pool_heads=-1,
        attn_pool_num_latents=64,
        num_time_tokens=4,
        max_seq_len=128,
    ):
        super().__init__()

        print(locals())

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if not isinstance(num_res_blocks, list):
            num_res_blocks = [
                num_res_blocks,
            ] * len(channel_mult)

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.cond_on_text = cond_on_text

        self.time_and_text_encoder = TimeAndTextEncoder(
            model_channels=model_channels,
            cond_on_text=cond_on_text,
            cond_on_vec=cond_on_vec,
            cond_on_discrete=cond_on_discrete,
            lowres_time_cond=lowres_time_cond,
            learned_sinusoidal_pos_emb=learned_sinusoidal_pos_emb,
            dim_text=dim_text,
            dim_vec=dim_vec,
            num_classes=num_classes,
            attn_pool_depth=attn_pool_depth,
            attn_pool_dim_head=attn_pool_dim_head,
            attn_pool_heads=attn_pool_heads,
            attn_pool_num_latents=attn_pool_num_latents,
            num_time_tokens=num_time_tokens,
            max_seq_len=max_seq_len,
        )

        # assert len(self.num_res_blocks) == len(self.channel_mult)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepMaybeTextEmbedSequential(
                    conv_nd(dims, in_channels, ch, 3, padding=1)
                )
            ]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        self.time_and_text_encoder.time_cond_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            cross_attn_dim=None
                            if not cond_on_text
                            else self.time_and_text_encoder.cond_dim,
                        )
                    )
                self.input_blocks.append(TimestepMaybeTextEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepMaybeTextEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_and_text_encoder.time_cond_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepMaybeTextEmbedSequential(
            ResBlock(
                ch,
                self.time_and_text_encoder.time_cond_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                cross_attn_dim=None
                if not cond_on_text
                else self.time_and_text_encoder.cond_dim,
            ),
            ResBlock(
                ch,
                self.time_and_text_encoder.time_cond_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_and_text_encoder.time_cond_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            cross_attn_dim=None
                            if not cond_on_text
                            else self.time_and_text_encoder.cond_dim,
                        )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_and_text_encoder.time_cond_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepMaybeTextEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(
        self,
        x,
        timesteps,
        text_embeds=None,
        lowres_noise_times=None,
        lowres_cond_img=None,
        vec=None,
        labels=None,
        cond_drop_prob=0.1,
        text_mask=None,
    ):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if lowres_cond_img is not None:
            x = th.cat((x, lowres_cond_img), 1)

        t, c = self.time_and_text_encoder(
            timesteps,
            lowres_timesteps=lowres_noise_times,
            text_embeds=text_embeds,
            cond_drop_prob=cond_drop_prob,
            vec=vec,
            text_mask=text_mask,
            labels=labels,
        )
        if c is not None:
            c = c.transpose(-1, -2).contiguous()

        hs = []
        # h = x.type(self.dtype)
        for module in self.input_blocks:
            x = module(x, t, c=c)
            hs.append(x)

        
        x = self.middle_block(x, t, c=c)
        

        for module in self.output_blocks:
            
            x = th.cat([x, hs.pop()], dim=1)
            x = module(x, t, c=c)
        # h = h.type(x.dtype)
        return self.out(x)

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs, cond_drop_prob=0.0)
        if cond_scale == 1:
            return logits
        else:
            null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
            return null_logits + (logits - null_logits) * cond_scale


class UNetModelPosEmbed(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
        In this class we use non-trainable embeddings for class, more precisely,
        use this https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
        num_classes=None,
        learned_sinusoidal_pos_emb=True,  # T AND TEXT EMBEDDING PARAMS
        cond_on_text=False,
        cond_on_vec=False,
        cond_on_discrete=False,
        lowres_time_cond=False,
        dim_text=None,
        dim_vec=None,
        attn_pool_depth=2,
        attn_pool_dim_head=64,
        attn_pool_heads=-1,
        attn_pool_num_latents=64,
        num_time_tokens=4,
        max_seq_len=128,
    ):
        super().__init__()

        print(locals())

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if not isinstance(num_res_blocks, list):
            num_res_blocks = [
                num_res_blocks,
            ] * len(channel_mult)

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        self.cond_on_text = cond_on_text

        self.time_and_text_encoder_with_pos_embed = TimeAndTextEncoderWithPosEmbed(
            model_channels=model_channels,
            cond_on_text=cond_on_text,
            cond_on_vec=cond_on_vec,
            cond_on_discrete=cond_on_discrete,
            lowres_time_cond=lowres_time_cond,
            learned_sinusoidal_pos_emb=learned_sinusoidal_pos_emb,
            dim_text=dim_text,
            dim_vec=dim_vec,
            num_classes=num_classes,
            attn_pool_depth=attn_pool_depth,
            attn_pool_dim_head=attn_pool_dim_head,
            attn_pool_heads=attn_pool_heads,
            attn_pool_num_latents=attn_pool_num_latents,
            num_time_tokens=num_time_tokens,
            max_seq_len=max_seq_len,
        )

        # assert len(self.num_res_blocks) == len(self.channel_mult)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepMaybeTextEmbedSequential(
                    conv_nd(dims, in_channels, ch, 3, padding=1)
                )
            ]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        self.time_and_text_encoder_with_pos_embed.time_cond_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            cross_attn_dim=None
                            if not cond_on_text
                            else self.time_and_text_encoder.cond_dim,
                        )
                    )
                self.input_blocks.append(TimestepMaybeTextEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepMaybeTextEmbedSequential(
                        ResBlock(
                            ch,
                            self.time_and_text_encoder_with_pos_embed.time_cond_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepMaybeTextEmbedSequential(
            ResBlock(
                ch,
                self.time_and_text_encoder_with_pos_embed.time_cond_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                cross_attn_dim=None
                if not cond_on_text
                else self.time_and_text_encoder_with_pos_embed.cond_dim,
            ),
            ResBlock(
                ch,
                self.time_and_text_encoder_with_pos_embed.time_cond_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_and_text_encoder_with_pos_embed.time_cond_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            cross_attn_dim=None
                            if not cond_on_text
                            else self.time_and_text_encoder_with_pos_embed.cond_dim,
                        )
                    )
                if level and i == num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_and_text_encoder_with_pos_embed.time_cond_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepMaybeTextEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(
        self,
        x,
        timesteps,
        text_embeds=None,
        lowres_noise_times=None,
        lowres_cond_img=None,
        vec=None,
        labels=None,
        cond_drop_prob=0.1,
        text_mask=None,
    ):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if lowres_cond_img is not None:
            x = th.cat((x, lowres_cond_img), 1)

        t, c = self.time_and_text_encoder_with_pos_embed(
            timesteps,
            lowres_timesteps=lowres_noise_times,
            text_embeds=text_embeds,
            cond_drop_prob=cond_drop_prob,
            vec=vec,
            text_mask=text_mask,
            labels=labels,
        )
        if c is not None:
            c = c.transpose(-1, -2).contiguous()

        hs = []
        # h = x.type(self.dtype)
        for module in self.input_blocks:
            x = module(x, t, c=c)
            hs.append(x)

        
        x = self.middle_block(x, t, c=c)
        

        for module in self.output_blocks:
            
            x = th.cat([x, hs.pop()], dim=1)
            x = module(x, t, c=c)
        # h = h.type(x.dtype)
        return self.out(x)

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs, cond_drop_prob=0.0)
        if cond_scale == 1:
            return logits
        else:
            null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
            return null_logits + (logits - null_logits) * cond_scale


class Base300M(UNetModel):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            in_channels=3,
            out_channels=3,
            image_size=64,
            model_channels=192,
            channel_mult=[1, 2, 3, 4],
            attention_resolutions=[2, 4, 8],
            num_head_channels=64,
            num_res_blocks=3,
            cond_on_text=True,
            dim_text=4096,
            attn_pool_depth=1,
        )
        super().__init__(*args, **kwargs)


class Base500M(UNetModel):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            in_channels=3,
            out_channels=3,
            image_size=64,
            model_channels=256,
            channel_mult=[1, 2, 3, 4],
            attention_resolutions=[2, 4, 8],
            num_head_channels=64,
            num_res_blocks=3,
            cond_on_text=True,
            dim_text=4096,
            attn_pool_depth=1,
        )
        super().__init__(*args, **kwargs)


class Base1B(UNetModel):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            in_channels=3,
            out_channels=3,
            image_size=64,
            model_channels=384,
            channel_mult=[1, 2, 3, 4],
            attention_resolutions=[2, 4, 8],
            num_head_channels=64,
            num_res_blocks=3,
            cond_on_text=True,
            dim_text=1024,
            attn_pool_depth=1,
            max_seq_len=77,
            learned_sinusoidal_pos_emb=False
        )
        super().__init__(*args, **kwargs)


class Base2B(UNetModel):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            in_channels=3,
            out_channels=3,
            image_size=64,
            model_channels=512,
            channel_mult=[1, 2, 3, 4],
            attention_resolutions=[2, 4, 8],
            num_head_channels=64,
            num_res_blocks=3,
            cond_on_text=True,
            dim_text=4096,
            attn_pool_depth=1,
            conv_resample=False,
            learned_sinusoidal_pos_emb=False,
            resblock_updown=False,
        )
        super().__init__(*args, **kwargs)


class EncoderModel:
    pass


class SuperResModel:
    pass


class EncoderUNetModel:
    pass


class EfficientUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
        num_classes=None,
        learned_sinusoidal_pos_emb=True,  # T AND TEXT EMBEDDING PARAMS
        cond_on_text=False,
        cond_on_vec=False,
        lowres_time_cond=False,
        dim_text=None,
        dim_vec=None,
        attn_pool_depth=2,
        attn_pool_dim_head=64,
        attn_pool_heads=-1,
        attn_pool_num_latents=64,
        num_time_tokens=4,
        max_seq_len=128,
        scale_skip=1.0 / math.sqrt(2),
        skip_first_to_last=False,
    ):
        super().__init__()

        print(locals())

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if not isinstance(num_res_blocks, list):
            num_res_blocks = [
                num_res_blocks,
            ] * (len(channel_mult) - 1)

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.scale_skip = scale_skip
        self.skip_first_to_last = skip_first_to_last
        self.cond_on_text = cond_on_text

        self.time_and_text_encoder = TimeAndTextEncoder(
            model_channels=model_channels,
            cond_on_text=cond_on_text,
            cond_on_vec=cond_on_vec,
            lowres_time_cond=lowres_time_cond,
            learned_sinusoidal_pos_emb=learned_sinusoidal_pos_emb,
            dim_text=dim_text,
            dim_vec=dim_vec,
            attn_pool_depth=attn_pool_depth,
            attn_pool_dim_head=attn_pool_dim_head,
            attn_pool_heads=attn_pool_heads,
            attn_pool_num_latents=attn_pool_num_latents,
            num_time_tokens=num_time_tokens,
            max_seq_len=max_seq_len,
        )

        # assert len(self.num_res_blocks) == len(self.channel_mult)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [
                TimestepMaybeTextEmbedSequential(
                    conv_nd(dims, in_channels, ch, 3, padding=1),
                    ResBlock(
                        ch,
                        self.time_and_text_encoder.time_cond_dim,
                        dropout,
                        out_channels=ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_conv=True,
                    ),
                )
            ]
        )

        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult[1:]):
            out_ch = int(mult * model_channels)
            self.input_blocks.append(
                TimestepMaybeTextEmbedSequential(
                    ResBlock(
                        ch,
                        self.time_and_text_encoder.time_cond_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                        use_conv=True,
                    )
                    if resblock_updown
                    else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                )
            )
            ch = out_ch
            input_block_chans.append(ch)
            ds *= 2
            self._feature_size += ch
            ch = int(mult * model_channels)
            for k in range(num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        self.time_and_text_encoder.time_cond_dim,
                        dropout,
                        out_channels=ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_conv=True,
                    )
                ]
                if ds in attention_resolutions and k == num_res_blocks[level] - 1:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            cross_attn_dim=None
                            if not cond_on_text
                            else self.time_and_text_encoder.cond_dim,
                        )
                    )
                self.input_blocks.append(TimestepMaybeTextEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

        self.middle_block = TimestepMaybeTextEmbedSequential(
            ResBlock(
                ch,
                self.time_and_text_encoder.time_cond_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_conv=True,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                cross_attn_dim=None,
            ),
            ResBlock(
                ch,
                self.time_and_text_encoder.time_cond_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                use_conv=True,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])

        for level, mult in list(enumerate(channel_mult))[:0:-1]:
            for i in range(num_res_blocks[level - 1] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        self.time_and_text_encoder.time_cond_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        use_conv=True,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions and i == num_res_blocks[level - 1]:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            cross_attn_dim=None
                            if not cond_on_text
                            else self.time_and_text_encoder.cond_dim,
                        )
                    )
                if level and i == num_res_blocks[level - 1]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            self.time_and_text_encoder.time_cond_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_conv=True,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepMaybeTextEmbedSequential(*layers))
                self._feature_size += ch

        # self.final_res_block = ResBlock(2 * ch, self.time_and_text_encoder.time_cond_dim, dropout, out_channels=ch, use_conv=True, use_scale_shift_norm=True)

        if skip_first_to_last:
            in_ch_last = 2 * ch
        else:
            in_ch_last = ch
        self.out = nn.Sequential(
            normalization(in_ch_last),
            nn.SiLU(),
            zero_module(conv_nd(dims, in_ch_last, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(
        self,
        x,
        timesteps,
        lowres_noise_times=None,
        lowres_cond_img=None,
        text_embeds=None,
        vec=None,
        cond_drop_prob=0.1,
        text_mask=None,
    ):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x C x ...] Tensor of outputs.
        """

        t, c = self.time_and_text_encoder(
            timesteps,
            lowres_timesteps=lowres_noise_times,
            text_embeds=text_embeds,
            cond_drop_prob=cond_drop_prob,
            vec=vec,
        )
        if c is not None:
            c = c.transpose(-1, -2).contiguous()

        hs = []
        if lowres_cond_img is not None:
            x = th.cat((x, lowres_cond_img), axis=1)

        # h = x.type(self.dtype)
        for module in self.input_blocks:
            x = module(x, t, c=c)
            hs.append(x)
        x = self.middle_block(x, t, c=c)
        for module in self.output_blocks:
            x = th.cat([x, self.scale_skip * hs.pop()], dim=1)
            x = module(x, t, c=c)

        if self.skip_first_to_last:
            x = th.cat([x, self.scale_skip * hs.pop()], dim=1)
        # h = h.type(x.dtype)
        return self.out(x)

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs, cond_drop_prob=0.0)
        if cond_scale == 1:
            return logits
        else:
            null_logits = self.forward(*args, cond_drop_prob=1.0, **kwargs)
            return null_logits + (logits - null_logits) * cond_scale


class SRBase300M(EfficientUNetModel):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            in_channels=6,
            out_channels=3,
            image_size=256,
            model_channels=128,
            channel_mult=[1, 1, 2, 4, 4],
            attention_resolutions=[
                16,
            ],
            num_heads=8,
            num_res_blocks=[2, 4, 6, 8],
            cond_on_text=True,
            dim_text=4096,
            attn_pool_depth=1,
            scale_skip=1.0 / math.sqrt(2),
            lowres_time_cond=True,
        )
        super().__init__(*args, **kwargs)


class SRBase300MSimple(UNetModel):
    def __init__(self, *args, **kwargs):
        kwargs.update(
            in_channels=6,
            out_channels=3,
            image_size=256,
            model_channels=128,
            channel_mult=[1, 2, 2, 4, 4],
            attention_resolutions=[
                16,
            ],
            num_head_channels=64,
            num_res_blocks=[1, 2, 4, 6, 8],
            cond_on_text=True,
            dim_text=4096,
            attn_pool_depth=1,
            lowres_time_cond=True,
        )
        super().__init__(*args, **kwargs)
