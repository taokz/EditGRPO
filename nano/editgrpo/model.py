"""Pure-PyTorch Qwen2.5-VL (vision tower + patch merger + Qwen2 decoder with M-RoPE).

Structure and style follow huggingface/nanoVLM and Emericen/tiny-qwen (both MIT licensed):
one readable file, plain ``nn.Module`` blocks, no framework glue. The maths and the parameter
names follow the Hugging Face ``transformers`` implementation of Qwen2.5-VL
(``modeling_qwen2_5_vl.py``, Apache-2.0), so a Hub checkpoint loads with (almost) identity key
mapping; see ``load_pretrained`` for the two renames that exist.

Dependencies: torch, safetensors, numpy, PIL (preprocessing only), huggingface_hub (optional,
only to download a repo id). No ``transformers``.

Conventions used throughout
- ``input_ids``/``attention_mask``: ``(B, T)``; ``attention_mask`` is 1 for real tokens, 0 for
  padding. Left-padded prompts and right-padded responses are both supported.
- ``position_ids``: ``(3, B, T)`` -- the three M-RoPE axes (temporal, height, width).
- KV cache: ``list[(k, v)]`` with one entry per decoder layer, ``k``/``v`` of shape
  ``(B, num_key_value_heads, T_cached, head_dim)``.
- ``pixel_values``: ``(num_patches, in_channels * temporal_patch_size * patch_size**2)``,
  ``image_grid_thw``: ``(num_images, 3)``, exactly as produced by ``image_preprocess``.
"""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------


@dataclass
class VisionConfig:
    """Mirror of ``vision_config`` in a Qwen2.5-VL ``config.json`` (defaults: 3B-Instruct)."""

    depth: int = 32
    hidden_size: int = 1280
    intermediate_size: int = 3420
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 14
    spatial_patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    window_size: int = 112
    fullatt_block_indexes: list[int] = field(default_factory=lambda: [7, 15, 23, 31])
    out_hidden_size: int = 2048  # = language-model hidden size
    hidden_act: str = "silu"
    tokens_per_second: int = 2  # video only; kept so config.json round-trips

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


@dataclass
class Qwen25VLConfig:
    """Top-level config; field names match the Hub ``config.json`` of Qwen/Qwen2.5-VL-3B-Instruct."""

    hidden_size: int = 2048
    intermediate_size: int = 11008
    num_hidden_layers: int = 36
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    vocab_size: int = 151936
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 128000
    mrope_section: list[int] = field(default_factory=lambda: [16, 24, 24])
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_config: VisionConfig = field(default_factory=VisionConfig)
    torch_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if isinstance(self.vision_config, dict):
            self.vision_config = VisionConfig(**self.vision_config)
        if sum(self.mrope_section) != self.head_dim // 2:
            raise ValueError(f"sum(mrope_section) must equal head_dim/2 = {self.head_dim // 2}")
        if self.vision_config.out_hidden_size != self.hidden_size:
            raise ValueError("vision_config.out_hidden_size must equal hidden_size")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "Qwen25VLConfig":
        """Accepts both the transformers<5 layout (text fields top-level, ``rope_scaling``) and the
        transformers>=5 layout (``text_config`` nested, ``rope_parameters``)."""
        d = dict(cfg)
        d.update(d.pop("text_config", {}) or {})
        rope = d.get("rope_scaling") or d.get("rope_parameters") or {}
        kwargs = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        kwargs["mrope_section"] = list(rope.get("mrope_section", kwargs.get("mrope_section", [16, 24, 24])))
        if "rope_theta" in rope:
            kwargs["rope_theta"] = rope["rope_theta"]
        vis = dict(d.get("vision_config", {}))
        if "in_chans" in vis:  # older config.json spelling
            vis["in_channels"] = vis.pop("in_chans")
        vis_kwargs = {f.name: vis[f.name] for f in fields(VisionConfig) if f.name in vis}
        kwargs["vision_config"] = VisionConfig(**vis_kwargs)
        return cls(**kwargs)

    @classmethod
    def from_json(cls, path: str) -> "Qwen25VLConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        """HF-style dict (``rope_scaling`` holds ``mrope_section``), readable by ``from_dict``."""
        d = asdict(self)
        d["rope_scaling"] = {"type": "mrope", "mrope_section": d.pop("mrope_section")}
        d["model_type"] = "qwen2_5_vl"
        return d

    @classmethod
    def tiny(cls) -> "Qwen25VLConfig":
        """A CPU-sized configuration for tests (head_dim 16, so mrope_section sums to 8)."""
        return cls(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=512,
            mrope_section=[2, 3, 3],
            max_position_embeddings=1024,
            tie_word_embeddings=False,
            torch_dtype="float32",
            # token ids must be < vocab_size
            image_token_id=500,
            video_token_id=501,
            vision_start_token_id=502,
            vision_end_token_id=503,
            vision_config=VisionConfig(
                depth=2,
                hidden_size=32,
                intermediate_size=64,
                num_heads=2,
                out_hidden_size=64,
                fullatt_block_indexes=[1],
                window_size=112,
            ),
        )


# ----------------------------------------------------------------------------------------------
# Shared building blocks
# ----------------------------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Qwen2 RMSNorm: normalise in fp32, scale by ``weight`` in the input dtype."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.to(x.dtype)


class SwiGLU(nn.Module):
    """gate/up/down MLP shared by the decoder (no bias) and the vision blocks (bias)."""

    def __init__(self, dim: int, hidden: int, bias: bool) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=bias)
        self.up_proj = nn.Linear(dim, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, n_kv, T, D) -> (B, n_kv * n_rep, T, D) so GQA keys/values line up with query heads."""
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, t, d).reshape(b, n_kv * n_rep, t, d)


def rope_inv_freq(dim: int, theta: float, device: torch.device) -> torch.Tensor:
    """Standard RoPE inverse frequencies, ``dim // 2`` of them. Recomputed on demand (it is tiny)
    so the model has no persistent buffers and can be built on the meta device for loading."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))


# ----------------------------------------------------------------------------------------------
# Vision tower (HF module path: ``visual``)
# ----------------------------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Conv3d over (temporal_patch_size, patch_size, patch_size), applied to flattened patches."""

    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        k = (cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size)
        self.proj = nn.Conv3d(cfg.in_channels, cfg.hidden_size, kernel_size=k, stride=k, bias=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        x = pixel_values.view(-1, c.in_channels, c.temporal_patch_size, c.patch_size, c.patch_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, c.hidden_size)


class VisionRotaryEmbedding(nn.Module):
    """2D axial RoPE table: ``forward(n)`` gives ``(n, dim // 2)`` angles, ``dim = head_dim // 2``,
    so that h-angles and w-angles concatenated cover half the head and ``cat((emb, emb))`` all of it."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, seqlen: int, device: torch.device) -> torch.Tensor:
        inv_freq = rope_inv_freq(self.dim, self.theta, device)
        return torch.outer(torch.arange(seqlen, device=device, dtype=torch.float32), inv_freq)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    """q, k: (N, heads, head_dim); cos, sin: (N, head_dim). Computed in fp32 as HF does."""
    cos, sin = cos[:, None].float(), sin[:, None].float()
    qf, kf = q.float(), k.float()
    q_out = qf * cos + rotate_half(qf) * sin
    k_out = kf * cos + rotate_half(kf) * sin
    return q_out.to(q.dtype), k_out.to(k.dtype)


class VisionAttention(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.hidden_size, cfg.hidden_size * 3, bias=True)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x: torch.Tensor, cu_seqlens: list[int], cos: torch.Tensor, sin: torch.Tensor):
        n = x.shape[0]
        q, k, v = self.qkv(x).view(n, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3).unbind(0)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))  # (heads, N, head_dim)
        # Block-diagonal attention: each [start, end) segment of cu_seqlens attends only to itself.
        # A loop over segments (windows or whole images) keeps this plain SDPA.
        outs = [
            F.scaled_dot_product_attention(q[:, s:e], k[:, s:e], v[:, s:e])
            for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:])
        ]
        out = torch.cat(outs, dim=1).transpose(0, 1).reshape(n, -1)
        return self.proj(out)


class VisionBlock(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(cfg.hidden_size, eps=1e-6)
        self.attn = VisionAttention(cfg)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size, bias=True)

    def forward(self, x, cu_seqlens, cos, sin):
        x = x + self.attn(self.norm1(x), cu_seqlens, cos, sin)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    """RMSNorm -> group merge^2 neighbouring patches -> MLP to the LM hidden size."""

    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int) -> None:
        super().__init__()
        self.hidden_size = context_dim * spatial_merge_size**2
        self.ln_q = RMSNorm(context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.ln_q(x).view(-1, self.hidden_size))


class VisionTransformer(nn.Module):
    def __init__(self, cfg: VisionConfig) -> None:
        super().__init__()
        if cfg.hidden_act != "silu":
            raise ValueError(f"only silu is implemented for the vision MLP, got {cfg.hidden_act}")
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.rotary_pos_emb = VisionRotaryEmbedding(cfg.head_dim // 2)
        self.blocks = nn.ModuleList(VisionBlock(cfg) for _ in range(cfg.depth))
        self.merger = PatchMerger(cfg.out_hidden_size, cfg.hidden_size, cfg.spatial_merge_size)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """(h, w) coordinates of every patch, ordered so that each merge x merge block is contiguous
        (the order ``image_preprocess`` emits patches in), looked up in the rotary table."""
        m = self.cfg.spatial_merge_size
        device = grid_thw.device
        pos_ids = []
        for t, h, w in grid_thw.tolist():
            hpos = torch.arange(h, device=device)[:, None].expand(h, w)
            wpos = torch.arange(w, device=device)[None, :].expand(h, w)
            hpos = hpos.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            wpos = wpos.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids)
        table = self.rotary_pos_emb(int(grid_thw[:, 1:].max()), device)
        return table[pos_ids].flatten(1)  # (N, head_dim // 2): h angles then w angles

    def get_window_index(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """Permutation of merged-token units that groups them into window_size x window_size pixel
        windows, plus the cumulative patch counts of those windows (as in HF)."""
        c = self.cfg
        vit_win = c.window_size // c.spatial_merge_size // c.patch_size  # merged tokens per window side
        unit = c.spatial_merge_size**2
        window_index, cu_window_seqlens, offset = [], [0], 0
        for t, h, w in grid_thw.tolist():
            lh, lw = h // c.spatial_merge_size, w // c.spatial_merge_size
            index = torch.arange(t * lh * lw, device=grid_thw.device).reshape(t, lh, lw)
            pad_h = vit_win - lh % vit_win
            pad_w = vit_win - lw % vit_win
            nwh, nww = (lh + pad_h) // vit_win, (lw + pad_w) // vit_win
            index = F.pad(index, (0, pad_w, 0, pad_h), value=-100)
            index = index.reshape(t, nwh, vit_win, nww, vit_win).permute(0, 1, 3, 2, 4)
            index = index.reshape(t, nwh * nww, vit_win, vit_win)
            seqlens = (index != -100).sum([2, 3]).reshape(-1)
            index = index.reshape(-1)
            window_index.append(index[index != -100] + offset)
            cu = seqlens.cumsum(0) * unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu.tolist())
            offset += t * lh * lw
        # Empty (fully padded) windows produce repeated boundaries; drop them as HF does.
        cu_unique = [cu_window_seqlens[0]] + [
            b for a, b in zip(cu_window_seqlens, cu_window_seqlens[1:]) if b != a
        ]
        return torch.cat(window_index), cu_unique

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """pixel_values: (N, C*T*P*P) -> (N / merge^2, out_hidden_size), image order preserved."""
        unit = self.cfg.spatial_merge_size**2
        x = self.patch_embed(pixel_values)
        n = x.shape[0]
        rotary = self.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        # Full attention is per image (per frame): cumulative h*w over the t frames of each image.
        frame_lens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu_seqlens = [0] + frame_lens.cumsum(0).tolist()

        # Reorder patches (in merge-unit groups) into window order; undo after merging.
        x = x.view(n // unit, unit, -1)[window_index].view(n, -1)
        rotary = rotary.view(n // unit, unit, -1)[window_index].view(n, -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        cos, sin = emb.cos(), emb.sin()

        for i, blk in enumerate(self.blocks):
            cu = cu_seqlens if i in self.cfg.fullatt_block_indexes else cu_window_seqlens
            x = blk(x, cu, cos, sin)
        x = self.merger(x)
        return x[torch.argsort(window_index)]


# ----------------------------------------------------------------------------------------------
# Language model (HF module path: ``model`` + ``lm_head``)
# ----------------------------------------------------------------------------------------------


class MRotaryEmbedding(nn.Module):
    """cos/sin for the three M-RoPE axes: ``position_ids (3, B, T) -> (3, B, T, head_dim)``."""

    def __init__(self, dim: int, theta: float) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        inv_freq = rope_inv_freq(self.dim, self.theta, position_ids.device)
        freqs = position_ids[..., None].float() * inv_freq  # (3, B, T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section: list[int], unsqueeze_dim: int = 1):
    """HF ``apply_multimodal_rotary_pos_emb``: split the head dim into ``mrope_section * 2`` chunks
    and take chunk ``i`` from axis ``i % 3`` (t, h, w), i.e. the first 16 frequencies come from the
    temporal axis, the next 24 from height, the next 24 from width, and again for the second half."""
    sections = list(mrope_section) * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(sections, dim=-1))], dim=-1)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(sections, dim=-1))], dim=-1)
    cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Attention(nn.Module):
    def __init__(self, cfg: Qwen25VLConfig) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.mrope_section = cfg.mrope_section
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, attn_mask, past_kv):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.mrope_section)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v)
        n_rep = self.num_heads // self.num_kv_heads
        out = F.scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), attn_mask=attn_mask)
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1)), present


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen25VLConfig) -> None:
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask, past_kv):
        h, present = self.self_attn(self.input_layernorm(x), cos, sin, attn_mask, past_kv)
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, present


def build_attention_mask(attention_mask: torch.Tensor | None, q_len: int, kv_len: int, device) -> torch.Tensor:
    """Boolean SDPA mask ``(B or 1, 1, q_len, kv_len)``, True = may attend. Causal over the full
    (cached + new) key axis, and padding keys removed via the 2D ``attention_mask (B, kv_len)``.
    Query rows that would see no key at all (left-padding positions) are allowed to attend
    everywhere so SDPA does not produce NaNs; their outputs are padding and never used."""
    past = kv_len - q_len
    i = torch.arange(q_len, device=device)[:, None]
    j = torch.arange(kv_len, device=device)[None, :]
    mask = (j <= i + past)[None, None]
    if attention_mask is not None:
        mask = mask & attention_mask.bool()[:, None, None, :]
    return torch.where(mask.any(-1, keepdim=True), mask, torch.ones_like(mask))


class Qwen2Model(nn.Module):
    def __init__(self, cfg: Qwen25VLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = MRotaryEmbedding(cfg.head_dim, cfg.rope_theta)
        self.gradient_checkpointing = False

    def forward(self, inputs_embeds, position_ids, attention_mask=None, past_key_values=None, use_cache=False):
        b, t, _ = inputs_embeds.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[2]
        if attention_mask is not None and attention_mask.shape[-1] != past_len + t:
            raise ValueError(
                f"attention_mask covers {attention_mask.shape[-1]} keys but cache+input has {past_len + t}"
            )
        attn_mask = build_attention_mask(attention_mask, t, past_len + t, inputs_embeds.device)
        cos, sin = self.rotary_emb(position_ids, inputs_embeds.dtype)

        x = inputs_embeds
        new_cache = [] if use_cache else None
        ckpt = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        for i, layer in enumerate(self.layers):
            past_kv = None if past_key_values is None else past_key_values[i]
            if ckpt:
                x, present = checkpoint(layer, x, cos, sin, attn_mask, past_kv, use_reentrant=False)
            else:
                x, present = layer(x, cos, sin, attn_mask, past_kv)
            if use_cache:
                new_cache.append(present)
        return self.norm(x), new_cache


class Qwen25VL(nn.Module):
    """Qwen2.5-VL for conditional generation: ``visual`` + ``model`` + ``lm_head``."""

    def __init__(self, cfg: Qwen25VLConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.visual = VisionTransformer(cfg.vision_config)
        self.model = Qwen2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @property
    def gradient_checkpointing(self) -> bool:
        return self.model.gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value: bool) -> None:
        self.model.gradient_checkpointing = value

    def freeze_vision(self) -> None:
        """Freeze the whole vision tower incl. the patch merger (as EasyR1's ``freeze_vision_tower``)."""
        self.visual.requires_grad_(False)

    # -- position ids ----------------------------------------------------------------------------

    def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
        """3D position ids as in HF ``Qwen2_5_VLModel.get_rope_index`` (image + text only).

        Text tokens advance all three axes by 1. Each run of ``image_token_id`` tokens gets
        ``(t, h, w)`` indices over the ``(grid_t, grid_h // merge, grid_w // merge)`` llm grid, all
        offset by the position the run starts at; the following text continues from ``max + 1``.
        Padding positions (``attention_mask == 0``) are set to 1 as in HF.

        Returns ``position_ids (3, B, T)`` and ``rope_deltas (B, 1)`` = ``max_position + 1 - T``,
        which is what incremental decoding needs to continue the sequence.
        """
        b, t_len = input_ids.shape
        device = input_ids.device
        m = self.cfg.vision_config.spatial_merge_size
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        valid = attention_mask.bool()
        position_ids = torch.ones(3, b, t_len, dtype=torch.long, device=device)
        deltas = []
        image_index = 0
        for bi in range(b):
            tokens = input_ids[bi][valid[bi]].tolist()
            segments, st, cur = [], 0, 0  # cur: next free position
            while True:
                try:
                    ed = tokens.index(self.cfg.image_token_id, st)
                except ValueError:
                    break
                if image_grid_thw is None or image_index >= len(image_grid_thw):
                    raise ValueError("input_ids contain more image token runs than image_grid_thw rows")
                t, h, w = (int(v) for v in image_grid_thw[image_index])
                image_index += 1
                lh, lw = h // m, w // m
                text_len = ed - st
                segments.append(torch.arange(text_len, device=device).expand(3, -1) + cur)
                cur += text_len
                t_idx = torch.arange(t, device=device).view(t, 1, 1).expand(t, lh, lw).flatten()
                h_idx = torch.arange(lh, device=device).view(1, lh, 1).expand(t, lh, lw).flatten()
                w_idx = torch.arange(lw, device=device).view(1, 1, lw).expand(t, lh, lw).flatten()
                segments.append(torch.stack([t_idx, h_idx, w_idx]) + cur)
                cur += max(t, lh, lw)  # == segments[-1].max() + 1
                st = ed + t * lh * lw
            if st < len(tokens):
                segments.append(torch.arange(len(tokens) - st, device=device).expand(3, -1) + cur)
            pos = torch.cat(segments, dim=1)
            position_ids[:, bi, valid[bi]] = pos
            deltas.append(int(pos.max()) + 1 - t_len)
        if image_grid_thw is not None and image_index != len(image_grid_thw):
            raise ValueError(f"{len(image_grid_thw)} images given but {image_index} image runs found")
        return position_ids, torch.tensor(deltas, device=device).view(b, 1)

    # -- forward ---------------------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        """Returns ``(logits (B, T, vocab), past_key_values or None)``."""
        x = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            feats = self.visual(pixel_values, image_grid_thw).to(x.dtype)
            image_mask = input_ids == self.cfg.image_token_id
            n_tokens = int(image_mask.sum())
            if n_tokens != feats.shape[0]:
                raise ValueError(f"{n_tokens} image tokens in input_ids but {feats.shape[0]} image features")
            x = x.masked_scatter(image_mask[..., None].expand_as(x), feats)
        if position_ids is None:
            if past_key_values is not None:
                raise ValueError("position_ids are required when past_key_values is given")
            position_ids, _ = self.get_rope_index(input_ids, image_grid_thw, attention_mask)
        h, cache = self.model(x, position_ids, attention_mask, past_key_values, use_cache)
        return self.lm_head(h), cache

    # -- sampling --------------------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prefill with a KV cache, then one token per step.

        Returns ``response_ids (B, max_new_tokens)`` -- the EOS token is kept, everything after it is
        ``pad_token_id`` -- and ``finished (B,)``, True where an EOS was produced. Position ids for
        step ``g`` are ``max(prompt positions) + 1 + g`` on all three axes (HF/vLLM behaviour).
        """
        b, t_len = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if pad_token_id is None:
            raise ValueError("pad_token_id is required")
        eos_ids = [] if eos_token_id is None else (
            [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
        )
        eos = torch.tensor(eos_ids, device=device, dtype=input_ids.dtype)

        position_ids, deltas = self.get_rope_index(input_ids, image_grid_thw, attention_mask)
        logits, cache = self.forward(
            input_ids, attention_mask, position_ids, pixel_values, image_grid_thw, use_cache=True
        )
        next_pos = (deltas + t_len).view(1, b, 1).expand(3, b, 1)  # max prompt position + 1

        response = torch.full((b, max_new_tokens), pad_token_id, dtype=input_ids.dtype, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)
        for step in range(max_new_tokens):
            tok = sample_token(logits[:, -1].float(), temperature, top_p, top_k, generator)
            tok = torch.where(finished, torch.full_like(tok, pad_token_id), tok)
            response[:, step] = tok
            finished |= torch.isin(tok, eos)
            if bool(finished.all()) or step == max_new_tokens - 1:
                break
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(b, 1)], dim=1)
            logits, cache = self.forward(
                tok[:, None], attention_mask, next_pos, past_key_values=cache, use_cache=True
            )
            next_pos = next_pos + 1
        return response, finished


def sample_token(logits, temperature, top_p, top_k, generator=None) -> torch.Tensor:
    """Temperature / top-k / top-p sampling over ``logits (B, V)``; ``temperature == 0`` is greedy."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        # drop tokens once the mass *before* them already exceeds top_p (keeps at least one)
        remove = probs.cumsum(dim=-1) - probs > top_p
        logits = logits.masked_fill(remove.scatter(1, order, remove), float("-inf"))
    return torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).squeeze(-1)


# ----------------------------------------------------------------------------------------------
# Checkpoint loading
# ----------------------------------------------------------------------------------------------

# transformers>=5 saves Qwen2.5-VL with these prefixes; our modules use the transformers<5 names.
_KEY_RENAMES = (("model.language_model.", "model."), ("model.visual.", "visual."))


def _rename_key(key: str) -> str:
    for old, new in _KEY_RENAMES:
        if key.startswith(old):
            return new + key[len(old):]
    return key


def load_pretrained(model_dir: str, device: str | torch.device = "cpu", dtype: torch.dtype | None = None) -> Qwen25VL:
    """Build ``Qwen25VL`` from a Hub-style directory (``config.json`` + ``*.safetensors``).

    ``model_dir`` may also be a Hub repo id, in which case ``huggingface_hub.snapshot_download``
    fetches it. Weights are loaded strictly: missing or unexpected keys raise. ``dtype`` defaults
    to ``torch_dtype`` from config.json.
    """
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download  # optional dependency

        model_dir = snapshot_download(model_dir, allow_patterns=["*.json", "*.safetensors"])
    from safetensors.torch import load_file

    cfg = Qwen25VLConfig.from_json(os.path.join(model_dir, "config.json"))
    dtype = dtype or getattr(torch, cfg.torch_dtype)

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            shards = sorted({os.path.join(model_dir, v) for v in json.load(f)["weight_map"].values()})
    else:
        shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors in {model_dir}")

    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        for k, v in load_file(shard, device=str(device)).items():
            state[_rename_key(k)] = v.to(dtype)

    # Build on the meta device (no random init of billions of weights), materialise, then copy in.
    with torch.device("meta"):
        model = Qwen25VL(cfg)
    model.to_empty(device=device).to(dtype)
    if cfg.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight  # re-tie after to_empty
        state.setdefault("lm_head.weight", state.get("model.embed_tokens.weight"))

    expected = set(model.state_dict().keys())
    missing, unexpected = sorted(expected - state.keys()), sorted(state.keys() - expected)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model key mismatch.\n  missing ({len(missing)}): {missing[:10]}"
            f"\n  unexpected ({len(unexpected)}): {unexpected[:10]}"
        )
    model.load_state_dict(state, strict=True)
    return model.eval()


# ----------------------------------------------------------------------------------------------
# Image preprocessing (HF Qwen2VLImageProcessor, images only)
# ----------------------------------------------------------------------------------------------

OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def smart_resize(height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280):
    """Target (h, w): multiples of ``factor``, pixel count within [min, max], aspect ratio kept."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"aspect ratio must be < 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_preprocess(
    pil_image,
    min_pixels: int = 56 * 56,
    max_pixels: int = 28 * 28 * 1280,
    patch_size: int = 14,
    merge: int = 2,
    temporal_patch_size: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PIL image -> ``(pixel_values (grid_h*grid_w, C*T*P*P) float32, grid_thw (1, 3) long)``.

    Patch order is (h_block, w_block, merge_h, merge_w); within a patch the layout is
    (channel, temporal, patch_h, patch_w). The single frame is repeated ``temporal_patch_size``
    times, exactly like the HF processor does for still images.
    """
    from PIL import Image

    image = pil_image.convert("RGB")
    h, w = smart_resize(image.height, image.width, patch_size * merge, min_pixels, max_pixels)
    image = image.resize((w, h), resample=Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, C)
    arr = (arr - np.array(OPENAI_CLIP_MEAN, dtype=np.float32)) / np.array(OPENAI_CLIP_STD, dtype=np.float32)
    arr = arr.transpose(2, 0, 1)  # (C, H, W)
    frames = np.repeat(arr[None], temporal_patch_size, axis=0)  # (T, C, H, W)

    grid_t, grid_h, grid_w = 1, h // patch_size, w // patch_size
    c = arr.shape[0]
    patches = frames.reshape(
        grid_t, temporal_patch_size, c,
        grid_h // merge, merge, patch_size,
        grid_w // merge, merge, patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    pixel_values = patches.reshape(grid_t * grid_h * grid_w, c * temporal_patch_size * patch_size * patch_size)
    return torch.from_numpy(np.ascontiguousarray(pixel_values)), torch.tensor([[grid_t, grid_h, grid_w]])
