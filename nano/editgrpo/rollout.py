"""Group rollouts + post-rollout edits + re-tokenisation, i.e. the parts of
verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:241-1220 that are algorithm rather than vLLM plumbing.

`Sampler` is the only thing a fast backend (nano-vllm, vLLM) has to implement; `TorchSampler` uses the
pure-PyTorch model's KV-cache `generate`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

import torch

from .data import encode_ids
from .edit import EditConfig, EntityExtractor, clean_eos_tokens, edit_group


@dataclass
class RolloutConfig:
    """verl/workers/rollout/config.py:25-28 + examples/editgrpo_mimic_cxr_2_5.yaml:53-58."""
    n: int = 5
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    response_length: int = 4096
    edited_append_eos: bool = False   # ISSUES.md #1; False = original (edited rows carry no EOS)


class Sampler(Protocol):
    def __call__(self, batch: Dict[str, Any], n: int, cfg: RolloutConfig, generator: Optional[torch.Generator]) -> torch.Tensor:
        """Return response_ids of shape (B*n, response_length), prompt-major (row i*n+k is sample k of prompt i),
        right-padded with pad_token_id after the first EOS."""
        ...


def repeat_interleave(x, n: int):
    """vllm_rollout_spmd.py:46-51"""
    if isinstance(x, torch.Tensor):
        return x.repeat_interleave(n, dim=0)
    return [y for y in x for _ in range(n)]


class TorchSampler:
    """Own sampler on top of model.generate (nano/editgrpo/model.py). Prompts are repeated n times, like
    vLLM's SamplingParams(n=...) followed by `_repeat_interleave` (vllm_rollout_spmd.py:1061-1069)."""

    def __init__(self, model, eos_token_id: int, pad_token_id: int):
        self.model, self.eos, self.pad = model, eos_token_id, pad_token_id

    @torch.no_grad()
    def __call__(self, batch, n, cfg, generator=None):
        ids = repeat_interleave(batch["input_ids"], n)
        am = repeat_interleave(batch["attention_mask"], n)
        pv, grid = batch.get("pixel_values"), batch.get("image_grid_thw")
        if pv is not None and grid is not None and grid.numel():
            # pixel rows are per image; images belong to prompts in order -> repeat per prompt
            per_prompt = []
            counts = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
            img_of_prompt = _images_per_prompt(batch["input_ids"], self.model.config.image_token_id, grid)
            off_img = off_pix = 0
            for b in range(batch["input_ids"].shape[0]):
                k = img_of_prompt[b]
                npix = sum(counts[off_img:off_img + k])
                per_prompt.append((pv[off_pix:off_pix + npix], grid[off_img:off_img + k]))
                off_img += k; off_pix += npix
            pv = torch.cat([p for p, _ in per_prompt for _ in range(n)], 0)
            grid = torch.cat([g for _, g in per_prompt for _ in range(n)], 0)
        resp, _ = self.model.generate(ids, am, pixel_values=pv, image_grid_thw=grid, max_new_tokens=cfg.response_length,
                                      temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
                                      eos_token_id=self.eos, pad_token_id=self.pad, generator=generator)
        return resp


def _images_per_prompt(input_ids: torch.Tensor, image_token_id: int, grid: torch.Tensor) -> List[int]:
    """How many images each prompt holds (count vision_start-delimited image spans via token runs)."""
    counts = []
    for row in input_ids:
        is_img = (row == image_token_id).long()
        starts = ((is_img[1:] == 1) & (is_img[:-1] == 0)).sum().item() + int(is_img[0].item() == 1)
        counts.append(int(starts))
    return counts


def get_eos_mask(response_ids: torch.Tensor, eos_token_id: int, dtype=torch.long) -> torch.Tensor:
    """verl/utils/torch_functional.py:105-117: 1 up to and including the first EOS, 0 after."""
    eos = response_ids.eq(eos_token_id).long()
    after = (torch.cumsum(eos, dim=1) - eos).bool()
    return torch.logical_not(after).to(dtype)


def retokenize_edited(text: str, tokenizer, response_length: int, eos_token_id: int, pad_token_id: int,
                      append_eos: bool = False) -> torch.Tensor:
    """vllm_rollout_spmd.py:116-149 + 988-999: clean, encode without specials, drop EOS ids, truncate, right-pad.
    `append_eos=True` is the ISSUES.md #1 fix (off by default)."""
    text = clean_eos_tokens(str(text).strip())
    toks = [t for t in encode_ids(tokenizer, text) if t != eos_token_id]
    if append_eos:
        toks = toks[: response_length - 1] + [eos_token_id]
    else:
        toks = toks[:response_length]
    out = torch.full((response_length,), pad_token_id, dtype=torch.long)
    out[: len(toks)] = torch.tensor(toks, dtype=torch.long)
    return out


def decode_responses(response_ids: torch.Tensor, tokenizer, pad_token_id: int, special_ids: Sequence[int]) -> List[str]:
    """vllm_rollout_spmd.py:439-446: drop pads, decode with skip_special_tokens, clean EOS strings."""
    texts = []
    skip = set(special_ids) | {pad_token_id}
    for row in response_ids.tolist():
        texts.append(clean_eos_tokens(tokenizer.decode([t for t in row if t not in skip])))
    return texts


def rollout_with_edits(sampler: Sampler, batch: Dict[str, Any], tokenizer, cfg: RolloutConfig, eos_token_id: int,
                       pad_token_id: int, special_ids: Sequence[int], extractor: Optional[EntityExtractor] = None,
                       edit_cfg: EditConfig = EditConfig(), rng: Optional[random.Random] = None,
                       generator: Optional[torch.Generator] = None) -> Dict[str, Any]:
    """One EditGRPO rollout step for a batch of prompts. Returns prompt-major tensors of size B*n."""
    n = cfg.n
    response_ids = sampler(batch, n, cfg, generator).clone()
    texts = decode_responses(response_ids, tokenizer, pad_token_id, special_ids)
    gts = repeat_interleave(list(batch["ground_truth"]), n)
    records: List[Dict] = []
    if extractor is not None:
        new_texts, records = edit_group(texts, gts, n, extractor, edit_cfg, rng)
        for i, (o, m) in enumerate(zip(texts, new_texts)):
            if o != m:                                                   # :975-1004 only changed rows are replaced
                response_ids[i] = retokenize_edited(m, tokenizer, cfg.response_length, eos_token_id, pad_token_id,
                                                    cfg.edited_append_eos)
        texts = new_texts
    response_mask = get_eos_mask(response_ids, eos_token_id)              # :1197-1199
    out = {
        "prompts": repeat_interleave(batch["input_ids"], n),
        "prompt_mask": repeat_interleave(batch["attention_mask"], n),
        "responses": response_ids,
        "response_mask": response_mask,
        "texts": texts,
        "ground_truth": gts,
        "edits": records,
        "group_ids": [i // n for i in range(response_ids.shape[0])],
    }
    out["input_ids"] = torch.cat([out["prompts"], response_ids], dim=1)
    out["attention_mask"] = torch.cat([out["prompt_mask"], response_mask], dim=1)
    return out
