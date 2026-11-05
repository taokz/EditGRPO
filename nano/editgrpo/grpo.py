"""GRPO advantages, clipped policy loss, KL and loss aggregation — exactly what EditGRPO's actor computes.

Sources (EasyR1/verl fork in this repo):
  advantages   verl/trainer/core_algos.py:128-181   compute_grpo_outcome_advantage
  policy loss  verl/trainer/core_algos.py:339-394   compute_policy_loss  (clip_ratio=0.2, workers/actor/config.py:75)
  aggregation  verl/trainer/core_algos.py:302-337   agg_loss
  KL           verl/trainer/core_algos.py:434-465   kl_penalty ("low_var_kl" = k3, clamped to [-10, 10])
  combination  verl/workers/actor/dp_actor.py:254-279  loss = agg(pg) + kl_coef * agg(kl); no entropy term
Matching RL2 code (taokz/RL2): RL2/utils/algorithms.py:51-74 (advantage, `norm_var`), :180-196 (clipped
surrogate), :23-24 (k3 without clamp), RL2/utils/functions.py:117-146 (`token`/`sequence` aggregation; RL2 has
no constant-normalised token-sum, which is the paper's `seq-mean-token-sum-norm`).

Paper run (examples/mimic_cxr_editgrpo_ops1_para_half_tau_0_6.sbatch:117-118): norm_adv_by_std=False,
loss_agg_mode="seq-mean-token-sum-norm" (Dr. GRPO), low_var_kl with kl_coef=1e-2 (yaml:20-21), clip 0.2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch


@dataclass
class GRPOConfig:
    clip_ratio: float = 0.2
    kl_coef: float = 1e-2
    kl_penalty: str = "low_var_kl"           # kl | abs | mse | low_var_kl
    use_kl_loss: bool = True
    norm_adv_by_std: bool = False            # False = paper sbatch (Dr. GRPO); True = yaml default
    loss_agg_mode: str = "seq-mean-token-sum-norm"  # paper sbatch; yaml default is "token-mean"
    adv_eps: float = 1e-6                    # core_algos.py:133


def group_advantages(scores: torch.Tensor, group_ids: Sequence, norm_by_std: bool = False, eps: float = 1e-6) -> torch.Tensor:
    """core_algos.py:156-178. `scores`: (N,) sequence-level rewards; `group_ids`: hashable per row (uid).
    Group of size 1 -> mean 0, std 1 (:166-167). std is torch.std (unbiased) (:170)."""
    scores = scores.float()
    groups: Dict = {}
    for i, g in enumerate(group_ids):
        groups.setdefault(g, []).append(i)
    adv = torch.empty_like(scores)
    for idx in groups.values():
        vals = scores[idx]
        if len(idx) == 1:
            mean, std = torch.tensor(0.0), torch.tensor(1.0)
        else:
            mean, std = vals.mean(), vals.std()
        adv[idx] = (vals - mean) / (std + eps) if norm_by_std else vals - mean
    return adv


def broadcast_advantages(adv: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """core_algos.py:180: same advantage on every response token."""
    return adv.unsqueeze(-1) * response_mask


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """verl/utils/torch_functional.py:69-71"""
    return (x * mask).sum() / (mask.sum() + 1e-8)


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """core_algos.py:302-337. `loss_mask.shape[-1]` is the padded response width (= max_response_length)."""
    if mode == "token-mean":
        return masked_mean(loss_mat, loss_mask)
    if mode == "seq-mean-token-sum":
        return torch.sum(loss_mat * loss_mask, dim=-1).mean()
    if mode == "seq-mean-token-mean":
        return (torch.sum(loss_mat * loss_mask, dim=-1) / (loss_mask.sum(dim=-1) + 1e-8)).mean()
    if mode == "seq-mean-token-sum-norm":
        return torch.sum(torch.sum(loss_mat * loss_mask, dim=-1)) / loss_mask.shape[-1]
    raise ValueError(f"unknown loss_agg_mode {mode}")


def policy_loss(log_probs: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor,
                response_mask: torch.Tensor, clip_ratio: float = 0.2, loss_agg_mode: str = "seq-mean-token-sum-norm"):
    """core_algos.py:339-394. Clipping is done in log space (:381), equivalent to clamping the ratio.
    Returns (loss, metrics)."""
    neg_approx_kl = log_probs - old_log_probs
    ratio = torch.exp(neg_approx_kl)
    clipped_ratio = torch.exp(torch.clamp(neg_approx_kl, torch.log(torch.tensor(1.0 - clip_ratio)), torch.log(torch.tensor(1.0 + clip_ratio))))
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * clipped_ratio
    per_token = torch.max(pg_losses, pg_losses2)
    loss = agg_loss(per_token, response_mask, loss_agg_mode)
    metrics = {
        "pg_loss": loss.detach(),
        "pg_clipfrac": masked_mean((pg_losses2 > pg_losses).float(), response_mask).detach(),
        "ppo_kl": masked_mean(-neg_approx_kl, response_mask).detach(),
    }
    return loss, metrics


def kl_penalty(log_probs: torch.Tensor, ref_log_probs: torch.Tensor, kind: str = "low_var_kl") -> torch.Tensor:
    """core_algos.py:434-465 (per token)."""
    if kind == "kl":
        return log_probs - ref_log_probs
    if kind == "abs":
        return (log_probs - ref_log_probs).abs()
    if kind == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()
    if kind == "low_var_kl":                          # k3 estimator, :457-460
        kl = ref_log_probs - log_probs
        kld = kl.exp() - kl - 1
        return torch.clamp(kld, min=-10, max=10)
    raise ValueError(f"unknown kl_penalty {kind}")


def editgrpo_loss(log_probs, old_log_probs, ref_log_probs, advantages, response_mask, cfg: GRPOConfig = GRPOConfig()):
    """dp_actor.py:254-279: pg loss (+ kl_coef * KL aggregated with the same mode). `advantages` is per token."""
    loss, metrics = policy_loss(log_probs, old_log_probs, advantages, response_mask, cfg.clip_ratio, cfg.loss_agg_mode)
    if cfg.use_kl_loss and ref_log_probs is not None:
        kld = kl_penalty(log_probs, ref_log_probs, cfg.kl_penalty)
        kl_loss = agg_loss(kld, response_mask, cfg.loss_agg_mode)
        loss = loss + kl_loss * cfg.kl_coef
        metrics["kl_loss"] = kl_loss.detach()
    return loss, metrics
