"""Hand-computed checks for nano/editgrpo/grpo.py against the formulas in verl/trainer/core_algos.py."""
import math
import torch
from editgrpo.grpo import GRPOConfig, agg_loss, broadcast_advantages, editgrpo_loss, group_advantages, kl_penalty, policy_loss


def test_group_advantages_paper_and_yaml_variants():
    scores = torch.tensor([1.0, 2.0, 3.0, 0.5, 0.5])
    ids = ["a", "a", "a", "b", "c"]
    adv = group_advantages(scores, ids, norm_by_std=False)
    # core_algos.py:166-167 sets mean=0, std=1 for a group of size 1, so a singleton's advantage is its raw score.
    assert torch.allclose(adv, torch.tensor([-1.0, 0.0, 1.0, 0.5, 0.5]))
    assert adv[3].item() == 0.5 and adv[4].item() == 0.5
    adv_n = group_advantages(scores, ids, norm_by_std=True)
    std = torch.tensor([1.0, 2.0, 3.0]).std().item()                         # unbiased: 1.0
    assert math.isclose(adv_n[0].item(), -1.0 / (std + 1e-6), rel_tol=1e-6)
    assert math.isclose(adv_n[3].item(), 0.5 / (1.0 + 1e-6), rel_tol=1e-6)


def test_agg_modes_hand_values():
    loss = torch.tensor([[1.0, 2.0, 3.0, 9.0], [4.0, 9.0, 9.0, 9.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    assert math.isclose(agg_loss(loss, mask, "token-mean").item(), 10 / 4, rel_tol=1e-6)
    assert math.isclose(agg_loss(loss, mask, "seq-mean-token-sum").item(), (6 + 4) / 2, rel_tol=1e-6)
    assert math.isclose(agg_loss(loss, mask, "seq-mean-token-mean").item(), (2 + 4) / 2, rel_tol=1e-5)
    assert math.isclose(agg_loss(loss, mask, "seq-mean-token-sum-norm").item(), (6 + 4) / 4, rel_tol=1e-6)  # / padded width


def test_policy_loss_clipping_and_ratio_one():
    adv = torch.tensor([[1.0, -1.0, 2.0]])
    mask = torch.ones(1, 3)
    old = torch.log(torch.tensor([[0.5, 0.5, 0.5]]))
    # ratio = 1 exactly (first update after recomputing old log-probs, as EditGRPO does): loss = -A
    loss, m = policy_loss(old.clone(), old, adv, mask, 0.2, "token-mean")
    assert math.isclose(loss.item(), -(1 - 1 + 2) / 3, rel_tol=1e-6) and m["pg_clipfrac"].item() == 0.0
    # ratios 1.5 (A>0 -> clipped to 1.2), 0.5 (A<0 -> clipped to 0.8), 1.0
    new = torch.log(torch.tensor([[0.75, 0.25, 0.5]]))
    loss, m = policy_loss(new, old, adv, mask, 0.2, "seq-mean-token-sum")
    expected = -(1.0 * 1.2) + (-(-1.0) * 0.8) * 1 + -(2.0 * 1.0)   # max(-A r, -A clip(r)) per token
    # token1: max(-1.5, -1.2) = -1.2; token2: max(0.5, 0.8) = 0.8; token3: -2  -> sum = -2.4
    assert math.isclose(loss.item(), -1.2 + 0.8 - 2.0, rel_tol=1e-5)
    assert math.isclose(m["pg_clipfrac"].item(), 2 / 3, rel_tol=1e-6)


def test_low_var_kl_and_clamp():
    lp = torch.tensor([0.0, -5.0]); ref = torch.tensor([0.0, 10.0])
    k = kl_penalty(lp, ref, "low_var_kl")
    assert k[0].item() == 0.0 and k[1].item() == 10.0        # exp(15)-15-1 clamped to 10
    assert torch.allclose(kl_penalty(lp, ref, "kl"), lp - ref)
    assert torch.allclose(kl_penalty(lp, ref, "mse"), 0.5 * (lp - ref) ** 2)


def test_editgrpo_loss_combines_pg_and_kl_with_same_agg():
    mask = torch.ones(2, 4)
    adv = broadcast_advantages(torch.tensor([1.0, -1.0]), mask)
    lp = torch.full((2, 4), -1.0, requires_grad=True)
    cfg = GRPOConfig()  # paper defaults
    loss, m = editgrpo_loss(lp, lp.detach(), lp.detach() - 0.1, adv, mask, cfg)
    pg = agg_loss(-adv, mask, cfg.loss_agg_mode)
    kl = agg_loss(kl_penalty(lp, lp.detach() - 0.1, "low_var_kl"), mask, cfg.loss_agg_mode)
    assert math.isclose(loss.item(), (pg + 1e-2 * kl).item(), rel_tol=1e-6)
    loss.backward(); assert torch.isfinite(lp.grad).all()
