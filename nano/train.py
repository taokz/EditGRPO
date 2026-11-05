"""nano-EditGRPO training loop (nanochat scripts/chat_rl.py style, torchrun-ready).

One step = sample G rollouts per prompt -> edit G//2 of them toward the gold report -> clinical rewards ->
group-mean advantages -> recompute old/ref log-probs on the (edited) tokens -> clipped surrogate + low-var KL.
Defaults reproduce examples/mimic_cxr_editgrpo_ops1_para_half_tau_0_6.sbatch; see nano/README.md.

CPU smoke:   python nano/train.py --synthetic --steps 2
Real run:    torchrun --nproc_per_node 4 nano/train.py --model /path/sft-ckpt --train_json data.json --reward clinical
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from editgrpo.data import (ByteFallbackTokenizer, SyntheticCXRDataset, collate_left_pad, load_records,
                           load_tokenizer, make_example, pad_token_id, IM_END)
from editgrpo.edit import DEFAULT_LEXICON, EditConfig, LexiconExtractor
from editgrpo.eval import repetition_stats
from editgrpo.grpo import GRPOConfig, broadcast_advantages, editgrpo_loss, group_advantages
from editgrpo.model import Qwen25VL, Qwen25VLConfig, image_preprocess, load_pretrained
from editgrpo.reward import make_reward
from editgrpo.rollout import RolloutConfig, TorchSampler, rollout_with_edits


def parse_args():
    p = argparse.ArgumentParser()
    # data / model
    p.add_argument("--model", default=None, help="SFT checkpoint dir (config.json + safetensors + tokenizer.json)")
    p.add_argument("--train_json", default=None)
    p.add_argument("--synthetic", action="store_true", help="tiny random model + synthetic reports (CPU test)")
    p.add_argument("--min_pixels", type=int, default=262144)
    p.add_argument("--max_pixels", type=int, default=401408)
    p.add_argument("--max_prompt_length", type=int, default=4096)
    p.add_argument("--freeze_vision_tower", type=int, default=1)       # sbatch:94
    # rollout (yaml:53-58, sbatch:98-103)
    p.add_argument("--rollout_batch_size", type=int, default=32)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max_response_length", type=int, default=4096)
    p.add_argument("--edit", type=int, default=1, help="apply post-rollout edits (extractor=True)")
    p.add_argument("--extractor", default="lexicon", choices=["lexicon", "rate"])
    p.add_argument("--num_ops_per_para", type=int, default=1)
    p.add_argument("--num_para", type=int, default=-1)
    p.add_argument("--tau", type=float, default=0.6)
    p.add_argument("--fix_tau_scaling", type=int, default=0)
    p.add_argument("--edited_append_eos", type=int, default=0)
    # reward (sbatch:105-108)
    p.add_argument("--reward", default="cheap", choices=["cheap", "clinical"])
    p.add_argument("--reward_components", default="radgraphf1,cxb14micro,ratescore")
    p.add_argument("--reward_device", default="cpu")
    # algorithm (yaml:17-24 overridden by sbatch:117-118) and optimiser (workers/actor/config.py:38-76)
    p.add_argument("--norm_adv_by_std", type=int, default=0)
    p.add_argument("--loss_agg_mode", default="seq-mean-token-sum-norm")
    p.add_argument("--kl_coef", type=float, default=1e-2)
    p.add_argument("--kl_penalty", default="low_var_kl")
    p.add_argument("--clip_ratio", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--ppo_epochs", type=int, default=1)
    p.add_argument("--micro_batch_size", type=int, default=2)
    p.add_argument("--steps", type=int, default=-1, help="-1: one episode over the data (total_episodes=1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--log_every", type=int, default=1)
    return p.parse_args()


def setup_dist():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        rank, world = 0, 1
    return rank, world


def sequence_log_probs(model, input_ids, attention_mask, responses, pixel_values, image_grid_thw, temperature):
    """dp_actor.py:169-210 — per-token log-probs of the response tokens; logits divided by temperature (:126,149).
    A sampled <|image_pad|> inside the response (only plausible for the tiny random test model) would be mistaken for
    an image slot, so those positions are fed as pad while the log-prob is still taken for the sampled id."""
    L = responses.shape[1]
    cfg = model.module.config if hasattr(model, "module") else model.config
    feed = input_ids.clone()
    tail = feed[:, -L:]
    tail[tail == cfg.image_token_id] = cfg.vision_end_token_id
    logits, _ = model(feed, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    logits = logits[:, -L - 1:-1, :].float() / temperature
    return torch.gather(torch.log_softmax(logits, dim=-1), 2, responses.unsqueeze(-1)).squeeze(-1)


def main():
    args = parse_args()
    rank, world = setup_dist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank); random.seed(args.seed + rank)
    rng = random.Random(args.seed + rank)                     # paragraph_rng (vllm_rollout_spmd.py)
    gen = torch.Generator(device="cpu").manual_seed(args.seed + rank)
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    # ---- model, tokenizer, data
    if args.synthetic:
        tok = ByteFallbackTokenizer()
        cfg = Qwen25VLConfig.tiny(); cfg.vocab_size = tok.vocab_size
        cfg.image_token_id = tok.token_to_id("<|image_pad|>"); cfg.vision_start_token_id = tok.token_to_id("<|vision_start|>")
        cfg.vision_end_token_id = tok.token_to_id("<|vision_end|>")
        model = Qwen25VL(cfg)
        records = SyntheticCXRDataset(n=max(4, args.rollout_batch_size), seed=args.seed).records
        args.min_pixels, args.max_pixels = 28 * 28, 56 * 56
        args.max_response_length = min(args.max_response_length, 48)
    else:
        assert args.model and args.train_json, "--model and --train_json are required (or --synthetic)"
        tok = load_tokenizer(os.path.join(args.model, "tokenizer.json"))
        model = load_pretrained(args.model)
        records = load_records(args.train_json)
    eos_id = tok.token_to_id(IM_END)                           # Qwen2.5 eos = <|im_end|>, pad = <|endoftext|>
    pad_id = pad_token_id(tok)
    special_ids = [i for i in (tok.token_to_id(s) for s in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>",
                                                             "<|vision_end|>", "<|image_pad|>"]) if i is not None]
    model.to(device).train()
    if args.freeze_vision_tower:
        model.freeze_vision()                                  # fsdp_workers.py:231-233
    model.gradient_checkpointing = True                        # yaml:37
    ref = copy.deepcopy(model).eval().requires_grad_(False)    # frozen SFT reference for the KL term
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    raw = model.module if world > 1 else model
    sampler = TorchSampler(raw, eos_id, pad_id)
    opt = torch.optim.AdamW([p for p in raw.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=args.weight_decay)   # constant LR, warmup 0 (fsdp_workers.py:290-309)

    # ---- edit / reward / algorithm config
    edit_cfg = EditConfig(num_ops_per_para=args.num_ops_per_para, num_para=args.num_para, similarity_threshold=args.tau,
                          fix_tau_scaling=bool(args.fix_tau_scaling))
    extractor = None
    if args.edit:
        if args.extractor == "lexicon":
            extractor = LexiconExtractor(DEFAULT_LEXICON)
        else:
            from editgrpo.plugins.rate_extractor import RaTEExtractor
            extractor = RaTEExtractor(device=args.reward_device)
    reward = make_reward(args.reward_components.split(","), response_type="free_form", device=args.reward_device,
                         cheap=(args.reward == "cheap"))
    rcfg = RolloutConfig(n=args.n, temperature=args.temperature, response_length=args.max_response_length,
                         edited_append_eos=bool(args.edited_append_eos))
    gcfg = GRPOConfig(clip_ratio=args.clip_ratio, kl_coef=args.kl_coef, kl_penalty=args.kl_penalty,
                      norm_adv_by_std=bool(args.norm_adv_by_std), loss_agg_mode=args.loss_agg_mode)

    # ---- training loop: total_episodes=1 -> one pass over shuffled records in rollout batches
    order = list(range(len(records))); random.Random(args.seed).shuffle(order)
    order = order[rank::world]
    per_rank_bs = max(1, args.rollout_batch_size // world)
    batches = [order[i:i + per_rank_bs] for i in range(0, len(order), per_rank_bs)]
    if args.steps > 0:
        batches = (batches * (args.steps // max(1, len(batches)) + 1))[:args.steps]
    pre = lambda pil, mn, mx: image_preprocess(pil, mn, mx)
    for step, idx in enumerate(batches):
        t0 = time.time()
        exs = [make_example(records[i], tok, pre, args.min_pixels, args.max_pixels, args.max_prompt_length) for i in idx]
        batch = collate_left_pad(exs, pad_id)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        # 1) rollout + edits (rollout worker)
        raw.eval()
        ro = rollout_with_edits(sampler, batch, tok, rcfg, eos_id, pad_id, special_ids, extractor, edit_cfg, rng, gen)
        raw.train()
        # 2) rewards: scalar per sequence (custom.py:141-161), summed components (all_flask_server.py:203-235)
        totals, comps = reward.score_components(ro["texts"], ro["ground_truth"])
        scores = torch.tensor(totals, dtype=torch.float32, device=device)
        # 3) advantages (core_algos.py:128-181)
        adv = broadcast_advantages(group_advantages(scores, ro["group_ids"], gcfg.norm_adv_by_std, gcfg.adv_eps), ro["response_mask"].float())
        # 4) old / ref log-probs on the final (edited) tokens (ray_trainer.py:696-705)
        pv, grid = batch.get("pixel_values"), batch.get("image_grid_thw")
        pv_rep = grid_rep = None
        if pv is not None and grid is not None and grid.numel():
            pv_rep, grid_rep = _repeat_images(batch, raw.config.image_token_id, args.n)
        N = ro["responses"].shape[0]
        with torch.no_grad():
            old_lp = torch.cat([sequence_log_probs(raw, ro["input_ids"][s], ro["attention_mask"][s], ro["responses"][s],
                                                   *_slice_images(pv_rep, grid_rep, s), args.temperature)
                                for s in _chunks(N, args.micro_batch_size)])
            ref_lp = torch.cat([sequence_log_probs(ref, ro["input_ids"][s], ro["attention_mask"][s], ro["responses"][s],
                                                   *_slice_images(pv_rep, grid_rep, s), args.temperature)
                                for s in _chunks(N, args.micro_batch_size)])
        # 5) update (dp_actor.py:230-279): ppo_epochs x micro-batches, grad accumulation, clip, skip non-finite
        metrics = {}
        for _ in range(args.ppo_epochs):
            opt.zero_grad(set_to_none=True)
            n_micro = len(list(_chunks(N, args.micro_batch_size)))
            for s in _chunks(N, args.micro_batch_size):
                lp = sequence_log_probs(model, ro["input_ids"][s], ro["attention_mask"][s], ro["responses"][s],
                                        *_slice_images(pv_rep, grid_rep, s), args.temperature)
                loss, m = editgrpo_loss(lp, old_lp[s], ref_lp[s], adv[s], ro["response_mask"][s].float(), gcfg)
                (loss / n_micro).backward()
                for k, v in m.items():
                    metrics[k] = metrics.get(k, 0.0) + float(v) / n_micro
            gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.max_grad_norm)
            if torch.isfinite(gnorm):
                opt.step()
            else:
                log(f"step {step}: non-finite grad norm, skipping update")
        # 6) logging: reward, components, edits, repetition
        rep = [repetition_stats(t) for t in ro["texts"]]
        if step % args.log_every == 0:
            log(json.dumps({"step": step, "reward": scores.mean().item(), **{k: sum(v) / len(v) for k, v in comps.items()},
                            "n_edited": len(ro["edits"]), "n_ops": sum(len(e["ops"]) for e in ro["edits"]),
                            "resp_len": ro["response_mask"].sum(1).float().mean().item(),
                            "rep_sent": sum(r["repeated_sentence_rate"] for r in rep) / len(rep),
                            "grad_norm": float(gnorm), **{k: round(v, 6) for k, v in metrics.items()},
                            "sec": round(time.time() - t0, 2)}))
    if args.save_dir and rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in raw.state_dict().items()}, os.path.join(args.save_dir, "model.safetensors"))
        log(f"saved to {args.save_dir}")
    if world > 1:
        dist.destroy_process_group()


def _chunks(n, size):
    for i in range(0, n, size):
        yield slice(i, min(n, i + size))


def _repeat_images(batch, image_token_id, n):
    """Repeat each prompt's pixel rows / grid rows n times (prompt-major order)."""
    from editgrpo.rollout import _images_per_prompt
    pv, grid = batch["pixel_values"], batch["image_grid_thw"]
    counts = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
    k_per = _images_per_prompt(batch["input_ids"], image_token_id, grid)
    pvs, grids, oi, op = [], [], 0, 0
    for k in k_per:
        npix = sum(counts[oi:oi + k])
        pvs.extend([pv[op:op + npix]] * n); grids.extend([grid[oi:oi + k]] * n)
        oi += k; op += npix
    return pvs, grids                                          # lists indexed by row


def _slice_images(pvs, grids, s):
    if pvs is None:
        return None, None
    return torch.cat(pvs[s], 0), torch.cat(grids[s], 0)


if __name__ == "__main__":
    main()
