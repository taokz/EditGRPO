# nano-EditGRPO status

## Done (CPU-tested, `python -m pytest nano/tests -q`)
- DESIGN.md: full component -> file:line map, loss variant pinned against RL2.
- `edit.py`: faithful sentence-level edit (server entity matching + client ops + G/2 source->target group logic);
  `test_edit.py` compares against the original `process_text_with_server_analysis` executed in isolation (5 seeds).
- `grpo.py`: group advantages (std on/off), clipped surrogate, low-var KL (clamped k3), four aggregation modes;
  hand-computed tests.
- `reward.py`: unweighted component sum as the Flask server; cheap text rewards; lazy clinical plugins.
- `eval.py`: raw metrics (default), `legacy_eval_dedup` reproduction, repetition metrics.
- `data.py`: json records, Qwen2.5-VL chat prompt, image resize, synthetic CXR data, left-pad collate.
- `rollout.py`: `Sampler` protocol, `TorchSampler`, EOS mask, edited re-tokenisation (+ `edited_append_eos` fix flag).
- `train.py`: torchrun-ready loop with the paper defaults (AdamW 1e-6, clip 0.2, low_var_kl 1e-2, Dr. GRPO agg).
- `plugins/rate_extractor.py`: RaTE-NER + BioLORD extractor in-process.
- ISSUES.md (12 issues), SECURITY_NOTES.md, `fix-original` branch (5 commits, one per bug, defaults unchanged).

## Model (`model.py`, ~800 lines, torch + safetensors only)
- Vision tower (patch embed, 2D RoPE, window/full attention blocks, PatchMerger), Qwen2 LM with M-RoPE
  (`get_rope_index`, `mrope_section` interleave), GQA, 2D attention masks for left-padded prompts, per-layer KV cache,
  gradient checkpointing, `generate` with temperature/top-p/top-k, `load_pretrained` (HF key names kept) and
  `image_preprocess` (smart_resize to multiples of 28, CLIP normalisation, HF patch flattening order).
- `tests/test_model.py` (10 tests, tiny random config): shapes, finite logits, cached vs uncached greedy decoding
  agree, hand-computed `get_rope_index`, gradients finite with checkpointing on, preprocess/grid consistency.
- CPU smoke run: `python nano/train.py --synthetic --steps 2 --rollout_batch_size 4 --n 4 --micro_batch_size 4`
  -> finite loss, 8/16 rollouts replaced by edited versions (2 per group of 4), reward and repetition logged.
- Real-weight equivalence with HF is NOT verified here (no GPU, no weights); see the list below.

## Stubbed / not reproduced here
- Clinical rewards and the RaTE extractor are untested in this environment (no GPU, libraries not installed).
- No vLLM-speed sampling: `TorchSampler` is a plain KV-cache loop; nano-vllm goes behind `Sampler`.
- Multi-GPU: DDP only (original: FSDP + Ray). No parameter/optimizer offload.
- `cxb_bal_acc` reward component (needs class-frequency file absent from the repo) is not ported.

## Must be validated on GPU
1. Real-weight equivalence of `model.py` vs HF `Qwen2_5_VLForConditionalGeneration` (logits on a real image+prompt,
   greedy continuation identical for 64 tokens; `get_rope_index` identical).
2. Reward parity: `make_reward([...], device="cuda")` vs the Flask `/predict` on 100 (hyp, ref) pairs.
3. Edit parity: `RaTEExtractor` + `analyze_pair` vs `/analyze` server output on the same pairs.
4. Training curves vs the original on `examples/data/sample_3000_mimic_cxr.json`-like data (reward, response
   length, repetition rate), paper defaults; then the flagged fixes (`--edited_append_eos 1`, `--fix_tau_scaling 1`).
