# nano-EditGRPO plan (living document)

Revised after reading the original code (not the prompt's assumptions).

## What the original actually does (short)
1. Qwen2.5-VL (SFT init) generates G=5 rollouts per CXR prompt with vLLM, T=1.0.
2. **Post-rollout edit** (`vllm_rollout_spmd.py:437-1060`): per prompt, G//2 rollouts are randomly
   chosen as *sources*; each is edited toward the gold report using RaTE-NER entities
   (`flask_server/RaTEscore_extractor.py` + `verl/utils/RaTEscore_extractor.py`), and the edited
   text *replaces a different* rollout (a *target*). So a group of 5 ends up as
   3 unedited + 2 edited copies (`extractor_num_para=-1`, `extractor_num_ops_per_para=1`, tau=0.6).
3. Edited text is re-tokenized without EOS and padded; old log-probs are then recomputed by the
   actor on the edited sequences (so the PPO ratio is 1 at the first update, but edited samples
   are off-policy w.r.t. the sampler).
4. Rewards: Flask server computes RadGraph-F1, CheXbert-14 micro-F1, RaTEScore; combined to a scalar.
5. GRPO advantage (group mean; std-normalisation **off** in the paper run), PPO clipped surrogate,
   low-var KL to the SFT reference (coef 1e-2), loss_agg "seq-mean-token-sum-norm" (Dr. GRPO).

## Steps (push after each)
1. DESIGN.md + PLAN.md + DECISIONS.md, draft PR.            [done first]
2. `nano/editgrpo/edit.py` + tests: faithful port of the sentence-level edit (server entity
   matching + client sentence ops), extractor behind a protocol, cheap lexicon extractor for CPU.
3. `nano/editgrpo/grpo.py` + tests: advantages, ratio/clip, low-var KL, both loss_agg modes.
4. `nano/editgrpo/model.py`: pure-PyTorch Qwen2.5-VL (vision, merger, LM, M-RoPE), KV-cache generate.
5. `reward.py` (text rewards + lazy clinical plugins), `rollout.py`, `train.py`; CPU smoke run.
6. README, ISSUES, SECURITY_NOTES, STATUS; `fix-original` branch for confirmed bugs.

## Changes versus the prompt's plan (why)
- Edit = "edit source, overwrite a different target" (not "edit each rollout in place"); the plan
  text was wrong, the code is the authority.
- The RaTE-driven edit has two halves (server: entity matching; client: sentence ops). Both are
  ported in-process; the NER/embedding models become an optional plugin, a lexicon extractor
  stands in on CPU.
- Faithful default for the tau check keeps a scaling quirk (see ISSUES.md #3) behind a flag.

## Final state (end of session)
Steps 1-6 done and pushed on `nano` (draft PR taokz/EditGRPO#2); `fix-original` pushed with 5 single-bug commits.
Deviation from the plan: model.py, reward/eval and data/docs were written by parallel subagents against interfaces
fixed up front; the edit operation, the loss and the integration (rollout.py, train.py) were done by hand.
