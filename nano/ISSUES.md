# Issues found while auditing EditGRPO (original code)

Paths are relative to the repo root. Line numbers were re-verified by reading the files.
Status: **nano** = fixed/flagged in `nano/`; **fix-original** = planned for the `fix-original` branch; **doc** = documented only.
Everything marked "needs GPU validation" changes training behaviour and has not been run at scale.

## #1 [high] Edited / GT-injected responses are re-tokenised without EOS
- Where: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:116-149` (`_clean_text_and_tokenize` strips every EOS id), `:969-1004` (edited text tokenised, right-padded with pad id); mask built by `get_eos_mask`, `verl/utils/torch_functional.py:113-117` (mask is 1 up to and including the first EOS).
- Evidence: with no EOS in the row, `get_eos_mask` returns all ones across the padding. Reward is placed at `valid_response_length - 1` (`verl/workers/reward/custom.py:147,156`), i.e. on a pad token; policy loss and KL run over the pads; the model is never taught to emit EOS after an edited report (2 of 5 rollouts per group).
- Root cause: EOS deliberately removed from edited text and never re-appended.
- Fix: `edited_append_eos` flag in nano (`rollout.retokenize_edited`); default False = original, True appends EOS and masks from it. Needs GPU validation. Plausible cause of repetition / non-stopping generations.
- Status: nano (flag) + fix-original.
- Test: `pytest nano/tests -k retokenize`; check `response_mask.sum(1)` equals true length for edited rows with the flag on.

## #2 [high] Evaluation dedups sentences in hyps and refs, training does not
- Where: `eval/lightweight_eval.py:245-341` (`clean_text`: exact + Jaccard > 0.8 at `:302-309` + subset > 0.9 at `:313-318`), `eval/green_eval.py:19-60` (exact dedup). Training rewards (`flask_server/all_flask_server.py`) never dedup.
- Evidence: repeated sentences vanish before scoring, so a repetitive model looks fine at eval; references are also altered.
- Root cause: eval-time cleaning applied symmetrically to hyps and refs.
- Fix: nano `eval.py` reports raw metrics by default, `legacy_eval_dedup=True` reproduces the old numbers, and a repetition rate is always reported.
- Status: nano; fix-original (flag).
- Test: `pytest nano/tests -k dedup`: a hyp with a sentence repeated 10x scores lower raw than legacy.

## #3 [med] tau check double-scales
- Where: `verl/utils/RaTEscore_extractor.py:228-237` (`score /= len(fp_entities)`), then `:240` requires `best_score >= similarity_threshold * len(fp_entities)`.
- Evidence: score is a mean in [0,1] but the threshold is scaled by the entity count; with >= 2 FP entities and tau = 0.6 the threshold is >= 1.2, so replacement can never fire and the op falls through to delete.
- Fix: nano `fix_tau_scaling` flag (default False = faithful, because the paper results were produced with it).
- Status: nano (flag) + fix-original.
- Test: `pytest nano/tests -k tau`: 2-entity FP sentence with cos 0.9 replaces only when the flag is on.

## #4 [high, latent] Batch reward mode zeroes all but the first sample
- Where: server returns length-1 `final_scores` (`flask_server/all_flask_server.py:211,224,233-235`); client indexes `[i]` with 0.0 fallback (`verl/utils/reward_score/rad_scores.py:163-175`).
- Masked by the default `reward_calc_type='sequential'` (`verl/workers/reward/custom.py:61`); batch branch at `custom.py:163`.
- Fix: nano scores in-process per sample; no batch path. fix-original: raise on length mismatch.
- Status: nano (n/a) ; fix-original.
- Test: call the server with 2 hyps and assert `len(final_scores) == 2`.

## #5 [med] Reward client returns 0 on any error
- Where: `verl/utils/reward_score/rad_scores.py:119-193` (non-200 `:126`, `result != success` `:137`, bare exception `:186-193` all return zeros).
- Evidence: a down/overloaded server silently yields reward 0 for every sample -> zero advantage -> training signal collapses without any failure.
- Fix: nano `reward.py` raises. fix-original: raise after retries.
- Status: nano; fix-original.
- Test: `pytest nano/tests -k reward_error`.

## #6 [med] `_clean_eos_from_text` collapses all whitespace
- Where: `vllm_rollout_spmd.py:151-223`, `' '.join(text.split())` at `:203`; used for edited rows (`:131`) and for GT text (`:444,488,496`).
- Evidence: newlines/paragraph structure are removed in edited rows but kept in sampled rows -> a spurious style signal correlated with "was edited".
- Fix: nano keeps default behaviour (faithful) and offers `clean_whitespace=False` option in `retokenize_edited`.
- Status: nano (flag); doc.
- Test: round-trip a multi-line report through `retokenize_edited` with the flag off/on.

## #7 [med] No length / repetition penalty in the reward
- Where: reward = unweighted sum of RadGraph F1 + CheXbert-14 micro-F1 + RaTEScore (`all_flask_server.py:203-235`).
- Evidence: all three are recall-friendly; repeating findings does not lower them -> repetition is not penalised and can be rewarded (reward-hacking lead, not proven).
- Fix: none by default (faithful). Repetition rate is logged in nano; an optional penalty is left as an experiment.
- Status: doc (+ nano metric).
- Test: score a report vs. the same report repeated 3x.

## #8 [high, eval] RadCliQ inputs are corpus-level constants per sample
- Where: `eval/lightweight_eval.py:399-400` (`radgraph_scores_array = [radgraph_combined_score] * len(hyps)`, `bleu2_scores = [bleu2_score] * len(hyps)`), consumed at `:411`.
- Evidence: per-sample RadCliQ mixes per-sample BERTScore/semb with one corpus RadGraph and one corpus BLEU-2 value, so it is not the RadCliQ composite metric.
- Fix: compute RadGraph and BLEU-2 per sample (fix-original); nano reports corpus-level components separately and does not report this composite.
- Status: fix-original; doc.
- Test: two hyps of different quality must get different RadGraph/BLEU inputs.

## #9 [low] Config / packaging bugs
- `class_frequencies` is only defined under `__main__` (`flask_server/all_flask_server.py:256-261`) but used at `:167,169` -> NameError / HTTP 500 under `flask run`/WSGI.
- yaml `compute_score: radgraph_chexbert_14` (`examples/editgrpo_*_2_5.yaml:74`) is not accepted by `CustomRewardManager` (`verl/workers/reward/custom.py:93-129` raises NotImplementedError); only the sbatch override `rad_server` (`examples/mimic_cxr_editgrpo_ops1_para_half_tau_0_6.sbatch:105`) works.
- `__all__` is missing a comma (`verl/utils/reward_score/__init__.py:30-31`): `"r1v_compute_score" "bleu4_compute_score"` concatenate into one bogus name.
- `flask_server/README.md` documents a different server than `all_flask_server.py`.
- Status: fix-original; doc.
- Test: `python -c "import verl.utils.reward_score as r; [getattr(r, n) for n in r.__all__]"`.

## #10 [low] Train/eval metric scale mismatch
- Training reward uses per-sample micro-F1 (CheXbert-14); eval reports corpus-level scores. Absolute values are not comparable; reward curves cannot be read against eval tables.
- Status: doc; nano logs both per-sample and corpus-level.
- Test: compare mean per-sample micro-F1 with corpus micro-F1 on a fixed set.
