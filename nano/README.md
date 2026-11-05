# nano-EditGRPO (draft)

A small, readable re-implementation of EditGRPO (EasyR1/verl fork): GRPO for chest X-ray report
generation where a fraction of each rollout group is *edited toward the gold report* using
RaTE-NER entity matching. Pure PyTorch (no vLLM, Ray, FSDP, transformers, Flask); heavy pieces are
lazy plugins. Design rationale and file:line mapping: [DESIGN.md](DESIGN.md); plan: [PLAN.md](PLAN.md);
decisions: [DECISIONS.md](DECISIONS.md); audit: [ISSUES.md](ISSUES.md); privacy: [SECURITY_NOTES.md](SECURITY_NOTES.md).

## Layout
```
nano/
  editgrpo/
    data.py     records, Qwen2.5-VL prompt, synthetic data, left-pad collate
    model.py    pure-PyTorch Qwen2.5-VL, load safetensors, KV-cache generate, image_preprocess
    edit.py     entity comparison + sentence ops (replace/delete/add)
    grpo.py     group advantages, clipped policy loss, loss_agg modes, low_var_kl
    reward.py   RadGraph / CheXbert / RaTEScore sum + plugins
    rollout.py  Sampler interface, group sampling, edited re-tokenisation
    eval.py     raw metrics, legacy dedup flag, repetition rate
  train.py      training loop
  tests/        pytest, CPU only, no network
```

## Mapping to the original
See the table in [DESIGN.md](DESIGN.md) (component -> original file:line -> nano module, verdict keep/simplify/drop).

## Original hyperparameters (nano defaults follow the paper run)
| Setting | Value |
|---|---|
| rollouts per prompt n / temperature | 5 / 1.0 (top_p 1, top_k -1) |
| lr / schedule | 1e-6 / constant, no warmup |
| AdamW | betas (0.9, 0.999), weight decay 1e-2 |
| grad clip | 1.0 (skip non-finite) |
| PPO clip | 0.2 (no dual clip, no entropy) |
| KL | low_var_kl, coef 1e-2 |
| norm_adv_by_std | False (paper sbatch) / True (yaml default) |
| loss_agg | seq-mean-token-sum-norm (paper) / token-mean (yaml) |
| max response / prompt length | 4096 / 4096 |
| rollout batch / micro batch | 32 / 2 |
| edit ops per paragraph | 1 |
| edited paragraphs `num_para` | -1 (= G//2 = 2 edits; sources overwrite targets) |
| tau (client) / server threshold | 0.6 / 0.6 |
| negative-pair weight | 0.3612 |
| freeze vision tower | True |
| min / max pixels | 262144 / 401408 |
| reward | radgraphf1 + cxb14micro + ratescore (unweighted sum, range [0,3]) |

## Clinical rewards and the RaTE extractor (plugins)
- Install `radgraph`, `f1chexbert`, `RaTEScore` (+ `transformers`); apply the patches in `modified_lib/` of the original repo to those packages.
- The RaTE extractor (`Angelakeke/RaTE-NER-Deberta` + `FremyCompany/BioLORD-2023-C`) is loaded lazily as a plugin that returns entities and embeddings; the lexicon extractor is the CPU default.
- Rewards: `editgrpo.reward.make_reward(["radgraphf1","cxb14micro","ratescore"], response_type="free_form", device="cuda")`
  builds `SumReward` over `RadGraphF1Reward` (`pip install radgraph`), `CheXbert14MicroReward` (`pip install f1chexbert`),
  `RaTEScoreReward` (`pip install RaTEScore`); apply the patches in `modified_lib/` to match the original server. In
  `train.py`: `--reward clinical --reward_device cuda`. `--reward cheap` swaps in text stand-ins for CPU tests.
- Edit extractor: `editgrpo.plugins.rate_extractor.RaTEExtractor(device=...)` (needs `transformers`), selected with
  `--extractor rate`; `--extractor lexicon` is the CPU stand-in (`editgrpo.edit.LexiconExtractor`).
- Edit knobs: `--num_ops_per_para 1 --num_para -1 --tau 0.6` (paper); `--fix_tau_scaling 1`, `--edited_append_eos 1`
  enable the fixes from ISSUES.md #3 / #1 (off by default).

## Plugging in nano-vllm
`rollout.py` defines a `Sampler` interface (generate G sequences per prompt given prompt ids and images,
returning token ids and log-probs not required, since old log-probs are recomputed by the actor).
Implement the `editgrpo.rollout.Sampler` protocol: `sampler(batch, n, cfg: RolloutConfig, generator) -> response_ids`
of shape `(B*n, cfg.response_length)`, prompt-major (row `i*n+k` is sample `k` of prompt `i`), right-padded with the pad
id after the first EOS. `TorchSampler` is the reference implementation on top of `model.generate`; a nano-vllm sampler
needs to sync weights from the training model after every step (the original lets vLLM reload actor weights).
Pass your sampler to `rollout_with_edits(...)` in `train.py`.

## CPU smoke test
```
python nano/train.py --synthetic --steps 2     # tiny random model, synthetic reports, cheap rewards, lexicon edits
python -m pytest nano/tests -q
```

## Fixed vs original
Summary of the audit (training-time EOS bug, eval dedup, tau double-scaling, silent zero rewards, ...) and each
item's status (nano flag / `fix-original` branch / documented only) is in [ISSUES.md](ISSUES.md). Defaults
reproduce the original behaviour; fixes are opt-in flags except raw eval metrics and raising on reward errors.
