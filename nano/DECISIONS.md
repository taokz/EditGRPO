# Decisions log

- D1 Branch `nano` created from `main`; only `nano/` is written. Original files untouched.
- D2 Commit identity: Kai Zhang <kai.eric.zhang@gmail.com> (repo owner), no trailers.
- D3 Edit op ported in two layers: `compare_entities` (server logic) and `apply_edits` (client
  logic). The transport (Flask/JSON) disappears; the maths is identical, incl. affinity matrix,
  negative-class weight 0.3612, server threshold 0.6.
- D4 Default extractor on CPU is a lexicon/hashing extractor so tests run without models; the real
  RaTE-NER + BioLORD extractor is `editgrpo.plugins.rate` with lazy transformers import.
- D5 Known quirk `best_score >= tau * len(fp_entities)` after mean-normalising the score is kept
  as default (faithful) with `fix_tau_scaling=True` as the corrected variant.
- D6 Loss: implement both `token-mean` (yaml default) and `seq-mean-token-sum-norm` (paper run),
  `norm_adv_by_std` switchable; defaults follow the paper sbatch (no std norm, Dr. GRPO agg).
- D7 Edited responses: original tokenises edited text without EOS. nano keeps that as default
  (`edited_append_eos=False`) and offers the fix behind a flag (needs GPU validation).
- D8 Repetition: evaluation reports raw metrics, legacy-dedup metrics (flag) and a repetition rate.
