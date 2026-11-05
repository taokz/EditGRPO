"""Bug B: tau is scaled by len(fp_entities) although the score is already averaged.

Loads verl/utils/RaTEscore_extractor.py directly by file path (importlib), which
avoids the heavy verl package import chain; it only needs torch + requests.
Run: python tests_fix/test_bug_b_tau.py
"""
import importlib.util, os
import torch

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verl", "utils", "RaTEscore_extractor.py")
spec = importlib.util.spec_from_file_location("rate_extractor", path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

e = lambda v: {"embedding": torch.tensor(v, dtype=torch.float)}
fp_ents = [dict(e([1.0, 0.0]), text_span="a", sentence="Bad sentence."), e([0.0, 1.0])]
fp = dict(fp_ents[0], sentence_entities=fp_ents)
refs = {"Good sentence.": [e([1.0, 0.0]), e([0.0, 1.0])]}  # normalised score == 1.0
text = "Bad sentence. Other."

orig, _ = m.replace_fp_sentence_complex_rs(text, fp, refs, 0.6)
fixed, info = m.replace_fp_sentence_complex_rs(text, fp, refs, 0.6, fix_tau_scaling=True)
assert orig == text, orig            # 1.0 >= 0.6*2 is False -> never replaced
assert info is not None and "Good sentence." in fixed, fixed  # 1.0 >= 0.6
print("OK original:", repr(orig), "| fixed:", repr(fixed))
