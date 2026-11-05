"""Bug A: clean_text dedup hides repetition.

eval/lightweight_eval.py and eval/green_eval.py import radgraph/green_score, so
we cannot import them here. We exec the source of `repetition_metrics` and
`clean_text` (plus the LEGACY_DEDUP flag) extracted via ast, in isolation.
Run: python tests_fix/test_bug_a_dedup.py
"""
import ast, re, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
REPORT = ("The heart is normal. No pleural effusion. The heart is normal. "
          "No pleural effusion. The heart is normal.<|endoftext|>")

def load(path):
    src = open(os.path.join(ROOT, path)).read()
    ns = {"re": re}
    for node in ast.parse(src).body:
        if (isinstance(node, ast.FunctionDef) and node.name in ("clean_text", "repetition_metrics")) or \
           (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "LEGACY_DEDUP"):
            exec(compile(ast.Module([node], []), path, "exec"), ns)
    return ns

for path in ("eval/lightweight_eval.py", "eval/green_eval.py"):
    ns = load(path)
    assert ns["LEGACY_DEDUP"] is True
    legacy = ns["clean_text"](REPORT)
    ns["LEGACY_DEDUP"] = False
    raw = ns["clean_text"](REPORT)
    assert legacy.count("The heart is normal") == 1, legacy
    assert raw.count("The heart is normal") == 3 and "endoftext" not in raw, raw
    m_leg, m_raw = ns["repetition_metrics"]([legacy]), ns["repetition_metrics"]([raw])
    assert m_raw["repeated_sentence_rate"] > m_leg["repeated_sentence_rate"] == 0
    print(path, "OK", m_leg, m_raw)
