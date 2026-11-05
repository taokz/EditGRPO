"""Bug D: server returns length-1 score lists for multiple hyps.

Loads verl/utils/reward_score/rad_scores.py by file path (avoids the verl
package import chain; needs only `requests`) and monkeypatches requests.post
with a fake response dict.
Run: python tests_fix/test_bug_d_batch_shape.py
"""
import importlib.util, json, os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verl", "utils", "reward_score", "rad_scores.py")
spec = importlib.util.spec_from_file_location("rad_scores", path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

KEYS = ["final_scores", "format_scores", "precision_radgraph_scores", "recall_radgraph_scores",
        "f1_radgraph_scores", "balanced_accuracy_chexbert", "accuracy_chexbert", "micro_chexbert_14",
        "macro_chexbert_14", "micro_chexbert_5", "macro_chexbert_5"]

class FakeResp:
    status_code = 200
    def __init__(self, n):
        self.text = json.dumps({"result": "success", "data": {"predictions": {k: [0.7] * n for k in KEYS}}})

m.requests.post = lambda *a, **k: FakeResp(1)  # server returns length-1 lists
try:
    m.rad_compute_score(["a", "b", "c"], ["x", "y", "z"])
    raise SystemExit("FAIL: expected ValueError")
except ValueError as e:
    assert "single score for 3 hyps" in str(e), e
    print("OK raised:", e)

# well-formed responses still work
m.requests.post = lambda *a, **k: FakeResp(3)
assert [r["overall"] for r in m.rad_compute_score(["a", "b", "c"], ["x", "y", "z"])] == [0.7] * 3
m.requests.post = lambda *a, **k: FakeResp(1)
assert m.rad_compute_score("a", "x")["overall"] == 0.7
print("OK normal paths")
