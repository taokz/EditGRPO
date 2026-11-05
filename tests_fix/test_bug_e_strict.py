"""Bug E: server errors silently become all-zero rewards.

Loads verl/utils/reward_score/rad_scores.py by file path (avoids the verl
package import chain; needs only `requests`) and monkeypatches requests.post.
Run: python tests_fix/test_bug_e_strict.py
"""
import importlib.util, os

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verl", "utils", "reward_score", "rad_scores.py")
spec = importlib.util.spec_from_file_location("rad_scores", path)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

def boom(*a, **k):
    raise ConnectionError("server down")

class Resp500:
    status_code = 500
    text = "internal error"

for post in (boom, lambda *a, **k: Resp500()):
    m.requests.post = post
    res = m.rad_compute_score(["a", "b"], ["x", "y"])  # default: silent zeros (original)
    assert [r["overall"] for r in res] == [0.0, 0.0]
    for strict_call in (lambda: m.rad_compute_score(["a", "b"], ["x", "y"], strict=True),
                        lambda: m.rad_compute_score("a", "x", strict=True)):
        try:
            strict_call()
            raise SystemExit("FAIL: strict=True did not raise")
        except (ConnectionError, RuntimeError) as e:
            pass
print("OK default returns zeros; strict=True raises")
