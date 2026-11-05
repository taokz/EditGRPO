import pytest

from editgrpo.eval import evaluate, legacy_clean_text, legacy_clean_text_green, repetition_stats
from editgrpo.reward import make_reward


def test_legacy_exact_duplicate_removed():
    out = legacy_clean_text("The heart is normal. The heart is normal. Lungs are clear.")
    assert out == "The heart is normal. Lungs are clear."


def test_legacy_subset_removed():
    out = legacy_clean_text("No pleural effusion. No pleural effusion or pneumothorax.")
    assert out == "No pleural effusion."


def test_legacy_misc():
    assert legacy_clean_text("   ") == "[EMPTY]"
    assert legacy_clean_text("Heart normal<|endoftext|><|endoftext|>") == "Heart normal."
    assert legacy_clean_text("Lungs clear. Indication: cough") == "Lungs clear."
    assert legacy_clean_text("Short. Lungs are clear.") == "Lungs are clear."  # 1-word dropped
    assert legacy_clean_text("junk \\boxed{ Lungs  clear }") == "Lungs clear"
    # faithful quirk: a '.' inside the box splits the sentence, the lone '}' is dropped
    # as a short fragment and the box is lost (lightweight_eval.py:259,288)
    assert legacy_clean_text("junk \\boxed{ Lungs clear. }") == "junk \\boxed{ Lungs clear."


def test_legacy_green_exact_only():
    t = "No pleural effusion. No pleural effusion or pneumothorax. no pleural effusion."
    assert legacy_clean_text_green(t) == "No pleural effusion. No pleural effusion or pneumothorax."


def test_repetition_stats_hand_built():
    s = repetition_stats("The heart is normal. The heart is normal. Lungs are clear.")
    assert s["n_sentences"] == 3
    assert s["repeated_sentence_rate"] == pytest.approx(1 / 3)
    assert s["n_words"] == 11
    # 8 4-grams (11 words); second copy of "the heart is normal" is seen -> grams:
    # [the heart is normal],[heart is normal the],[is normal the heart],[normal the heart is],
    # [the heart is normal]*dup,[heart is normal lungs],[is normal lungs are],[normal lungs are clear]
    assert s["repeated_4gram_rate"] == pytest.approx(1 / 8)
    z = repetition_stats("")
    assert z["repeated_sentence_rate"] == 0.0 and z["n_words"] == 0


def test_evaluate_raw_vs_legacy():
    rew = make_reward(["radgraphf1", "ratescore"], cheap=True)
    refs = ["No pleural effusion. Heart is normal."]
    hyps = ["No pleural effusion. Heart is normal. Heart is normal. Heart is normal. Heart is normal."]
    res = evaluate(hyps, refs, rew)
    assert set(res) == {"raw", "repetition"}
    assert res["repetition"]["repeated_sentence_rate"] == pytest.approx(3 / 5)
    res2 = evaluate(hyps, refs, rew, legacy_eval_dedup=True)
    assert res2["raw"] == res["raw"]
    assert res2["legacy"]["radgraphf1"] == pytest.approx(1.0)
    assert res["raw"]["radgraphf1"] < res2["legacy"]["radgraphf1"]  # dedup inflates
    assert "total" in res["raw"]
