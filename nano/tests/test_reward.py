import pytest

from editgrpo.reward import (FormatReward, LexiconF1Reward, SentenceSetF1Reward, SumReward,
                             TokenF1Reward, extract_boxed_content, format_reward, make_reward,
                             RadGraphF1Reward, Reward)


def test_protocol():
    assert isinstance(TokenF1Reward(), Reward)


def test_token_f1_known_values():
    r = TokenF1Reward()
    # hyp {the,heart,is,normal}, ref {the,heart,is,enlarged}: overlap 3/4 -> F1 .75
    assert r.score(["The heart is normal."], ["The heart is enlarged!"]) == pytest.approx([0.75])
    assert r.score(["a b"], ["a b"]) == [1.0]
    assert r.score(["a"], ["b"]) == [0.0]
    assert r.score([""], [""]) == [1.0]
    assert r.score([""], ["x"]) == [0.0]
    # bag semantics: hyp a a b vs ref a b b -> overlap 2, P=R=2/3
    assert r.score(["a a b"], ["a b b"]) == pytest.approx([2 / 3])


def test_sentence_set_f1():
    r = SentenceSetF1Reward()
    h = "No effusion. Heart is normal."
    ref = "No effusion.\nLungs are clear."
    assert r.score([h], [ref]) == pytest.approx([0.5])
    assert r.score(["No effusion. NO EFFUSION!"], ["No effusion."]) == [1.0]


def test_lexicon_negation():
    r = LexiconF1Reward()
    assert r.labels("There is a pleural effusion.") == {"pleural_effusion"}
    assert r.labels("No pleural effusion.") == set()
    assert r.labels("Lungs clear without consolidation. Mild cardiomegaly.") == {"cardiomegaly"}
    assert r.labels("Negative for pneumothorax.") == set()
    # hyp: effusion+cardiomegaly, ref: cardiomegaly only -> 2*1/(2+1)
    assert r.score(["Pleural effusion. Cardiomegaly."], ["Cardiomegaly."]) == pytest.approx([2 / 3])
    # negated finding vs present finding disagree
    assert r.score(["No pneumothorax."], ["Large pneumothorax."]) == [0.0]
    assert r.score(["No pneumothorax."], ["Lungs clear."]) == [1.0]


def test_sum_reward_sums_components():
    sr = make_reward(["radgraphf1", "cxb14micro", "ratescore"], cheap=True)
    hyps = ["Pleural effusion. Cardiomegaly.", "No effusion."]
    refs = ["Cardiomegaly.", "No effusion."]
    total, parts = sr.score_components(hyps, refs)
    assert set(parts) == {"radgraphf1", "cxb14micro", "ratescore"}
    for i in range(2):
        assert total[i] == pytest.approx(sum(parts[k][i] for k in parts))
    assert sr.score(hyps, refs) == total
    assert total[1] == pytest.approx(3.0)


def test_format_regexes():
    assert format_reward("\\boxed{abc} tail", "short") == 1.0
    assert format_reward("no box", "short") == 0.0
    assert format_reward("<think>x</think> \\boxed{a}", "cot") == 1.0
    assert format_reward("\\boxed{a}", "cot") == 0.0
    assert format_reward("anything", "free_form") == 1.0
    assert format_reward("anything", "weird") == 0.0
    # short pattern is fullmatch from the start: leading text fails
    assert format_reward("pre \\boxed{a}", "short") == 0.0


def test_format_only_added_when_not_free_form():
    hyps, refs = ["\\boxed{Cardiomegaly.}"], ["Cardiomegaly."]
    free = make_reward(["radgraphf1", "format"], response_type="free_form", cheap=True)
    t, p = free.score_components(hyps, refs)
    assert "format" in p and t[0] == pytest.approx(p["radgraphf1"][0])
    short = make_reward(["radgraphf1", "format"], response_type="short", cheap=True)
    t, p = short.score_components(hyps, refs)
    assert p["format"] == [1.0]
    # boxed extraction fed to the content reward -> perfect match, plus 1 format
    assert p["radgraphf1"] == [1.0] and t == [2.0]


def test_boxed_extraction():
    assert extract_boxed_content("x \\boxed{abc} y") == "abc"
    assert extract_boxed_content("\\boxed{a {b} c} z") == "a {b} c"
    assert extract_boxed_content("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_boxed_content("\\boxed{one} then \\boxed{two}") == "two"  # last
    assert extract_boxed_content("\\boxed{unclosed") == "None"
    assert extract_boxed_content("nothing") == "None"


def test_clinical_plugin_import_error_is_clear():
    pytest.importorskip("torch")
    try:
        import radgraph  # noqa
        pytest.skip("radgraph installed")
    except ImportError:
        with pytest.raises(ImportError, match="pip install radgraph"):
            RadGraphF1Reward()


def test_unknown_component():
    with pytest.raises(ValueError):
        make_reward(["nope"], cheap=True)
