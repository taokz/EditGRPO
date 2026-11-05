"""Unit tests for the sentence-level edit operation (nano/editgrpo/edit.py) on hand-made reports.

Expected values were derived by tracing the original functions in verl/utils/RaTEscore_extractor.py and
flask_server/RaTEscore_extractor.py by hand; where the original can be imported in isolation
(`verl/utils/RaTEscore_extractor.py` needs only torch/requests) we compare against it directly."""
import importlib.util
import os
import random
import sys

import pytest
import torch

from editgrpo.edit import (DEFAULT_LEXICON, EditConfig, LexiconExtractor, analyze_pair, apply_edits,
                           compare_entities, edit_group, replace_sentence_in_text, split_sentences)

EXT = LexiconExtractor(DEFAULT_LEXICON)
GOLD = "The heart is normal in size. There is a small right pleural effusion. No pneumothorax."


def test_split_and_replace():
    assert split_sentences("A b.  C d!\nE f") == ["A b.", "C d!", "E f"]
    assert replace_sentence_in_text("A b. C d.", "C d.", "X y.") == "A b. X y."
    assert replace_sentence_in_text("A b. C d.", "Zz.", "X y.") == "A b. C d."


def test_compare_entities_fp_fn_contradiction():
    gen = "The heart is normal in size. There is a left pneumothorax. No pleural effusion."
    a = analyze_pair(gen, GOLD, EXT)
    fp_spans = {e["text_span"] for e in a["false_positives"]}
    fn_spans = {e["text_span"] for e in a["false_negatives"]}
    # gen 'pneumothorax' (ABNORMALITY) vs gold 'pneumothorax' (NON-ABNORMALITY, negated): same concept
    # -> cos 1 >= 0.6 and a negative-class pair -> contradiction; effective score 1*0.543*0.3612 < 0.6 -> also FP.
    assert any(c["gen_entity"]["text_span"] == "pneumothorax" for c in a["contradictions"])
    assert "pneumothorax" in fp_spans
    # gold 'pleural effusion' is part of a contradiction with gen's negated effusion -> not an FN (:330-336)
    assert "pleural effusion" not in fn_spans
    # Faithful quirk (ISSUES.md #11): identical ANATOMY terms score cos 1 * affinity 0.2856 < 0.6, so the
    # original marks them as "false positives" and may delete the sentence that contains them.
    assert "heart" in fp_spans


def test_delete_then_add_sequence():
    gen = "There is a large left pneumothorax. Cardiomegaly is present."
    gold = "The lungs are clear. No pneumothorax."
    cfg = EditConfig(num_ops_per_para=3, similarity_threshold=0.6)
    a = analyze_pair(gen, gold, EXT)
    out, changed, log = apply_edits(gen, a, cfg, random.Random(0))
    ops = [l["operation"] for l in log if not l["operation"].endswith("failed")]
    assert changed
    # cycle 1: gen 'pneumothorax' vs gold negated 'pneumothorax' share a concept (cos 1 >= tau) -> replace the
    # FP sentence with "No pneumothorax."; cycle 2: cardiomegaly has no similar gold entity -> delete; cycle 3: add FN.
    assert ops[0] == "replace_fp_complex" and log[0]["details"]["new_sentence"] == "No pneumothorax."
    assert ops[1] == "delete_fp_simple" and "Cardiomegaly" not in out
    assert ops[2] == "add_fn" and "The lungs are clear." in out
    assert out == "No pneumothorax. The lungs are clear."


def test_replace_fp_complex_fires_when_similar():
    ext = LexiconExtractor({"effusion": ("ABNORMALITY", "effusion"), "fluid": ("ABNORMALITY", "effusion")})
    gen = "There is fluid on the right."
    gold = "There is a small effusion on the right."
    # gen 'fluid' is ABNORMALITY, gold 'effusion' ABNORMALITY, same concept -> cos 1 * 0.4276 < 0.6 -> FP;
    # replace_fp_complex: mean max cos = 1 >= tau*1 -> replaced by the gold sentence.
    a = analyze_pair(gen, gold, ext)
    out, changed, log = apply_edits(gen, a, EditConfig(), random.Random(0))
    assert changed and out == gold and log[0]["operation"] == "replace_fp_complex"


def test_tau_scaling_quirk_flag():
    ext = LexiconExtractor({"fluid": ("ABNORMALITY", "a"), "haze": ("ABNORMALITY", "b"),
                            "effusion": ("ABNORMALITY", "a"), "opacity": ("ABNORMALITY", "b")})
    gen = "There is fluid and haze."
    gold = "There is effusion and opacity."
    a = analyze_pair(gen, gold, ext)
    # FP sentence carries 2 entities; mean of max-cos = 1.0, but original bar = 0.6*2 = 1.2 -> never replaces.
    fp = a["false_positives"][0]
    fp["sentence_entities"] = [e for e in a["false_positives"] if e["sentence"] == fp["sentence"]]
    out, _, log = apply_edits(gen, a, EditConfig(num_ops_per_para=1), random.Random(0))
    assert log[0]["operation"] == "replace_fp_complex_failed" or out != gold
    out2, _, log2 = apply_edits(gen, a, EditConfig(num_ops_per_para=1, fix_tau_scaling=True), random.Random(0))
    assert out2 == gold and log2[0]["operation"] == "replace_fp_complex"


def test_edit_group_half_sources_overwrite_targets():
    G = 5
    gens = [f"Report {i}. There is a left pneumothorax." for i in range(G)]
    gts = [GOLD] * G
    out, rec = edit_group(gens, gts, G, EXT, EditConfig(), random.Random(1))
    assert len(rec) == G // 2  # 2 edits for G=5
    for r in rec:
        assert r["source"] != r["target"]
        assert out[r["source"]] == gens[r["source"]]        # source untouched
        assert out[r["target"]] != gens[r["target"]]        # target overwritten
        assert out[r["target"]].startswith(f"Report {r['source']}.")  # ... with the edited *source*
    assert sum(o == g for o, g in zip(out, gens)) == G - G // 2


@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "..", "verl", "utils", "RaTEscore_extractor.py")),
                    reason="original repo not present")
def test_against_original_process_text():
    """Load the original client function in isolation and compare on the same analysis dict."""
    path = os.path.join(os.path.dirname(__file__), "..", "..", "verl", "utils", "RaTEscore_extractor.py")
    src = open(path).read().replace("import requests\n", "")  # avoid the requests dependency
    mod = type(sys)("orig_rate"); exec(compile(src, path, "exec"), mod.__dict__)
    gen = "There is a large left pneumothorax. Cardiomegaly is present. The heart is enlarged."
    gold = "The lungs are clear. No pneumothorax. There is a small effusion."
    for seed in range(5):
        a = analyze_pair(gen, gold, EXT)
        tol = lambda e: dict(e, embedding=e["embedding"].tolist())  # the server serialised embeddings as lists
        a_list = {"contradictions": [dict(c, gen_entity=tol(c["gen_entity"]), ref_entity=tol(c["ref_entity"])) for c in a["contradictions"]],
                  "false_positives": [tol(e) for e in a["false_positives"]],
                  "false_negatives": [tol(e) for e in a["false_negatives"]],
                  "ref_entities_grouped_by_sentence": {s: [tol(e) for e in es] for s, es in a["ref_entities_grouped_by_sentence"].items()}}
        random.seed(seed); o_text, o_changed, o_log = mod.process_text_with_server_analysis(gen, a_list, num_operations=2, similarity_threshold=0.6)
        n_text, n_changed, n_log = apply_edits(gen, a, EditConfig(num_ops_per_para=2), random.Random(seed))
        assert (o_text, o_changed) == (n_text, n_changed)
        assert [l["operation"] for l in o_log] == [l["operation"] for l in n_log]
