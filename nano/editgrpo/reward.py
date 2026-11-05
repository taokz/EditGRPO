"""Rewards for nano EditGRPO.

Faithful to the original Flask reward server and its client:
  * flask_server/all_flask_server.py:92-250  (components, sum combination)
  * verl/utils/reward_score/rad_scores.py    (client; one hyp per request)
  * verl/workers/reward/custom.py:141-161    (decode w/ skip_special_tokens,
                                              scalar reward per sample)

Per-sample semantics
--------------------
The server returns *batch means* (e.g. F1RadGraph averages over the request),
but the client sends ONE (hyp, ref) pair per request (custom.py:141-161), so the
original reward is per-sample.  Here every `Reward.score` returns a per-sample
list, which reproduces that behaviour exactly while allowing batched calls.

Deliberate deviation (see ISSUES): the original client silently returned 0.0 on
any server error / non-200 / exception (rad_scores.py:119-193), which turns a
crashed scorer into a "bad sample" signal.  We RAISE instead.

Core deps are torch/numpy only; radgraph / f1chexbert / RaTEScore are lazy
imports inside the clinical plugin classes.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

DEFAULT_COMPONENTS = ["radgraphf1", "cxb14micro", "ratescore"]
# Paper run: examples/mimic_cxr_editgrpo_ops1_para_half_tau_0_6.sbatch:105-108
DEFAULT_RESPONSE_TYPE = "free_form"


# --------------------------------------------------------------------------
# Server helpers, ported verbatim in behaviour
# --------------------------------------------------------------------------
def extract_boxed_content(text: str) -> str:
    """Content of the LAST ``\\boxed{...}`` with brace matching, else "None".

    Port of all_flask_server.py:20-42 (note the literal string "None" fallback).
    """
    depth = 0
    start_pos = text.rfind(r"\boxed{")
    end_pos = -1
    if start_pos != -1:
        content = text[start_pos + len(r"\boxed{"):]
        for i, char in enumerate(content):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            if depth == -1:
                end_pos = i
                break
    if end_pos != -1:
        return content[:end_pos].strip()
    return "None"


def format_reward(predict_str: str, type: str = "short") -> float:
    """Port of easyr1_math_format_reward (all_flask_server.py:60-80)."""
    if type == "short":
        pattern = re.compile(r"\\boxed\{.*\}.*", re.DOTALL)
    elif type == "cot":
        pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    elif type == "free_form":
        return 1.0
    else:
        return 0.0  # server printed "no implementation" and returned 0.0
    return 1.0 if re.fullmatch(pattern, predict_str) else 0.0


# --------------------------------------------------------------------------
# Protocol + helpers
# --------------------------------------------------------------------------
@runtime_checkable
class Reward(Protocol):
    name: str

    def score(self, hyps: List[str], refs: List[str]) -> List[float]:
        """Per-sample scores, len == len(hyps)."""
        ...


def _check(hyps, refs):
    if len(hyps) != len(refs):
        raise ValueError(f"hyps/refs length mismatch: {len(hyps)} vs {len(refs)}")


_PUNCT = re.compile(r"[^\w\s]")


def _tokens(text: str) -> List[str]:
    return _PUNCT.sub(" ", text.lower()).split()


def _f1_from_counts(overlap: int, n_h: int, n_r: int) -> float:
    if n_h == 0 and n_r == 0:
        return 1.0
    if overlap == 0:
        return 0.0
    p, r = overlap / n_h, overlap / n_r
    return 2 * p * r / (p + r)


# --------------------------------------------------------------------------
# Cheap deterministic rewards (CPU tests / smoke runs)
# --------------------------------------------------------------------------
class TokenF1Reward:
    """Bag-of-words F1 after lowercasing and punctuation stripping.
    Stand-in for RadGraph F1. Both empty -> 1.0, one empty -> 0.0."""
    name = "token_f1"

    def score(self, hyps, refs):
        _check(hyps, refs)
        out = []
        for h, r in zip(hyps, refs):
            ch, cr = Counter(_tokens(h)), Counter(_tokens(r))
            out.append(_f1_from_counts(sum((ch & cr).values()), sum(ch.values()), sum(cr.values())))
        return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def _norm_sentence(s: str) -> str:
    return " ".join(_tokens(s))


class SentenceSetF1Reward:
    """F1 over the SET of normalised sentences (exact match). Stand-in for
    RaTEScore. Being a set, repeating a sentence neither helps nor hurts."""
    name = "sentence_set_f1"

    def score(self, hyps, refs):
        _check(hyps, refs)
        out = []
        for h, r in zip(hyps, refs):
            sh = {_norm_sentence(s) for s in split_sentences(h)} - {""}
            sr = {_norm_sentence(s) for s in split_sentences(r)} - {""}
            out.append(_f1_from_counts(len(sh & sr), len(sh), len(sr)))
        return out


# ~14 CheXbert findings -> label (several surface terms per label).
DEFAULT_CXR_LEXICON: Dict[str, str] = {
    "enlarged cardiomediastinum": "enlarged_cardiomediastinum",
    "widened mediastinum": "enlarged_cardiomediastinum",
    "cardiomegaly": "cardiomegaly",
    "enlarged heart": "cardiomegaly",
    "cardiac enlargement": "cardiomegaly",
    "lung opacity": "lung_opacity",
    "opacity": "lung_opacity",
    "opacities": "lung_opacity",
    "lung lesion": "lung_lesion",
    "nodule": "lung_lesion",
    "mass": "lung_lesion",
    "edema": "edema",
    "consolidation": "consolidation",
    "pneumonia": "pneumonia",
    "atelectasis": "atelectasis",
    "pneumothorax": "pneumothorax",
    "pleural effusion": "pleural_effusion",
    "effusion": "pleural_effusion",
    "effusions": "pleural_effusion",
    "pleural other": "pleural_other",
    "pleural thickening": "pleural_other",
    "fracture": "fracture",
    "support devices": "support_devices",
    "pacemaker": "support_devices",
    "catheter": "support_devices",
    "tube": "support_devices",
}

_NEG = re.compile(r"\bno\b|\bwithout\b|\bnegative for\b")


class LexiconF1Reward:
    """Approximate CheXbert-14 micro label F1 with a term lexicon.

    A label is *positive* for a report if any of its terms occurs in a
    sentence that has no negation cue ("no", "without", "negative for").
    Per-sample F1 = 2|H&R| / (|H|+|R|) over positive labels; both empty -> 1.0
    (mirrors "No Finding" agreement).
    """
    name = "lexicon_f1"

    def __init__(self, lexicon: Optional[Dict[str, str]] = None):
        self.lexicon = dict(DEFAULT_CXR_LEXICON if lexicon is None else lexicon)
        self._pats = [(re.compile(r"\b" + re.escape(t.lower()) + r"\b"), l) for t, l in self.lexicon.items()]

    def labels(self, text: str) -> set:
        pos = set()
        for sent in split_sentences(text):
            s = sent.lower()
            if _NEG.search(s):
                continue
            for pat, label in self._pats:
                if pat.search(s):
                    pos.add(label)
        return pos

    def score(self, hyps, refs):
        _check(hyps, refs)
        out = []
        for h, r in zip(hyps, refs):
            lh, lr = self.labels(h), self.labels(r)
            out.append(_f1_from_counts(len(lh & lr), len(lh), len(lr)))
        return out


class FormatReward:
    """Format component (all_flask_server.py:60-80, 174-175)."""
    name = "format"

    def __init__(self, response_type: str = DEFAULT_RESPONSE_TYPE):
        self.response_type = response_type

    def score(self, hyps, refs):
        _check(hyps, refs)
        return [format_reward(h, self.response_type) for h in hyps]


# --------------------------------------------------------------------------
# Clinical plugins (lazy imports)
# --------------------------------------------------------------------------
def _missing(pkg: str, what: str, err: Exception):
    return ImportError(
        f"{what} needs the '{pkg}' package ({err}). Install with `pip install {pkg}`. "
        "The original repo uses patched copies under modified_lib/ (see modified_lib/ "
        "for the patch); results match the paper only with those patches. "
        "For CPU tests use make_reward(..., cheap=True)."
    )


def _first(x):
    """Server idiom: wrap scalars / take first element of list-likes
    (all_flask_server.py:131-133, 137-141)."""
    while isinstance(x, (list, tuple)):
        x = x[0]
    return float(x)


class RadGraphF1Reward:
    """RadGraph partial-level F1 (all_flask_server.py:15,125-133).

    F1RadGraph(reward_level="all") returns (scores, ...); scores[1] is the
    partial triple (precision, recall, f1) -> index [2] of that is F1.
    Called per pair so that the result is per-sample (client behaviour).
    """
    name = "radgraphf1"

    def __init__(self, device: str = "cpu", **kwargs):
        try:
            from radgraph import F1RadGraph
        except ImportError as e:
            raise _missing("radgraph", "RadGraphF1Reward", e)
        # model_type left at the library default, as in the original (L15)
        self._m = F1RadGraph(reward_level="all", device=device, **kwargs)

    def score(self, hyps, refs):
        _check(hyps, refs)
        out = []
        for h, r in zip(hyps, refs):
            scores, _, _, _ = self._m(hyps=[h], refs=[r])
            out.append(_first(scores[1][2]))  # partial level, F1
        return out


class CheXbert14MicroReward:
    """CheXbert-14 micro-avg F1 (all_flask_server.py:17,150-153)."""
    name = "cxb14micro"

    def __init__(self, device: str = "cpu", **kwargs):
        try:
            from f1chexbert import F1CheXbert
        except ImportError as e:
            raise _missing("f1chexbert", "CheXbert14MicroReward", e)
        self._m = F1CheXbert(device=device, **kwargs)

    def score(self, hyps, refs):
        _check(hyps, refs)
        out = []
        for h, r in zip(hyps, refs):
            _acc, _, class_report_14, _ = self._m(hyps=[h], refs=[r])
            out.append(_first(class_report_14["micro avg"]["f1-score"]))
        return out


class RaTEScoreReward:
    """RaTEScore (all_flask_server.py:18,126,137): compute_score returns
    (mean, std)-like; element [0] is the mean. `device` is accepted for API
    symmetry; RaTEScore() chooses its own device in the original."""
    name = "ratescore"

    def __init__(self, device: str = "cpu", **kwargs):
        try:
            from RaTEScore import RaTEScore
        except ImportError as e:
            raise _missing("RaTEScore", "RaTEScoreReward", e)
        self.device = device
        self._m = RaTEScore(**kwargs)

    def score(self, hyps, refs):
        _check(hyps, refs)
        return [_first(self._m.compute_score([h], [r])[0]) for h, r in zip(hyps, refs)]


# --------------------------------------------------------------------------
# Combination
# --------------------------------------------------------------------------
class SumReward:
    """Unweighted sum of component scores per sample
    (all_flask_server.py:203-235).

    * ``format`` is added only when response_type != "free_form" (L209-210,
      219-221); it is still reported in the component dict.
    * For response_type in {"short","cot"} the content components see
      ``extract_boxed_content(hyp)`` (L123-124, 148-149); format sees the raw hyp.
    * ``cxb_bal_acc`` (/14) is not ported (needs class_frequencies; unused by
      the paper run).
    """
    name = "sum"

    def __init__(self, components: Sequence[Reward], response_type: str = DEFAULT_RESPONSE_TYPE):
        self.components = list(components)
        self.response_type = response_type

    def score_components(self, hyps: List[str], refs: List[str]) -> Tuple[List[float], Dict[str, List[float]]]:
        _check(hyps, refs)
        content_hyps = hyps if self.response_type == "free_form" else [extract_boxed_content(h) for h in hyps]
        parts: Dict[str, List[float]] = {}
        total = [0.0] * len(hyps)
        for c in self.components:
            if isinstance(c, FormatReward):
                s = c.score(hyps, refs)
                add = self.response_type != "free_form"
            else:
                s = c.score(content_hyps, refs)
                add = True
            parts[c.name] = list(s)
            if add:
                total = [t + x for t, x in zip(total, s)]
        return total, parts

    def score(self, hyps, refs):
        return self.score_components(hyps, refs)[0]


_CLINICAL = {
    "radgraphf1": RadGraphF1Reward,
    "cxb14micro": CheXbert14MicroReward,
    "ratescore": RaTEScoreReward,
}


def make_reward(names: Optional[List[str]] = None, response_type: str = DEFAULT_RESPONSE_TYPE,
                device: str = "cpu", cheap: bool = False) -> SumReward:
    """Factory. ``cheap=True`` swaps clinical models for CPU stand-ins:
    radgraphf1->TokenF1, cxb14micro->LexiconF1, ratescore->SentenceSetF1.
    Component names keep their original names so logs line up."""
    names = list(DEFAULT_COMPONENTS if names is None else names)
    comps: List[Reward] = []
    for n in names:
        if n == "format":
            comps.append(FormatReward(response_type))
        elif cheap and n in _CLINICAL:
            impl = {"radgraphf1": TokenF1Reward, "cxb14micro": LexiconF1Reward,
                    "ratescore": SentenceSetF1Reward}[n]()
            impl.name = n
            comps.append(impl)
        elif n in _CLINICAL:
            comps.append(_CLINICAL[n](device=device))
        else:
            raise ValueError(f"unknown reward component {n!r}; known: {sorted(_CLINICAL) + ['format']}")
    return SumReward(comps, response_type)
