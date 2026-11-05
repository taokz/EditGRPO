"""Evaluation helpers.

ISSUES: the original eval (eval/lightweight_eval.py:245-341) deduplicates
sentences (exact + Jaccard>0.8 + subset>0.9) in hyps AND refs before scoring.
That hides exactly the repetition failure mode we want to measure and inflates
metrics.  So `evaluate` reports RAW metrics by default plus repetition stats;
the legacy cleaned numbers are opt-in via `legacy_eval_dedup=True`.
"""
from __future__ import annotations

import re
from typing import Dict, List

from .reward import SumReward, split_sentences


def legacy_clean_text(text: str) -> str:
    """Port of eval/lightweight_eval.py:245-341 `clean_text` (logic unchanged)."""
    text = text.strip()                                   # :247
    text = re.sub(r'\s+', ' ', text)                      # :248
    text = re.sub(r'Indication:.*', '', text).strip()     # :249
    text = re.sub(r'<\|endoftext\|>+', '', text)          # :252
    text = re.sub(r'\s+', ' ', text).strip()              # :255

    # :259 sentence split (also splits "x.Y" with no space)
    sentences = re.split(r'(?<=[.!?])\s+|(?<=\.)\s*(?=[A-Z])', text)
    temp_sentences = []
    for sent in sentences:                                # :263-266
        temp_sentences.extend(re.split(r'(?<=[a-z])\.(?=\s*[A-Z])', sent))
    sentences = temp_sentences

    unique_sentences = []
    seen_normalized = set()
    for sentence in sentences:                            # :273
        sentence = sentence.strip()
        if not sentence:
            continue
        if not re.search(r'[.!?]$', sentence):            # :279
            sentence += '.'
        normalized = re.sub(r'[.!?,:;]+$', '', sentence).strip().lower()   # :283
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        if len(normalized.split()) < 2:                   # :288 short-sentence drop
            continue
        if normalized in seen_normalized:                 # :292 exact dup
            continue
        is_duplicate = False
        words = set(normalized.split())
        for seen_norm in seen_normalized:                 # :299 fuzzy dedup
            seen_words = set(seen_norm.split())
            if words and seen_words:
                inter = len(words & seen_words)
                union = len(words | seen_words)
                if inter / union > 0.8:                   # :309 Jaccard
                    is_duplicate = True
                    break
                smaller = words if len(words) <= len(seen_words) else seen_words
                larger = seen_words if len(words) <= len(seen_words) else words
                if len(smaller & larger) / len(smaller) > 0.9:   # :318 subset
                    is_duplicate = True
                    break
        if not is_duplicate and normalized:
            unique_sentences.append(sentence)
            seen_normalized.add(normalized)

    text = ' '.join(unique_sentences)                     # :328
    if text and not re.search(r'[.!?]$', text):           # :331
        text += '.'

    boxed_match = re.search(r'\\boxed\{(.*?)\}', text, re.DOTALL)   # :335
    if boxed_match:
        box = re.sub(r'\s+', ' ', boxed_match.group(1).strip())
        return box if box else "[EMPTY]"
    return text if text else "[EMPTY]"                    # :341


def legacy_clean_text_green(text: str) -> str:
    """Port of eval/green_eval.py:19-60 `clean_text` (exact dedup only)."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Indication:.*', '', text).strip()
    text = text.replace("<|endoftext|>", "")
    sentences = re.split(r'(?<=[.!?])\s+', text)
    unique, seen = [], set()
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        normalized = re.sub(r'[.!?]+$', '', sentence).strip().lower()
        if normalized and normalized not in seen:
            unique.append(sentence)
            seen.add(normalized)
    text = ' '.join(unique)
    if text and not re.search(r'[.!?]$', text):
        text += '.'
    boxed_match = re.search(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    if boxed_match:
        box = re.sub(r'\s+', ' ', boxed_match.group(1).strip())
        return box if box else "[EMPTY]"
    return text if text else "[EMPTY]"


def _norm(s: str) -> str:
    return " ".join(re.findall(r"\w+", s.lower()))


def repetition_stats(text: str) -> Dict[str, float]:
    """repeated_sentence_rate: fraction of sentences exactly equal (after
    normalisation) to an earlier one. repeated_4gram_rate: fraction of word
    4-grams already seen earlier in the text. Both 0.0 when undefined."""
    sents = [n for n in (_norm(s) for s in split_sentences(text)) if n]
    seen, rep = set(), 0
    for s in sents:
        if s in seen:
            rep += 1
        seen.add(s)
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    seen_g, rep_g = set(), 0
    for g in grams:
        if g in seen_g:
            rep_g += 1
        seen_g.add(g)
    return {
        "repeated_sentence_rate": rep / len(sents) if sents else 0.0,
        "repeated_4gram_rate": rep_g / len(grams) if grams else 0.0,
        "n_sentences": len(sents),
        "n_words": len(words),
    }


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _component_means(reward: SumReward, hyps, refs) -> Dict[str, float]:
    total, parts = reward.score_components(hyps, refs)
    out = {k: _mean(v) for k, v in parts.items()}
    out["total"] = _mean(total)
    return out


def evaluate(hyps: List[str], refs: List[str], reward: SumReward,
             legacy_eval_dedup: bool = False) -> Dict[str, Dict[str, float]]:
    """raw: component means on untouched texts; repetition: mean stats of hyps;
    legacy (only if legacy_eval_dedup): means after legacy_clean_text on BOTH
    hyps and refs, as the original eval does."""
    res = {"raw": _component_means(reward, hyps, refs)}
    stats = [repetition_stats(h) for h in hyps]
    res["repetition"] = {k: _mean([s[k] for s in stats]) for k in
                         ("repeated_sentence_rate", "repeated_4gram_rate", "n_sentences", "n_words")}
    if legacy_eval_dedup:
        res["legacy"] = _component_means(reward, [legacy_clean_text(h) for h in hyps],
                                         [legacy_clean_text(r) for r in refs])
    return res
