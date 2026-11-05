"""Post-rollout sentence-level edits (the "Edit" in EditGRPO).

Faithful in-process port of two original pieces:
  * server half  flask_server/RaTEscore_extractor.py:205-384  (entity extraction + FP/FN/contradiction matching)
  * client half  verl/utils/RaTEscore_extractor.py:43-391     (sentence operations driven by that analysis)
  * group logic  verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py:654-790 (which rollouts get edited)

The only thing that changes is the transport: no Flask/JSON, the analysis dict is produced locally.
Entity extraction (RaTE-NER + BioLORD embeddings) is behind the `EntityExtractor` protocol so the core
needs only torch; `plugins/rate_extractor.py` provides the real models, `LexiconExtractor` is a cheap
CPU stand-in with the same output format.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import torch

# --------------------------------------------------------------------------------------------
# Constants copied from flask_server/RaTEscore_extractor.py:29-59 (these come from RaTEScore).
# --------------------------------------------------------------------------------------------
DEFAULT_AFFINITY_MATRIX = {
    "abnormality_abnormality": 0.4276119164393705, "abnormality_anatomy": 0.6240929990607657,
    "abnormality_disease": 0.0034478181112993847, "abnormality_non-abnormality": 0.5431049700217344,
    "abnormality_non-disease": 0.27005425386213877, "anatomy_abnormality": 0.7487824274337533,
    "anatomy_anatomy": 0.2856134859160784, "anatomy_disease": 0.4592143222158069,
    "anatomy_non-abnormality": 0.02097055139911715, "anatomy_non-disease": 0.00013736314126696204,
    "disease_abnormality": 0.8396510075734789, "disease_anatomy": 0.9950209388542061,
    "disease_disease": 0.8460555030578727, "disease_non-abnormality": 0.9820689020512646,
    "disease_non-disease": 0.3789136708096537, "non-abnormality_abnormality": 0.16546764653692908,
    "non-abnormality_anatomy": 0.018670610691852826, "non-abnormality_disease": 0.719397354576018,
    "non-abnormality_non-abnormality": 0.0009357166071730684, "non-abnormality_non-disease": 0.0927333564267591,
    "non-disease_abnormality": 0.7759420231214385, "non-disease_anatomy": 0.1839139293714062,
    "non-disease_disease": 0.10073046076318157, "non-disease_non-abnormality": 0.03860183811876373,
    "non-disease_non-disease": 0.34065681486566446,
}
AFFINITY = {(k.split("_")[0].upper(), k.split("_")[1].upper()): v for k, v in DEFAULT_AFFINITY_MATRIX.items()}
NEG_CLASS = {
    ("NON-DISEASE", "DISEASE"), ("NON-ABNORMALITY", "ABNORMALITY"),
    ("DISEASE", "NON-DISEASE"), ("ABNORMALITY", "NON-ABNORMALITY"),
    ("NON-DISEASE", "ABNORMALITY"), ("NON-ABNORMALITY", "DISEASE"),
    ("DISEASE", "NON-ABNORMALITY"), ("ABNORMALITY", "NON-DISEASE"),
}
NEG_WEIGHT = 0.3612           # flask_server/RaTEscore_extractor.py:58
SERVER_THRESHOLD = 0.6        # flask_server/RaTEscore_extractor.py:59 (FP/FN decision threshold)
EOS_STRINGS = ["<|endoftext|>", "<|end_of_text|>", "</s>", "<eos>", "<|eot_id|>", "[EOS]", "[END]", "<|im_end|>"]


@dataclass
class EditConfig:
    """Paper settings (examples/mimic_cxr_editgrpo_ops1_para_half_tau_0_6.sbatch:98-103)."""
    num_ops_per_para: int = 1            # extractor_num_ops_per_para
    num_para: int = -1                   # -1: edit G//2 sources -> overwrite G//2 targets per prompt; >=1: old in-place path
    similarity_threshold: float = 0.6    # tau used by replace_fp_complex (client side)
    server_threshold: float = SERVER_THRESHOLD
    neg_weight: float = NEG_WEIGHT
    operation_priority: Tuple[str, ...] = ("replace_fp_complex", "delete", "add")  # RaTEscore_extractor.py:278
    fix_tau_scaling: bool = False        # ISSUES.md #3; False = faithful `best >= tau * len(fp_entities)`


# --------------------------------------------------------------------------------------------
# Text helpers (verl/utils/RaTEscore_extractor.py:8-89)
# --------------------------------------------------------------------------------------------
def clean_eos_tokens(text: str) -> str:
    """RaTEscore_extractor.py:8-41: strip EOS-like strings, collapse whitespace."""
    if not text:
        return text
    for p in EOS_STRINGS:
        text = text.replace(p, " ")
    return " ".join(text.split()).strip()


def split_sentences(text: str) -> List[str]:
    """RaTEscore_extractor.py:43-60 (`get_entity_containing_sentences` without an entity)."""
    if not text:
        return []
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]
    if not sents and text.strip():
        sents = [text.strip()]
    return sents


def sentence_containing(text: str, entity_text: str) -> Optional[str]:
    """RaTEscore_extractor.py:62-72: first sentence containing the entity (case-insensitive)."""
    needle = entity_text.lower()
    for s in split_sentences(text):
        if needle in s.lower():
            return s
    return None


def replace_sentence_in_text(text: str, old: str, new: str) -> str:
    """RaTEscore_extractor.py:74-89: replace first exact sentence match, re-join with single spaces."""
    if not old or not new or old == new:
        return text
    out, replaced = [], False
    for s in split_sentences(text):
        if not replaced and s == old:
            out.append(new); replaced = True
        else:
            out.append(s)
    return " ".join(out).strip() if replaced else text


# --------------------------------------------------------------------------------------------
# Server half: entities and their comparison (flask_server/RaTEscore_extractor.py:205-384)
# --------------------------------------------------------------------------------------------
class EntityExtractor(Protocol):
    """Returns [{'text_span', 'type' (UPPER), 'embedding' (1-D tensor)}] for a text. Types are RaTE's:
    ABNORMALITY, NON-ABNORMALITY, DISEASE, NON-DISEASE, ANATOMY."""
    def __call__(self, text: str) -> List[Dict[str, Any]]: ...


def extract_entities_with_context(text: str, extractor: EntityExtractor) -> List[Dict[str, Any]]:
    """RaTEscore_extractor.py (server):205-240 — attach the containing sentence, drop entities without one."""
    if not text.strip():
        return []
    out = []
    for e in extractor(text):
        sent = sentence_containing(text, e["text_span"])
        if sent:
            out.append({"text_span": e["text_span"], "type": e["type"].upper(),
                        "embedding": torch.as_tensor(e["embedding"], dtype=torch.float32), "sentence": sent})
    return out


def _cos_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, p=2, dim=-1)
    b = torch.nn.functional.normalize(b, p=2, dim=-1)
    return a @ b.T


def compare_entities(gen: List[Dict], ref: List[Dict], threshold: float = SERVER_THRESHOLD,
                     neg_weight: float = NEG_WEIGHT) -> Tuple[List, List, List]:
    """flask_server/RaTEscore_extractor.py:242-356. Returns (contradictions, false_positives, false_negatives)."""
    contradictions: List[Dict] = []
    if not gen and not ref:
        return [], [], []
    if not ref:
        return [], list(gen), []
    if not gen:
        return [], [], list(ref)
    sim = _cos_matrix(torch.stack([e["embedding"] for e in gen]), torch.stack([e["embedding"] for e in ref]))

    def eff(i, j):
        g, r = gen[i], ref[j]
        w = AFFINITY.get((g["type"], r["type"]), 0.0)
        pen = neg_weight if ((g["type"], r["type"]) in NEG_CLASS or (r["type"], g["type"]) in NEG_CLASS) else 1.0
        return sim[i, j].item() * w * pen

    # gen -> ref (FPs and contradictions)   (:278-310)
    g2r = []
    for i in range(len(gen)):
        scores = [eff(i, j) for j in range(len(ref))]
        j = max(range(len(ref)), key=lambda jj: scores[jj])   # first max, like the `>` loop
        g2r.append(scores[j])
        pair = (gen[i]["type"], ref[j]["type"])
        if (pair in NEG_CLASS or pair[::-1] in NEG_CLASS) and sim[i, j].item() >= threshold:
            contradictions.append({"gen_entity": gen[i], "ref_entity": ref[j], "similarity": sim[i, j].item()})
    false_positives = [gen[i] for i in range(len(gen)) if g2r[i] < threshold]
    # ref -> gen (FNs)   (:312-336)
    false_negatives = []
    for j in range(len(ref)):
        best = max(eff(i, j) for i in range(len(gen)))
        if best < threshold:
            in_contr = any(c["ref_entity"]["text_span"] == ref[j]["text_span"] and
                           c["ref_entity"]["sentence"] == ref[j]["sentence"] for c in contradictions)
            if not in_contr:
                false_negatives.append(ref[j])
    return contradictions, false_positives, false_negatives


def analyze_pair(gen_text: str, gt_text: str, extractor: EntityExtractor, cfg: EditConfig = EditConfig()) -> Dict[str, Any]:
    """flask_server/RaTEscore_extractor.py:358-384 (`analyze_text_pair`), the former server response."""
    gen_e = extract_entities_with_context(gen_text, extractor)
    ref_e = extract_entities_with_context(gt_text, extractor)
    contr, fps, fns = compare_entities(gen_e, ref_e, cfg.server_threshold, cfg.neg_weight)
    grouped: Dict[str, List[Dict]] = {}
    for e in ref_e:
        grouped.setdefault(e["sentence"], []).append(e)
    return {"contradictions": contr, "false_positives": fps, "false_negatives": fns,
            "ref_entities_grouped_by_sentence": grouped}


# --------------------------------------------------------------------------------------------
# Client half: sentence operations (verl/utils/RaTEscore_extractor.py:91-391)
# --------------------------------------------------------------------------------------------
def delete_fp_sentence(text: str, fp: Dict) -> Tuple[str, Optional[Dict]]:
    """:112-125"""
    target = fp["sentence"]
    if not text.strip() or not target:
        return text, None
    sents = split_sentences(text)
    kept = [s for s in sents if s != target]
    if len(kept) == len(sents):
        return text, None
    return " ".join(kept).strip(), {"deleted_sentence": target, "fp_token": fp["text_span"]}


def add_fn_sentence(text: str, fn: Dict) -> Tuple[str, Optional[Dict]]:
    """:127-140 — append the gold sentence that carries the missed entity."""
    new = clean_eos_tokens(fn["sentence"])
    if not new or new in split_sentences(text):
        return text, None
    return (clean_eos_tokens(text).strip() + " " + new).strip(), {"added_sentence": new, "fn_token": fn["text_span"]}


def replace_fp_sentence_complex(text: str, fp: Dict, ref_grouped: Dict[str, List[Dict]], tau: float,
                                rng: random.Random, fix_tau_scaling: bool = False) -> Tuple[str, Optional[Dict]]:
    """:162-253 — eq. (2): pick the gold sentence maximising mean_{e in FP sentence} max_{e' in gold sentence} cos(e,e')."""
    fp_sent = fp["sentence"]
    if not fp_sent or not ref_grouped:
        return text, None
    fp_entities = [e for e in fp.get("sentence_entities", [fp]) if "embedding" in e]
    if not fp_entities:                                  # :178-192 random fallback (never hit with embeddings)
        cands = [s for s in ref_grouped if s != fp_sent]
        if cands:
            new = rng.choice(cands)
            return replace_sentence_in_text(text, fp_sent, new), {"old_sentence": fp_sent, "new_sentence": new,
                                                                  "selection": "random (no embeddings)"}
        return text, None
    best_score, best_sent = -float("inf"), None
    for ref_sent, ref_ents in ref_grouped.items():
        if ref_sent == fp_sent:
            continue
        ref_ents = [e for e in ref_ents if "embedding" in e]
        if not ref_ents:
            continue
        score = 0.0
        for fe in fp_entities:
            fe_emb = torch.as_tensor(fe["embedding"], dtype=torch.float32)
            max_sim = 0.0                                # :221 starts at 0, so negative cosines count as 0
            for re_ in ref_ents:
                sim = torch.nn.functional.cosine_similarity(fe_emb, torch.as_tensor(re_["embedding"], dtype=torch.float32), dim=0).item()
                max_sim = max(max_sim, sim)
            score += max_sim
        score /= len(fp_entities)                         # :236 mean over FP-sentence entities
        if score > best_score:
            best_score, best_sent = score, ref_sent
    # :240 original compares the *mean* against tau * len(fp_entities)  (ISSUES.md #3)
    bar = tau if fix_tau_scaling else tau * len(fp_entities)
    if best_sent and best_score >= bar:
        new = clean_eos_tokens(best_sent)
        out = replace_sentence_in_text(text, fp_sent, new)
        if out != text:
            return out, {"fp_trigger_token": fp["text_span"], "old_sentence": fp_sent, "new_sentence": new,
                         "similarity_score": best_score, "selection": "maximum similarity"}
    return text, None


def apply_edits(gen_text: str, analysis: Dict[str, Any], cfg: EditConfig = EditConfig(),
                rng: Optional[random.Random] = None) -> Tuple[str, bool, List[Dict]]:
    """verl/utils/RaTEscore_extractor.py:255-391 `process_text_with_server_analysis`.

    Up to `cfg.num_ops_per_para` cycles; each cycle performs the first operation that succeeds in priority
    order. Returns (edited_text, changed, ops_log)."""
    rng = rng or random
    if not gen_text.strip():
        return gen_text, False, []
    text = gen_text
    log: List[Dict] = []
    changed = False
    fps = analysis.get("false_positives", [])
    fns = analysis.get("false_negatives", [])
    grouped = analysis.get("ref_entities_grouped_by_sentence", {})
    processed: set = set()
    for _ in range(cfg.num_ops_per_para):
        done = False
        for op in cfg.operation_priority:
            if done:
                break
            if op == "replace_fp_complex" and fps:
                for fp in fps:
                    if fp["sentence"] in processed:
                        continue
                    new, info = replace_fp_sentence_complex(text, fp, grouped, cfg.similarity_threshold, rng, cfg.fix_tau_scaling)
                    if info:
                        text, changed, done = new, True, True
                        log.append({"operation": "replace_fp_complex", "details": info})
                        processed.add(fp["sentence"]); processed.add(info["new_sentence"])
                        break
                    log.append({"operation": "replace_fp_complex_failed", "details": {"attempted_sentence": fp["sentence"]}})
            elif op == "delete" and fps:
                for fp in fps:
                    if fp["sentence"] in processed:
                        continue
                    new, info = delete_fp_sentence(text, fp)
                    if info:
                        text, changed, done = new, True, True
                        log.append({"operation": "delete_fp_simple", "details": info})
                        processed.add(fp["sentence"])
                        break
            elif op == "add" and fns:
                avail = [fn for fn in fns if fn["sentence"] not in processed]
                if avail:
                    new, info = add_fn_sentence(text, rng.choice(avail))
                    if info:
                        text, changed, done = new, True, True
                        log.append({"operation": "add_fn", "details": info})
                        processed.add(info["added_sentence"])
        if not done:
            break
    if not text.strip() and fns:                          # :365-381 empty-paragraph fallback
        avail = [fn for fn in fns if fn["sentence"] not in processed]
        if avail:
            new, info = add_fn_sentence(text, rng.choice(avail))
            if info:
                text, changed = new, True
                log.append({"operation": "add_fn_empty_fallback", "details": info})
    return clean_eos_tokens(text), changed, log


# --------------------------------------------------------------------------------------------
# Group logic: which rollouts are edited (vllm_rollout_spmd.py:641-790)
# --------------------------------------------------------------------------------------------
def edit_group(gen_texts: Sequence[str], gt_texts: Sequence[str], group_size: int, extractor: EntityExtractor,
               cfg: EditConfig = EditConfig(), rng: Optional[random.Random] = None) -> Tuple[List[str], List[Dict]]:
    """Apply EditGRPO edits to a flat batch of rollouts laid out prompt-major (B*G, G consecutive per prompt).

    `gt_texts` has one gold report per rollout (already repeated G times, as `non_tensor_batch["ground_truth"]`).
    num_para == -1 (paper): per prompt, shuffle indices, first max(1, G//2) are sources, next G//2 are targets;
    the edited source text *overwrites the target slot* (:654-706). num_para >= 1: edit `num_para` random
    rollouts of the whole batch in place (:731-790).
    Returns (texts, edit_records) where each record = {source, target, ops}."""
    rng = rng or random.Random(0)
    texts = list(gen_texts)
    records: List[Dict] = []
    analyses = [analyze_pair(g, clean_eos_tokens(t), extractor, cfg) for g, t in zip(gen_texts, gt_texts)]
    n_prompts = len(texts) // group_size
    if cfg.num_para == -1:
        n_src = max(1, group_size // 2)
        for p in range(n_prompts):
            idx = list(range(p * group_size, (p + 1) * group_size))
            rng.shuffle(idx)
            sources, targets = idx[:n_src], idx[n_src:2 * n_src]
            for s, t in zip(sources, targets):
                edited, changed, ops = apply_edits(gen_texts[s], analyses[s], cfg, rng)
                if changed:
                    texts[t] = clean_eos_tokens(edited)
                    records.append({"source": s, "target": t, "ops": ops})
    else:
        chosen = rng.sample(range(len(texts)), min(cfg.num_para, len(texts)))
        for i in chosen:
            edited, changed, ops = apply_edits(gen_texts[i], analyses[i], cfg, rng)
            texts[i] = clean_eos_tokens(edited)
            if changed:
                records.append({"source": i, "target": i, "ops": ops})
    return texts, records


# --------------------------------------------------------------------------------------------
# Cheap CPU extractor with the RaTE output format (tests / smoke runs). Real one: plugins/rate_extractor.py
# --------------------------------------------------------------------------------------------
class LexiconExtractor:
    """Dictionary NER: `lexicon` maps lowercase term -> (TYPE, concept). Embeddings are deterministic
    hashed vectors per *concept*, so synonyms share an embedding and different concepts are near-orthogonal.
    Negation words in the same sentence flip ABNORMALITY<->NON-ABNORMALITY, DISEASE<->NON-DISEASE."""
    NEG = re.compile(r"\b(no|without|negative for|free of|absent|clear of)\b", re.I)

    def __init__(self, lexicon: Dict[str, Tuple[str, str]], dim: int = 64):
        self.lexicon = {k.lower(): v for k, v in lexicon.items()}
        self.dim = dim

    def embed(self, concept: str) -> torch.Tensor:
        g = torch.Generator().manual_seed(abs(hash(concept)) % (2 ** 31))
        return torch.randn(self.dim, generator=g)

    def __call__(self, text: str) -> List[Dict[str, Any]]:
        out = []
        for sent in split_sentences(text):
            neg = bool(self.NEG.search(sent))
            low = sent.lower()
            for term, (typ, concept) in self.lexicon.items():
                if re.search(r"\b" + re.escape(term) + r"\b", low):
                    t = typ.upper()
                    if neg and t in ("ABNORMALITY", "DISEASE"):
                        t = "NON-" + t
                    elif neg and t.startswith("NON-"):
                        t = t[4:]
                    out.append({"text_span": term, "type": t, "embedding": self.embed(concept)})
        return out


DEFAULT_LEXICON: Dict[str, Tuple[str, str]] = {
    "pleural effusion": ("ABNORMALITY", "effusion"), "effusion": ("ABNORMALITY", "effusion"),
    "pneumothorax": ("ABNORMALITY", "pneumothorax"), "consolidation": ("ABNORMALITY", "consolidation"),
    "opacity": ("ABNORMALITY", "opacity"), "opacities": ("ABNORMALITY", "opacity"),
    "cardiomegaly": ("DISEASE", "cardiomegaly"), "enlarged cardiac silhouette": ("DISEASE", "cardiomegaly"),
    "edema": ("DISEASE", "edema"), "pneumonia": ("DISEASE", "pneumonia"), "atelectasis": ("ABNORMALITY", "atelectasis"),
    "fracture": ("ABNORMALITY", "fracture"), "lungs": ("ANATOMY", "lungs"), "heart": ("ANATOMY", "heart"),
    "mediastinum": ("ANATOMY", "mediastinum"), "support devices": ("ABNORMALITY", "devices"),
}
