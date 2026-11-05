"""Real RaTE entity extractor (optional plugin): RaTE-NER + BioLORD embeddings, exactly as
flask_server/RaTEscore_extractor.py:122-203 but in-process. Requires `transformers` (lazy import)."""
from __future__ import annotations

from typing import Any, Dict, List

import torch

NER_MODEL = "Angelakeke/RaTE-NER-Deberta"       # flask_server/RaTEscore_extractor.py:25
SYN_MODEL = "FremyCompany/BioLORD-2023-C"       # :26


class RaTEExtractor:
    def __init__(self, ner_model: str = NER_MODEL, syn_model: str = SYN_MODEL, device: str | None = None):
        try:
            from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError("RaTEExtractor needs `pip install transformers` (plugin only; the core does not)") from e
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.idx2label = AutoConfig.from_pretrained(ner_model).id2label
        self.ner_tok = AutoTokenizer.from_pretrained(ner_model)
        self.ner = AutoModelForTokenClassification.from_pretrained(ner_model).eval().to(self.device)
        self.syn_tok = AutoTokenizer.from_pretrained(syn_model)
        self.syn = AutoModel.from_pretrained(syn_model).eval().to(self.device)

    @torch.no_grad()
    def _ner(self, text: str) -> List[Dict[str, str]]:
        """:141-192 — BIO/BIES decoding over word-pieces, joined with convert_tokens_to_string."""
        inputs = self.ner_tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        pred = self.ner(**inputs).logits.argmax(-1)[0]
        tokens = self.ner_tok.convert_ids_to_tokens(inputs["input_ids"][0].cpu())
        labels = [self.idx2label.get(p.item(), "O") for p in pred]
        ents, cur, cur_lab = [], [], None
        skip = {self.ner_tok.cls_token, self.ner_tok.sep_token, self.ner_tok.pad_token}
        for tok, lab in zip(tokens, labels):
            if tok in skip:
                continue
            if lab.startswith("B-"):
                if cur:
                    ents.append({"text": self.ner_tok.convert_tokens_to_string(cur), "label": cur_lab})
                cur, cur_lab = [tok], lab[2:]
            elif lab.startswith("I-") and cur_lab == lab[2:]:
                cur.append(tok)
            else:
                if cur:
                    ents.append({"text": self.ner_tok.convert_tokens_to_string(cur), "label": cur_lab})
                    cur, cur_lab = [], None
                if lab != "O" and not lab.startswith("I-"):
                    cur, cur_lab = [tok], (lab[2:] if lab.startswith("S-") else lab)
        if cur:
            ents.append({"text": self.ner_tok.convert_tokens_to_string(cur), "label": cur_lab})
        return [e for e in ents if e["text"].strip() and e["label"] is not None]

    @torch.no_grad()
    def _embed(self, texts: List[str]) -> torch.Tensor:
        """:194-203 — mean-pooled last hidden state."""
        inputs = self.syn_tok(texts, padding=True, truncation=True, return_tensors="pt", max_length=128).to(self.device)
        return self.syn(**inputs).last_hidden_state.mean(dim=1).cpu()

    def __call__(self, text: str) -> List[Dict[str, Any]]:
        ents = self._ner(text)
        if not ents:
            return []
        emb = self._embed([e["text"] for e in ents])
        return [{"text_span": e["text"], "type": e["label"].upper(), "embedding": emb[i]} for i, e in enumerate(ents)]
