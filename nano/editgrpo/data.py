"""Data pipeline for nano-EditGRPO.

Faithful port of ``verl/utils/dataset.py:58-79`` (image resize) and ``:186-262``
(record -> prompt/ids) without ``transformers`` / ``datasets`` / ``qwen_vl_utils``.

Record format (json list, LLaVA style)::

    {"conversations": [{"from": "human", "value": "<image><image> prompt"},
                       {"from": "gpt",   "value": "gold report"}],
     "image": ["a.jpg", "b.jpg"]}

Heavy pieces are injected: ``image_preprocess(pil, min_pixels, max_pixels)``
(defined in model.py) returns ``(pixel_values, grid_thw)`` for one image.
"""
from __future__ import annotations

import json
import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:  # torch is needed for tensors, but keep import errors readable
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    from tokenizers import Tokenizer as _HFTokenizer
except ImportError:  # pragma: no cover - graceful fallback
    _HFTokenizer = None

IMAGE_TOKEN = "<image>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOS_TOKEN = "<|endoftext|>"
DEFAULT_SYSTEM = "You are a helpful assistant."
MERGE_SIZE = 2  # spatial merge of Qwen2.5-VL: tokens per image = t*h*w / merge^2

DEFAULT_MIN_PIXELS = 262144
DEFAULT_MAX_PIXELS = 401408


# --------------------------------------------------------------------------- records
def load_records(path: str) -> List[Dict[str, Any]]:
    """Load the json-list dataset (dataset.py:186-196, json branch)."""
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a json list of records")
    return data


def record_prompt(record: Dict[str, Any]) -> str:
    """prompt_key == 'conversations' (str, or list -> element 0 'value'); dataset.py:204-210."""
    conv = record["conversations"]
    return conv if isinstance(conv, str) else conv[0]["value"]


def record_gold(record: Dict[str, Any]) -> str:
    """answer_key == 'conversations' -> element 1 'value'; dataset.py:259-262."""
    conv = record["conversations"]
    return conv if isinstance(conv, str) else conv[1]["value"]


def record_images(record: Dict[str, Any]) -> list:
    return list(record.get("image", []) or [])


# --------------------------------------------------------------------------- images
def process_image(image: Any, min_pixels: int = DEFAULT_MIN_PIXELS, max_pixels: int = DEFAULT_MAX_PIXELS):
    """Port of dataset.py:61-79. Accepts path, bytes, {'bytes':..} or a PIL image."""
    from io import BytesIO

    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, (bytes, bytearray)):
        image = Image.open(BytesIO(image))
    elif isinstance(image, str):
        image = Image.open(image)

    if image.width * image.height > max_pixels:
        f = math.sqrt(max_pixels / (image.width * image.height))
        image = image.resize((int(image.width * f), int(image.height * f)))
    if image.width * image.height < min_pixels:
        f = math.sqrt(min_pixels / (image.width * image.height))
        image = image.resize((int(image.width * f), int(image.height * f)))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


# --------------------------------------------------------------------------- prompt text
def split_prompt_parts(prompt: str) -> List[Dict[str, str]]:
    """Split on ``<image>`` and emit content parts in order (dataset.py:212-222)."""
    parts: List[Dict[str, str]] = []
    for i, content in enumerate(prompt.split(IMAGE_TOKEN)):
        if i != 0:
            parts.append({"type": "image"})
        if content:
            parts.append({"type": "text", "text": content})
    return parts


def image_pad_count(grid_thw: Sequence[int], merge_size: int = MERGE_SIZE) -> int:
    t, h, w = (int(x) for x in grid_thw)
    return t * h * w // (merge_size ** 2)


def build_prompt_text(
    prompt: str,
    tokenizer_chat_template: Optional[Callable[[List[dict]], str]] = None,
    grid_thw: Optional[Sequence[Sequence[int]]] = None,
    system_prompt: str = DEFAULT_SYSTEM,
    add_generation_prompt: bool = True,
) -> str:
    """Qwen2.5-VL chat string for one user turn with N images.

    Mirrors ``processor.apply_chat_template(messages, add_generation_prompt=True,
    tokenize=False)`` including the default system prompt that the HF template
    inserts when none is given. Each image becomes ``<|vision_start|><|image_pad|>
    <|vision_end|>``; if ``grid_thw`` (one row per image) is given, the single
    ``<|image_pad|>`` is expanded to ``t*h*w/4`` tokens as the HF processor does.

    ``tokenizer_chat_template``: optional callable ``messages -> str`` overriding the
    built-in template (e.g. a jinja renderer); it receives the message list.
    """
    parts = split_prompt_parts(prompt)
    if tokenizer_chat_template is not None:
        messages = [{"role": "user", "content": parts}]
        text = tokenizer_chat_template(messages)
    else:
        body = []
        for p in parts:
            body.append(f"{VISION_START}{IMAGE_PAD}{VISION_END}" if p["type"] == "image" else p["text"])
        text = f"{IM_START}system\n{system_prompt}{IM_END}\n{IM_START}user\n{''.join(body)}{IM_END}\n"
        if add_generation_prompt:
            text += f"{IM_START}assistant\n"
    if grid_thw is not None:
        n_img = text.count(IMAGE_PAD)
        if n_img != len(grid_thw):
            raise ValueError(f"{n_img} image placeholders in prompt but {len(grid_thw)} images")
        out, rest = [], text
        for g in grid_thw:
            head, rest = rest.split(IMAGE_PAD, 1)
            out.append(head + IMAGE_PAD * image_pad_count(g))
        out.append(rest)
        text = "".join(out)
    return text


# --------------------------------------------------------------------------- tokenizer
class ByteFallbackTokenizer:
    """Minimal stand-in used when ``tokenizers`` / tokenizer.json are unavailable.

    Bytes map to ids 0..255; known special strings to 256+. For CPU smoke tests only.
    """

    SPECIALS = [EOS_TOKEN, IM_START, IM_END, VISION_START, VISION_END, IMAGE_PAD]

    def __init__(self):
        self.special_ids = {s: 256 + i for i, s in enumerate(self.SPECIALS)}
        self.vocab_size = 256 + len(self.SPECIALS)

    def token_to_id(self, tok: str) -> Optional[int]:
        return self.special_ids.get(tok)

    def encode_ids(self, text: str) -> List[int]:
        ids: List[int] = []
        i = 0
        while i < len(text):
            for s, sid in self.special_ids.items():
                if text.startswith(s, i):
                    ids.append(sid)
                    i += len(s)
                    break
            else:
                ids.extend(text[i].encode("utf-8"))
                i += 1
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        inv = {v: k for k, v in self.special_ids.items()}
        out, buf = [], bytearray()
        for t in ids:
            if t in inv:
                out.append(buf.decode("utf-8", "ignore")); buf = bytearray(); out.append(inv[t])
            else:
                buf.append(t)
        out.append(buf.decode("utf-8", "ignore"))
        return "".join(out)


def load_tokenizer(path: Optional[str]):
    """Load tokenizer.json via ``tokenizers``; fall back to ByteFallbackTokenizer."""
    if path and _HFTokenizer is not None:
        return _HFTokenizer.from_file(path)
    return ByteFallbackTokenizer()


def encode_ids(tokenizer: Any, text: str) -> List[int]:
    """Encode without adding special tokens, for tokenizers.Tokenizer or the fallback."""
    if hasattr(tokenizer, "encode_ids"):
        return tokenizer.encode_ids(text)
    try:
        return list(tokenizer.encode(text, add_special_tokens=False).ids)
    except TypeError:
        out = tokenizer.encode(text)
        return list(out.ids if hasattr(out, "ids") else out)


def pad_token_id(tokenizer: Any) -> int:
    """Qwen2.5 pad token is <|endoftext|>."""
    tid = tokenizer.token_to_id(EOS_TOKEN)
    return 0 if tid is None else tid


# --------------------------------------------------------------------------- example
def make_example(
    record: Dict[str, Any],
    tokenizer: Any,
    image_preprocess: Callable[[Any, int, int], Tuple[Any, Any]],
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    max_prompt_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Record -> model inputs (dataset.py:196-262).

    ``image_preprocess(pil, min_pixels, max_pixels) -> (pixel_values, grid_thw)``
    where grid_thw is a length-3 (t,h,w) (or (1,3)) per image; pixel_values are
    concatenated over images, image_grid_thw is (n_images, 3).
    """
    prompt = record_prompt(record)
    paths = record_images(record)
    pix_list, grids = [], []
    for img in paths:
        pil = process_image(img, min_pixels, max_pixels)
        pv, g = image_preprocess(pil, min_pixels, max_pixels)
        g = torch.as_tensor(g).reshape(-1, 3)
        pix_list.append(torch.as_tensor(pv))
        grids.append(g)
    grid_thw = torch.cat(grids, 0) if grids else torch.zeros(0, 3, dtype=torch.long)
    pixel_values = torch.cat(pix_list, 0) if pix_list else None

    text = build_prompt_text(prompt, grid_thw=grid_thw.tolist() if grids else None)
    ids = encode_ids(tokenizer, text)
    if max_prompt_length is not None and len(ids) > max_prompt_length:
        # EasyR1 truncation='error' by default; we left-truncate-free: raise to stay faithful
        raise ValueError(f"prompt length {len(ids)} > max_prompt_length {max_prompt_length}")
    input_ids = torch.tensor(ids, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "pixel_values": pixel_values,
        "image_grid_thw": grid_thw,
        "ground_truth": record_gold(record),
        "prompt_text": text,
    }


# --------------------------------------------------------------------------- collate
def left_pad(seqs: List["torch.Tensor"], pad_value: int, length: Optional[int] = None) -> "torch.Tensor":
    """Left-pad 1-D tensors (torch_functional.pad_sequence_to_length, left_pad=True)."""
    length = max(len(s) for s in seqs) if length is None else length
    out = torch.full((len(seqs), length), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        if len(s):
            out[i, length - len(s):] = s
    return out


def collate_left_pad(examples: List[Dict[str, Any]], pad_id: int) -> Dict[str, Any]:
    """Batch examples: left-pad input_ids (pad_id) and attention_mask (0); concat images."""
    batch: Dict[str, Any] = {
        "input_ids": left_pad([e["input_ids"] for e in examples], pad_id),
        "attention_mask": left_pad([e["attention_mask"] for e in examples], 0),
        "ground_truth": [e["ground_truth"] for e in examples],
    }
    am = batch["attention_mask"]
    batch["position_ids"] = torch.clamp(am.cumsum(-1) - 1, min=0)  # text-only; M-RoPE is model-side
    pvs = [e["pixel_values"] for e in examples if e.get("pixel_values") is not None]
    batch["pixel_values"] = torch.cat(pvs, 0) if pvs else None
    grids = [e["image_grid_thw"] for e in examples if e["image_grid_thw"].numel()]
    batch["image_grid_thw"] = torch.cat(grids, 0) if grids else torch.zeros(0, 3, dtype=torch.long)
    return batch


# --------------------------------------------------------------------------- synthetic
# ~14 findings (CheXpert-style), each with a positive and negated phrasing.
LEXICON: List[Tuple[str, str, str]] = [
    ("atelectasis", "There is bibasilar atelectasis.", "No atelectasis is seen."),
    ("cardiomegaly", "The heart is enlarged.", "The heart size is normal."),
    ("consolidation", "There is right lower lobe consolidation.", "No focal consolidation."),
    ("edema", "Findings are consistent with pulmonary edema.", "No pulmonary edema."),
    ("pleural effusion", "A small left pleural effusion is present.", "No pleural effusion."),
    ("pneumonia", "Findings suggest pneumonia.", "No evidence of pneumonia."),
    ("pneumothorax", "There is a small right pneumothorax.", "No pneumothorax."),
    ("fracture", "An acute rib fracture is noted.", "No acute fracture."),
    ("lung opacity", "There is a patchy opacity in the left lung.", "The lungs are clear of opacity."),
    ("lung lesion", "A nodular lesion is seen in the right upper lobe.", "No suspicious lung lesion."),
    ("enlarged cardiomediastinum", "The mediastinum is widened.", "The mediastinal contours are normal."),
    ("pleural other", "There is pleural thickening.", "No pleural abnormality."),
    ("support devices", "A right central line is in place.", "No support devices are seen."),
    ("no finding", "No acute cardiopulmonary process.", "Acute findings are present."),
]

SYNTH_PROMPT = "<image>Generate the findings section of the radiology report for this chest X-ray."


class SyntheticCXRDataset:
    """Random tiny images + lexicon-built reports; yields records in the json format.

    ``record['image']`` holds PIL images (``process_image`` accepts them), and
    ``record['findings']`` lists the positive finding names for reward/edit tests.
    """

    def __init__(self, n: int, seed: int = 0, image_size: int = 64, n_images: int = 1):
        self.n = n
        self.seed = seed
        self.image_size = image_size
        self.n_images = n_images
        self.records = [self._make(i) for i in range(n)]

    def _make(self, idx: int) -> Dict[str, Any]:
        rng = random.Random(self.seed * 1_000_003 + idx)
        k = rng.randint(2, 5)
        chosen = rng.sample(range(len(LEXICON)), k)
        sents, pos = [], []
        for j in chosen:
            name, p, neg = LEXICON[j]
            if rng.random() < 0.5:
                sents.append(p); pos.append(name)
            else:
                sents.append(neg)
        imgs = []
        for _ in range(self.n_images):
            s = self.image_size
            data = bytes(rng.getrandbits(8) for _ in range(s * s * 3))
            imgs.append(Image.frombytes("RGB", (s, s), data))
        prompt = IMAGE_TOKEN * (self.n_images - 1) + SYNTH_PROMPT
        return {
            "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": " ".join(sents)}],
            "image": imgs,
            "findings": pos,
        }

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.records[i]
