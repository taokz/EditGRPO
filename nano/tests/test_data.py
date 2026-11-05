import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from editgrpo import data as D  # noqa: E402

TINY = [
    {
        "conversations": [
            {"from": "human", "value": "<image><image>Write the report."},
            {"from": "gpt", "value": "No acute disease."},
        ],
        "image": ["a.jpg", "b.jpg"],
    }
]


def fake_preprocess(pil, mn, mx):
    # 28x28 patches grid for a 14px patch size: grid_h*grid_w patches
    h, w = pil.height // 14, pil.width // 14
    return torch.zeros(h * w, 4), (1, h, w)


def test_load_records(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps(TINY))
    recs = D.load_records(str(p))
    assert D.record_prompt(recs[0]) == "<image><image>Write the report."
    assert D.record_gold(recs[0]) == "No acute disease."
    assert D.record_images(recs[0]) == ["a.jpg", "b.jpg"]


def test_prompt_text_exact():
    t = D.build_prompt_text("<image><image>Hi", grid_thw=[(1, 4, 4), (1, 2, 4)])
    pv = "<|vision_start|>{}<|vision_end|>"
    expect = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        + pv.format("<|image_pad|>" * 4) + pv.format("<|image_pad|>" * 2) + "Hi<|im_end|>\n<|im_start|>assistant\n"
    )
    assert t == expect
    # unexpanded
    assert D.build_prompt_text("a<image>b").count("<|image_pad|>") == 1
    with pytest.raises(ValueError):
        D.build_prompt_text("<image>", grid_thw=[(1, 2, 2), (1, 2, 2)])


def test_make_example_fallback_tokenizer():
    tok = D.ByteFallbackTokenizer()
    ds = D.SyntheticCXRDataset(2, seed=1, image_size=64, n_images=2)
    ex = D.make_example(ds[0], tok, fake_preprocess, min_pixels=64 * 64, max_pixels=64 * 64)
    assert ex["image_grid_thw"].shape == (2, 3)
    n_pad = int(sum(g[0] * g[1] * g[2] for g in ex["image_grid_thw"].tolist())) // 4
    assert (ex["input_ids"] == tok.token_to_id("<|image_pad|>")).sum().item() == n_pad
    assert ex["attention_mask"].shape == ex["input_ids"].shape
    assert ex["pixel_values"].shape[0] == sum(h * w for _, h, w in ex["image_grid_thw"].tolist())
    assert ex["ground_truth"] == D.record_gold(ds[0])


def test_synthetic_dataset():
    a, b = D.SyntheticCXRDataset(5, seed=3), D.SyntheticCXRDataset(5, seed=3)
    assert len(a) == 5
    assert [r["conversations"][1]["value"] for r in a.records] == [r["conversations"][1]["value"] for r in b.records]
    r = a[0]
    assert r["image"][0].size == (64, 64) and "<image>" in D.record_prompt(r)
    assert len(D.LEXICON) == 14


def test_process_image_resize():
    from PIL import Image
    big = Image.new("RGB", (2000, 2000))
    out = D.process_image(big, 262144, 401408)
    assert out.width * out.height <= 401408
    small = Image.new("L", (64, 64))
    out = D.process_image(small, 262144, 401408)
    assert out.mode == "RGB" and out.width * out.height >= 262144 * 0.98


def test_left_pad():
    ex = [
        {"input_ids": torch.tensor([1, 2, 3]), "attention_mask": torch.ones(3, dtype=torch.long),
         "pixel_values": None, "image_grid_thw": torch.zeros(0, 3, dtype=torch.long), "ground_truth": "a"},
        {"input_ids": torch.tensor([7]), "attention_mask": torch.ones(1, dtype=torch.long),
         "pixel_values": None, "image_grid_thw": torch.zeros(0, 3, dtype=torch.long), "ground_truth": "b"},
    ]
    b = D.collate_left_pad(ex, pad_id=99)
    assert b["input_ids"].tolist() == [[1, 2, 3], [99, 99, 7]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]
    assert b["position_ids"].tolist() == [[0, 1, 2], [0, 0, 0]]
    assert b["ground_truth"] == ["a", "b"] and b["pixel_values"] is None
