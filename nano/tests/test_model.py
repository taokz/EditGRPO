"""CPU tests for nano/editgrpo/model.py with the tiny random config (no network, no transformers)."""

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from editgrpo.model import (
    Qwen25VL,
    Qwen25VLConfig,
    image_preprocess,
    load_pretrained,
    smart_resize,
)

CFG = Qwen25VLConfig.tiny()
IMG, VS, VE = CFG.image_token_id, CFG.vision_start_token_id, CFG.vision_end_token_id
PAD = 0
PATCH_DIM = 3 * 2 * 14 * 14


def make_model(seed: int = 0) -> Qwen25VL:
    torch.manual_seed(seed)
    return Qwen25VL(CFG).eval()


def make_batch(seed: int = 0):
    """Row 0: text + one [1,4,4] image (4 merged tokens); row 1: shorter text, left-padded."""
    g = torch.Generator().manual_seed(seed)
    row0 = [11, 12, 13, VS] + [IMG] * 4 + [VE, 14, 15, 16]
    row1 = [21, 22, 23, 24, 25, 26]
    T = len(row0)
    input_ids = torch.tensor([row0, [PAD] * (T - len(row1)) + row1])
    attention_mask = (input_ids != PAD).long()
    grid = torch.tensor([[1, 4, 4]])
    pixel_values = torch.randn(16, PATCH_DIM, generator=g)
    return input_ids, attention_mask, pixel_values, grid


def test_forward_shapes_and_finite():
    model = make_model()
    input_ids, attention_mask, pixel_values, grid = make_batch()
    logits, cache = model(input_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=grid, use_cache=True)
    assert logits.shape == (2, input_ids.shape[1], CFG.vocab_size)
    assert torch.isfinite(logits).all()
    assert len(cache) == CFG.num_hidden_layers
    k, v = cache[0]
    assert k.shape == (2, CFG.num_key_value_heads, input_ids.shape[1], CFG.head_dim)
    # image features actually change the logits of the image row
    logits_noimg, _ = model(input_ids, attention_mask)
    assert not torch.allclose(logits[0], logits_noimg[0])
    assert torch.allclose(logits[1], logits_noimg[1], atol=1e-5)


def test_left_padding_matches_unpadded():
    """The left-padded row must give the same logits as the same sequence without padding."""
    model = make_model()
    input_ids, attention_mask, pixel_values, grid = make_batch()
    logits, _ = model(input_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=grid)
    n_valid = int(attention_mask[1].sum())
    solo = input_ids[1:, -n_valid:]
    logits_solo, _ = model(solo)
    assert torch.allclose(logits[1, -n_valid:], logits_solo[0], atol=1e-4)


def test_generate_with_cache_matches_no_cache_greedy():
    model = make_model()
    input_ids, attention_mask, pixel_values, grid = make_batch()
    steps = 6
    resp, finished = model.generate(
        input_ids, attention_mask, pixel_values, grid, max_new_tokens=steps,
        temperature=0.0, eos_token_id=-1, pad_token_id=PAD,
    )
    assert resp.shape == (2, steps) and not finished.any()

    # Reference: full forward (no cache) on the growing sequence, positions from get_rope_index.
    seq, mask = input_ids, attention_mask
    for step in range(steps):
        logits, _ = model(seq, mask, pixel_values=pixel_values, image_grid_thw=grid)
        nxt = logits[:, -1].argmax(-1)
        assert torch.equal(nxt, resp[:, step]), f"mismatch at step {step}"
        seq = torch.cat([seq, nxt[:, None]], 1)
        mask = torch.cat([mask, torch.ones(2, 1, dtype=mask.dtype)], 1)


def test_generate_eos_padding_and_sampling():
    model = make_model()
    input_ids, attention_mask, pixel_values, grid = make_batch()
    # Greedy: find the first token of row 1 and declare it EOS -> response is [eos, pad, pad, ...].
    resp, _ = model.generate(input_ids, attention_mask, pixel_values, grid, max_new_tokens=3,
                             temperature=0.0, eos_token_id=-1, pad_token_id=PAD)
    eos = int(resp[1, 0])
    resp2, finished = model.generate(input_ids, attention_mask, pixel_values, grid, max_new_tokens=4,
                                     temperature=0.0, eos_token_id=[eos], pad_token_id=PAD)
    assert bool(finished[1]) and resp2[1, 0].item() == eos and (resp2[1, 1:] == PAD).all()
    # Sampling is reproducible with a generator and respects top_k.
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    s1, _ = model.generate(input_ids, attention_mask, pixel_values, grid, max_new_tokens=4,
                           temperature=1.0, top_p=0.9, top_k=5, eos_token_id=-1, pad_token_id=PAD, generator=g1)
    s2, _ = model.generate(input_ids, attention_mask, pixel_values, grid, max_new_tokens=4,
                           temperature=1.0, top_p=0.9, top_k=5, eos_token_id=-1, pad_token_id=PAD, generator=g2)
    assert torch.equal(s1, s2)


def test_get_rope_index_hand_example():
    model = make_model()
    # [t, t, VS, I, I, I, I, VE, t] with llm grid 2x2
    row0 = [11, 12, VS, IMG, IMG, IMG, IMG, VE, 13]
    # same sequence, left-padded by two
    row1 = [PAD, PAD, 11, 12, VS, IMG, IMG, IMG, IMG, VE, 13]
    input_ids = torch.tensor([row0 + [PAD, PAD], row1])
    attention_mask = torch.tensor([[1] * 9 + [0, 0], [0, 0] + [1] * 9])
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])
    pos, deltas = model.get_rope_index(input_ids, grid, attention_mask)
    assert pos.shape == (3, 2, 11)
    expect = torch.tensor([
        [0, 1, 2, 3, 3, 3, 3, 5, 6],  # temporal
        [0, 1, 2, 3, 3, 4, 4, 5, 6],  # height
        [0, 1, 2, 3, 4, 3, 4, 5, 6],  # width
    ])
    assert torch.equal(pos[:, 0, :9], expect)
    assert (pos[:, 0, 9:] == 1).all()  # padding positions are 1 as in HF
    assert torch.equal(pos[:, 1, 2:], expect)
    assert (pos[:, 1, :2] == 1).all()
    assert torch.equal(deltas, torch.tensor([[7 - 11], [7 - 11]]))

    # Non-square image and text-only row: text continues from max(h, w) after the image.
    row = [VS] + [IMG] * 6 + [VE]
    pos, deltas = model.get_rope_index(torch.tensor([row, [1] * 8]), torch.tensor([[1, 4, 6]]))
    assert torch.equal(pos[:, 0], torch.tensor([
        [0, 1, 1, 1, 1, 1, 1, 4],
        [0, 1, 1, 1, 2, 2, 2, 4],
        [0, 1, 2, 3, 1, 2, 3, 4],
    ]))
    assert torch.equal(pos[:, 1], torch.arange(8).expand(3, 8))
    assert torch.equal(deltas, torch.tensor([[5 - 8], [0]]))


def test_gradient_flow_with_checkpointing():
    model = make_model()
    model.train()
    model.gradient_checkpointing = True
    input_ids, attention_mask, pixel_values, grid = make_batch()
    logits, _ = model(input_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=grid)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, CFG.vocab_size).float(), input_ids[:, 1:].reshape(-1)
    )
    loss.backward()
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for _, g in grads), [n for n, g in grads if g is None]
    assert model.visual.patch_embed.proj.weight.grad.abs().sum() > 0  # gradient reached the vision tower

    # freeze_vision stops gradients to the tower (merger included) but not to the LM
    model.zero_grad()
    model.freeze_vision()
    assert not any(p.requires_grad for p in model.visual.parameters())
    logits, _ = model(input_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=grid)
    logits.float().logsumexp(-1).mean().backward()
    assert model.visual.merger.mlp[0].weight.grad is None
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_image_preprocess_shapes():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, size=(70, 100, 3), dtype=np.uint8))
    pv, grid = image_preprocess(img, min_pixels=56 * 56, max_pixels=28 * 28 * 16)
    t, h, w = grid[0].tolist()
    assert t == 1 and grid.shape == (1, 3)
    assert pv.shape == (t * h * w, PATCH_DIM) and pv.dtype == torch.float32
    assert (h * 14) % 28 == 0 and (w * 14) % 28 == 0 and h * w * 14 * 14 <= 28 * 28 * 16
    assert h % 2 == 0 and w % 2 == 0  # mergeable
    # smart_resize keeps aspect ratio approximately and respects bounds
    hh, ww = smart_resize(70, 100, 28, 56 * 56, 28 * 28 * 16)
    assert (hh // 14, ww // 14) == (h, w)
    # a constant-colour image gives constant normalised values per channel
    flat = Image.new("RGB", (56, 84), (255, 128, 0))
    pv2, grid2 = image_preprocess(flat, min_pixels=56 * 56, max_pixels=28 * 28 * 16)
    assert grid2.tolist() == [[1, 6, 4]]
    per_channel = pv2.view(-1, 3, 2 * 14 * 14)
    expected = (torch.tensor([255.0, 128.0, 0.0]) / 255 - torch.tensor([0.48145466, 0.4578275, 0.40821073])) / torch.tensor(
        [0.26862954, 0.26130258, 0.27577711]
    )
    assert torch.allclose(per_channel, expected[None, :, None].expand_as(per_channel), atol=1e-5)
    # feeding the result through the model consumes exactly h*w/4 image tokens
    model = make_model()
    ids = torch.tensor([[VS] + [IMG] * (h * w // 4) + [VE]])
    logits, _ = model(ids, pixel_values=pv, image_grid_thw=grid)
    assert torch.isfinite(logits).all()


def test_vision_windows_are_a_permutation():
    """Larger grid -> several (partially padded) windows; output order must still follow the image."""
    model = make_model()
    grid = torch.tensor([[1, 12, 8], [1, 4, 4]])  # llm grids 6x4 and 2x2, window = 4 merged tokens
    window_index, cu = model.visual.get_window_index(grid)
    assert sorted(window_index.tolist()) == list(range(6 * 4 + 2 * 2))
    assert cu[0] == 0 and cu[-1] == (12 * 8 + 4 * 4) and all(b > a for a, b in zip(cu, cu[1:]))
    pv = torch.randn(12 * 8 + 16, PATCH_DIM)
    feats = model.visual(pv, grid)
    assert feats.shape == (24 + 4, CFG.hidden_size) and torch.isfinite(feats).all()
    # the second image's features do not depend on the first image being present
    feats_solo = model.visual(pv[96:], grid[1:])
    assert torch.allclose(feats[24:], feats_solo, atol=1e-5)


def test_config_roundtrip_and_load_pretrained(tmp_path):
    from safetensors.torch import save_file

    model = make_model(seed=3)
    with open(tmp_path / "config.json", "w") as f:
        json.dump(CFG.to_dict(), f)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    # two shards + index, as on the Hub
    keys = sorted(state)
    half = len(keys) // 2
    save_file({k: state[k] for k in keys[:half]}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({k: state[k] for k in keys[half:]}, str(tmp_path / "model-00002-of-00002.safetensors"))
    index = {"weight_map": {k: ("model-00001-of-00002.safetensors" if i < half else "model-00002-of-00002.safetensors")
                            for i, k in enumerate(keys)}}
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)

    loaded = load_pretrained(str(tmp_path))
    assert loaded.config == CFG
    for k, v in loaded.state_dict().items():
        assert torch.equal(v, state[k]), k
    input_ids, attention_mask, pixel_values, grid = make_batch()
    a, _ = model(input_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=grid)
    b, _ = loaded(input_ids, attention_mask, pixel_values=pixel_values, image_grid_thw=grid)
    assert torch.allclose(a, b)

    # strict loading: a missing key is reported
    os.remove(tmp_path / "model.safetensors.index.json")
    os.remove(tmp_path / "model-00002-of-00002.safetensors")
    with pytest.raises(RuntimeError, match="missing"):
        load_pretrained(str(tmp_path))


def test_config_from_hf_style_dict():
    hf = {
        "hidden_size": 2048, "intermediate_size": 11008, "num_hidden_layers": 36, "num_attention_heads": 16,
        "num_key_value_heads": 2, "vocab_size": 151936, "rms_norm_eps": 1e-06, "rope_theta": 1000000.0,
        "tie_word_embeddings": True, "max_position_embeddings": 128000, "torch_dtype": "bfloat16",
        "image_token_id": 151655, "video_token_id": 151656, "vision_start_token_id": 151652,
        "vision_end_token_id": 151653, "rope_scaling": {"type": "mrope", "mrope_section": [16, 24, 24]},
        "vision_config": {"depth": 32, "hidden_act": "silu", "hidden_size": 1280, "intermediate_size": 3420,
                          "num_heads": 16, "in_chans": 3, "out_hidden_size": 2048, "patch_size": 14,
                          "spatial_merge_size": 2, "spatial_patch_size": 14, "window_size": 112,
                          "fullatt_block_indexes": [7, 15, 23, 31], "tokens_per_second": 2, "temporal_patch_size": 2},
        "architectures": ["Qwen2_5_VLForConditionalGeneration"], "model_type": "qwen2_5_vl",
    }
    cfg = Qwen25VLConfig.from_dict(hf)
    assert cfg.head_dim == 128 and cfg.mrope_section == [16, 24, 24]
    assert cfg.vision_config.in_channels == 3 and cfg.vision_config.fullatt_block_indexes == [7, 15, 23, 31]
    assert Qwen25VLConfig.from_dict(cfg.to_dict()) == cfg
