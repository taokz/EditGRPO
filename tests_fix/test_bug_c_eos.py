"""Bug C: edited responses have no EOS, so the response mask is all ones after padding.

Pure-torch reproduction. `get_eos_mask` is copied from
verl/utils/torch_functional.py:113-117 (importing verl pulls heavy deps).
The padding/append logic mirrors vllm_rollout_spmd.py (extractor_append_eos).
Run: python tests_fix/test_bug_c_eos.py
"""
import torch

EOS, PAD, L = 2, 0, 8

def get_eos_mask(response_ids, eos_token_id, dtype=torch.int64):  # copy of torch_functional.get_eos_mask
    eos_mask = response_ids.eq(eos_token_id).long()
    eos_mask = (torch.cumsum(eos_mask, dim=1) - eos_mask).bool()
    eos_mask = torch.logical_not(eos_mask).to(dtype)
    return eos_mask

def pad(tokens, append_eos):
    if append_eos:
        tokens = tokens[:L - 1] + [EOS]
    out = torch.full((L,), PAD)
    out[:len(tokens)] = torch.tensor(tokens)
    return out

edited = [5, 6, 7]
m_orig = get_eos_mask(pad(edited, False)[None], EOS)[0]
m_fix = get_eos_mask(pad(edited, True)[None], EOS)[0]
assert m_orig.tolist() == [1] * L, m_orig          # original: padding is unmasked
assert m_fix.tolist() == [1, 1, 1, 1, 0, 0, 0, 0], m_fix  # fixed: EOS kept, padding masked
# long edit is truncated to L-1 then EOS is still last
assert pad(list(range(3, 20)), True)[-1].item() == EOS
print("OK", m_orig.tolist(), m_fix.tolist())
