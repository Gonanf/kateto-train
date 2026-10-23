#!/usr/bin/env python3
"""Standalone test for special token masking and renormalized top-K reverse KL."""
from pathlib import Path
import sys
from typing import List, Tuple

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

import torch
from rwkv_pipeline.opd_train import (
    MARKERS,
    marker_spans,
    masked_positions,
    reverse_kl_step,
)


class StubTokenizer:
    """Tokenizador stub: clase con decodeBytes(ids: List[int]) -> bytes que concatena una tabla fija TOK: List[bytes]."""

    def __init__(self, tok_table: List[bytes]):
        self.tok_table = tok_table

    def decodeBytes(self, ids: List[int]) -> bytes:
        return b"".join(self.tok_table[i] for i in ids)


def main() -> None:
    # -----------------------------------------------------------------------
    # Part 1 & 2 & 3: Stub tokenizer and synthetic response with all 5 markers
    # -----------------------------------------------------------------------
    TOK: List[bytes] = [
        b"Hello ",            # 0: normal
        b"<|im_",             # 1: part of <|im_start|>
        b"start",             # 2: part of <|im_start|>
        b"|>",                # 3: part of <|im_start|>
        b" user text ",       # 4: normal
        b"<|im_user",         # 5: part of <|im_user|>
        b"|>",                # 6: part of <|im_user|>
        b" middle ",          # 7: normal
        b"<|im_dev|>",        # 8: whole marker in 1 token
        b" dev notes ",       # 9: normal
        b"<|",                # 10: part of <|im_end|>
        b"im_",               # 11: part of <|im_end|>
        b"end|>",             # 12: part of <|im_end|>
        b" closing ",         # 13: normal
        b"<|end",             # 14: part of <|endoftext|>
        b"oftext|>",          # 15: part of <|endoftext|>
        b" final byte",       # 16: normal
    ]
    tok = StubTokenizer(TOK)
    response_ids = list(range(len(TOK)))

    token_bytes = [tok.decodeBytes([i]) for i in response_ids]
    resp_bytes = b"".join(token_bytes)

    for m in MARKERS:
        assert m in resp_bytes, f"Marker {m} missing from synthetic response"

    # Expected calculation with independent arithmetic
    test_spans: List[Tuple[int, int]] = []
    for m in MARKERS:
        start = 0
        while True:
            pos = resp_bytes.find(m, start)
            if pos == -1:
                break
            test_spans.append((pos, pos + len(m)))
            start = pos + 1

    expected = set()
    offset = 0
    for idx, tb in enumerate(token_bytes):
        t_start = offset
        t_end = offset + len(tb)
        offset = t_end
        for sp_start, sp_end in test_spans:
            if max(t_start, sp_start) < min(t_end, sp_end):
                expected.add(idx)

    spans = marker_spans(resp_bytes)
    masked = set(masked_positions(token_bytes, spans))

    assert masked == expected, f"masked_positions mismatch: {masked} != {expected}"
    assert len(expected) > 0, "expected must contain at least one masked position"

    # -----------------------------------------------------------------------
    # Part 4: Synthetic reverse_kl_step
    # -----------------------------------------------------------------------
    torch.manual_seed(42)
    vocab = 64
    row = torch.randn(vocab, requires_grad=True)
    top_k = 10
    v, i = torch.topk(row, top_k)
    student_idx = {f"s{t}".encode(): t for t in i.tolist()}

    # Teacher logp: real truncated logprobs with sum(exp) < 1
    # Sum of exp(-2.5 - 0.1 * idx) over 10 items is ~0.545 < 1.0
    teacher_logp = {k: -2.5 - 0.1 * idx for idx, k in enumerate(student_idx.keys())}

    loss, met = reverse_kl_step(row, student_idx, teacher_logp, top_k, tok)
    loss.backward()

    # Asserts required by brief:
    assert 0.0 <= met["coverage"] <= 1.0, f"coverage out of bounds: {met['coverage']}"
    assert torch.isfinite(loss).item() and float(loss.item()) >= 0.0, f"loss invalid: {loss.item()}"
    assert row.grad is not None and float(row.grad.abs().sum().item()) > 0.0, "grad zero or None"
    assert 0.0 < met["teacher_mass_kept"] < 1.0, f"teacher_mass_kept invalid: {met['teacher_mass_kept']}"
    assert abs(met.get("p_t_sum", 0.0) - 1.0) < 1e-6, f"p_t does not sum to 1: {met.get('p_t_sum')}"

    # -----------------------------------------------------------------------
    # Part 5: Print output line
    # -----------------------------------------------------------------------
    print(f"OK markers=5 masked={len(masked)} expected={len(expected)} "
          f"loss={met['loss_rkl']:.5f} coverage={met['coverage']:.4f}")


if __name__ == "__main__":
    main()
