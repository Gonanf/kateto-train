#!/usr/bin/env python3
"""Pure-PyTorch mini RWKV-7-style student for OPD mechanism proof.

The real RWKV-PEFT `RWKV7` autograd path requires a custom CUDA operator whose
build hangs on this ROCm box (out/opd/_opcompile.log), and full 1.5B fine-tune
does not fit in 4 GB VRAM (§1.3). To exercise the *entire* OPD pipeline
(rollout -> byte-string mapping -> reverse KL -> coverage -> optimizer ->
baseline) with REAL numbers and autograd, this module provides a small,
self-contained, differentiable RWKV-flavoured student that uses the same real
RWKV tokenizer (vocab 65536) and the same `forward_normal(input_ids) -> logits
(B,T,vocab)` contract the pipeline expects.

This is an explicit REDUCED-SCALE mechanism engine. It is NOT a stand-in claim
for the 1.5B; the report marks it as such. Same byte-string alignment ↔ same
coverage story (coverage is tokenizer-alignment-bound, not size-bound).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Shift(nn.Module):
    """RWKV-style time-shift: concat(0, x[:-1])."""

    def forward(self, x):
        pad = torch.zeros_like(x[:, :1])
        return torch.cat([pad, x[:, :-1]], dim=1)


class MiniBlock(nn.Module):
    """RWKV-flavoured block: time-mixing (att) + channel-mixing (ffn), pure torch."""

    def __init__(self, C: int = 96):
        super().__init__()
        self.ln1 = nn.LayerNorm(C)
        self.ln2 = nn.LayerNorm(C)
        self.shift = _Shift()
        # time-mixing gating (receptance r, key k, value v)
        with torch.no_grad():
            self.xk = nn.Parameter(torch.ones(1, 1, C) * 0.5)
            self.xv = nn.Parameter(torch.ones(1, 1, C) * 0.5)
            self.xr = nn.Parameter(torch.ones(1, 1, C) * 0.5)
            self.xf = nn.Parameter(torch.ones(1, 1, C) * 0.5)
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        # channel-mixing
        self.key2 = nn.Linear(C, C * 4, bias=False)
        self.value2 = nn.Linear(C * 4, C, bias=False)

    def forward(self, h):
        xx = self.shift(h) - h
        xk = h + xx * self.xk
        xv = h + xx * self.xv
        xr = h + xx * self.xr
        k = self.key(self.ln1(xk))
        v = self.value(self.ln1(xv))
        r = torch.sigmoid(self.receptance(self.ln1(xr)))
        att = r * (k * v)                     # elementwise "decay-1" attn
        h = h + self.output(att)
        # channel mixing
        xf = h + (self.shift(h) - h) * self.xf
        f = torch.relu(self.key2(self.ln2(xf))) ** 2
        h = h + self.value2(f)
        return h


class MiniRWKV(nn.Module):
    """Tiny differentiable RWKV-style student over the REAL RWKV vocab."""

    def __init__(self, vocab: int = 65536, n_embd: int = 96, n_layer: int = 2,
                 ctx_len: int = 512):
        super().__init__()
        self.vocab, self.n_embd, self.n_layer, self.ctx_len = vocab, n_embd, n_layer, ctx_len
        self.emb = nn.Embedding(vocab, n_embd, padding_idx=0)
        self.blocks = nn.ModuleList([MiniBlock(n_embd) for _ in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)

    def forward_normal(self, input_ids: torch.Tensor,
                       inputs_embeds=None, attention_mask=None, **kwargs):
        if x is None and inputs_embeds is not None:  # pragma: no cover
            x = inputs_embeds
        else:
            x = self.emb(input_ids)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_out(x))