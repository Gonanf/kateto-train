#!/usr/bin/env python3
"""R2: el flag `--rosa` no debe cambiar nada cuando esta apagado.

    venv/bin/python -m pytest tests/test_rosa_wiring.py -v
"""
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rwkv_pipeline.mini_rwkv import MiniRWKV
from rwkv_pipeline.opd_train import wire_rosa
from rwkv_pipeline.rosa_module import RosaAssociativeMemory, RosaHead
from rwkv_pipeline.smoke_rosa_cpu import run as smoke_run

VOCAB, N_EMBD, SEQ = 1024, 64, 16


def _student():
    torch.manual_seed(1337)
    return MiniRWKV(vocab=VOCAB, n_embd=N_EMBD, n_layer=2, ctx_len=SEQ)


def _ids():
    g = torch.Generator().manual_seed(1337)
    return torch.randint(0, VOCAB, (1, SEQ), generator=g)


def test_flag_off_returns_zero_and_keeps_plain_linear_head():
    s = _student()
    before = s.head
    assert wire_rosa(s, False) == 0
    assert s.head is before
    assert type(s.head) is torch.nn.Linear


def test_flag_off_is_bit_identical_to_unwired_forward():
    s = _student()
    x = _ids()
    reference = s.forward_normal(x).clone()
    wire_rosa(s, False)
    assert torch.equal(s.forward_normal(x), reference)


def test_flag_off_does_not_import_rosa_module():
    code = ("import sys;"
            "from rwkv_pipeline.opd_train import wire_rosa;"
            "from rwkv_pipeline.mini_rwkv import MiniRWKV;"
            "s=MiniRWKV(vocab=32,n_embd=8,n_layer=1,ctx_len=8);"
            "wire_rosa(s,False);"
            "print('imported', 'rwkv_pipeline.rosa_module' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=PROJECT,
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("imported False")


def test_flag_off_never_instantiates_rosa(monkeypatch):
    calls = []
    original = RosaAssociativeMemory.__init__

    def spy(self, *a, **kw):
        calls.append(1)
        original(self, *a, **kw)

    monkeypatch.setattr(RosaAssociativeMemory, "__init__", spy)
    wire_rosa(_student(), False)
    assert calls == []


def test_flag_on_installs_gated_parallel_branch():
    s = _student()
    added = wire_rosa(s, True)
    assert isinstance(s.head, RosaHead)
    expected = 3 * N_EMBD * 128 + 128 * VOCAB + N_EMBD + 1
    assert added == expected
    assert all(p.requires_grad for p in s.head.rosa.parameters())


def test_flag_on_logits_are_base_plus_gated_rosa():
    s = _student()
    hidden = torch.randn(1, SEQ, N_EMBD)
    base = s.head(hidden).clone()
    wire_rosa(s, True)
    rosa_logits, gate = s.head.rosa.forward_train(hidden)
    assert gate.shape == (1, SEQ, 1)
    assert float(gate.detach().min()) >= 0.0 and float(gate.detach().max()) <= 1.0
    assert torch.allclose(s.head(hidden), base + gate * rosa_logits, atol=1e-6)


def test_flag_on_gradients_reach_rosa_params():
    s = _student()
    wire_rosa(s, True)
    ids = _ids()
    logits = s.forward_normal(ids)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                           ids[:, 1:].reshape(-1))
    loss.backward()
    grads = [p.grad for p in s.head.rosa.parameters()]
    assert all(g is not None and float(g.abs().sum()) > 0 for g in grads)


def test_smoke_rosa_on_two_steps_finite_with_live_grads():
    r = smoke_run(steps=2, rosa=True)
    assert r["head_type"] == "RosaHead"
    assert r["rosa_params"] > 0
    assert len(r["losses"]) == 2
    assert all(torch.isfinite(torch.tensor(v)) for v in r["losses"])
    assert all(g > 0.0 for g in r["rosa_grad"])
    assert 0.0 < r["gate_mean"] < 1.0


@pytest.mark.parametrize("step,loss", [(1, 7.04908037), (2, 7.03626013)])
def test_smoke_rosa_off_matches_pre_change_baseline(step, loss):
    r = smoke_run(steps=2, rosa=False)
    assert r["losses"][step - 1] == pytest.approx(loss, abs=5e-9)
    assert r["head_type"] == "Linear"
    assert r["rosa_params"] == 0


def test_attach_rosa_is_idempotent():
    s = _student()
    wire_rosa(s, True)
    first = s.head
    assert wire_rosa(s, True) == sum(p.numel() for p in first.rosa.parameters())
    assert s.head is first