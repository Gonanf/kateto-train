#!/usr/bin/env python3
"""R3: `step_infer` incremental (O(1) el append) y equivalente a `forward_train`.

    venv/bin/python -m pytest tests/test_rosa_equivalence.py -v -s
"""
import sys
import time
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rwkv_pipeline.rosa_module import _KV_INIT_CAP, RosaAssociativeMemory

VOCAB, N_EMBD, RETR = 65536, 1024, 128  # los defaults reales del ckpt 0.4B
SEQ, REPS = 512, 80


def _rosa():
    torch.manual_seed(1337)
    return RosaAssociativeMemory(vocab_size=VOCAB, hidden_dim=N_EMBD,
                                 retrieval_dim=RETR).eval()


def _hidden(t):
    g = torch.Generator().manual_seed(1337)
    return torch.randn(1, t, N_EMBD, generator=g)


def _incremental(rosa, hidden):
    """Corre `step_infer` token a token sobre `hidden` y junta la salida."""
    memory, logits, gates = None, [], []
    for t in range(hidden.shape[1]):
        rosa_logits, gate, memory = rosa.step_infer(hidden[:, t:t + 1], memory)
        logits.append(rosa_logits)
        gates.append(gate)
    return torch.cat(logits, 1), torch.cat(gates, 1), memory


@pytest.mark.parametrize("seq", [1, 2, 7, 128, _KV_INIT_CAP + 88])
def test_step_infer_matches_forward_train(seq):
    """Mismo estado, mismos pesos: la fila t de `step_infer` es la fila t de `forward_train`."""
    rosa, hidden = _rosa(), _hidden(seq)
    ref_logits, ref_gate = rosa.forward_train(hidden)
    inc_logits, inc_gate, memory = _incremental(rosa, hidden)
    err_logits = (ref_logits - inc_logits).abs().max().item()
    err_gate = (ref_gate - inc_gate).abs().max().item()
    print(f"\n  T={seq:4d}  max|dlogits|={err_logits:.3e}  max|dgate|={err_gate:.3e}")
    assert memory[2] == seq, "el buffer no consumio los T tokens"
    assert err_logits <= 1e-4, err_logits
    assert err_gate <= 1e-4, err_gate


def test_append_does_not_reallocate_buffer():
    """Determinista: el append es in-place, asi que el objeto buffer no cambia.

    El tiempo solo no alcanza para probar esto -- a dims reales `out_head` domina
    el scan y la version con `torch.cat` tambien mide plano.
    """
    rosa, hidden = _rosa(), _hidden(_KV_INIT_CAP + 88)
    memory, buffers, grew_at = None, [], None
    for t in range(hidden.shape[1]):
        previous = memory[0] if memory else None
        _, _, memory = rosa.step_infer(hidden[:, t:t + 1], memory)
        buffers.append(memory[0])
        if previous is not None and memory[0] is not previous:
            grew_at = grew_at if grew_at is not None else t
    distinct = 1 + sum(1 for a, b in zip(buffers, buffers[1:]) if a is not b)
    print(f"\n  buffers distintos en {distinct} paso(s) de {len(buffers)}; "
          f"realloc en t={grew_at}")
    assert grew_at == _KV_INIT_CAP, grew_at
    assert distinct == 2, f"se reasigno {distinct} veces, deberia ser 2"


def _warm_cache(rosa, hidden, t):
    memory = None
    for i in range(t):
        _, _, memory = rosa.step_infer(hidden[:, i:i + 1], memory)
    return memory


def _step_time(rosa, hidden, t, memory):
    """Cronometra SOLO el paso t con el cache ya lleno; el clone va fuera del reloj."""
    warm = None if memory is None else (memory[0].clone(), memory[1].clone(), memory[2])
    start = time.perf_counter()
    rosa.step_infer(hidden[:, t - 1:t], warm)
    return time.perf_counter() - start


def test_per_step_cost_does_not_grow_with_T():
    """Gate del brief: costo por paso con T=1 y con T=512, medido A/B intercalado.

    El bound es flojo a proposito: en un host compartido el reloj sube 10-35% solo
    con la carga (medido: 1.108 / 1.298 / 1.353 en tres corridas seguidas). El gate
    fuerte de este item es `test_append_does_not_reallocate_buffer`, que es
    determinista; aca se reporta la medicion y se corta solo si se dispara de verdad.
    """
    rosa, hidden = _rosa(), _hidden(SEQ)
    caches = {t: _warm_cache(rosa, hidden, t) for t in (1, SEQ)}
    samples = {1: [], SEQ: []}
    for rep in range(REPS):
        for t in (1, SEQ) if rep % 2 == 0 else (SEQ, 1):
            samples[t].append(_step_time(rosa, hidden, t, caches[t]))
    one, two = sorted(samples[1]), sorted(samples[SEQ])
    median_one, median_two = one[len(one) // 2], two[len(two) // 2]
    ratio = median_two / median_one
    print(f"\n  T=1   {median_one * 1e6:8.1f} us/paso  (min {one[0] * 1e6:.1f})")
    print(f"  T={SEQ} {median_two * 1e6:8.1f} us/paso  (min {two[0] * 1e6:.1f})")
    print(f"  ratio T={SEQ}/T=1 = {ratio:.3f}")

    head = []
    rosa_q = rosa.proj_q(hidden[:, :1])
    for _ in range(200):
        start = time.perf_counter()
        rosa.out_head(rosa_q)
        head.append(time.perf_counter() - start)
    head.sort()
    print(f"  de los cuales out_head solo (fijo en T): {head[100] * 1e6:.1f} us "
          f"({head[100] / median_one:.0%} del paso)")
    assert ratio <= 1.6, f"el costo por paso CRECIO con T: ratio {ratio:.3f}"