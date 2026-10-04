#!/usr/bin/env python3
"""R7: `step_infer` es causal Y attendende al token actual — igual que `forward_train`.

    venv/bin/python -m pytest tests/test_rosa_causal.py -v -s

Que este archivo existe y por que:

El diagnostico R1 reporto que `step_infer` "no aplica mascara causal ni excluye el
token actual de la memoria, mientras que `forward_train` si (rosa_module.py:43-44)".
Eso fue una lectura del codigo, no una ejecucion, y es incorrecto en las dos partes:

  - La causalidad no depende de una mascara explicita en `step_infer`: la fila t se
    softmaxea contra los tokens ya escritos, o sea unicamente contra 0..t. El futuro
    no existe todavia en el buffer. La equivalencia fila a fila con `forward_train`
    esta probada en `tests/test_rosa_equivalence.py`.
  - El token actual SI entra, en los dos caminos. La mascara de `forward_train` es
    `torch.tril(..., diagonal=0)`; `diagonal=0` es la mascara que CONSERVA la
    diagonal, no la que la saca (esa seria `diagonal=-1`). Excluir el token propio
    seria cambiar el modelo, no arreglar un bug.

Asi que aca no hay bug que arreglar y por lo tanto no hay arreglo: este archivo fija
el comportamiento correcto para que el diagnostico no se vuelva a reabrir y para
que un "fix" futuro que saque la diagonal reviente en rojo en vez de en silencio.

Como se observa la atencion sin recalcularla
---------------------------------------------
Recalcular la matriz de atencion afuera del modulo no prueba nada del modulo: si el
bug esta en el append o en el corte del buffer, la recomputacion sale bien igual
(medido: con el bug inyectado, 4 de 5 tests de este archivo seguian en verde).

Para mirar la distribucion de atencion REAL se sustituyen solo las dos piezas que no
son la atencion:
  - `proj_v` pasa a devolver un one-hot por posicion de llamada, asi cada token aporta
    su propia base canonica y no se mezclan;
  - `out_head` pasa a ser la identidad.
Con eso `retrieved = attn @ one_hots` es la fila de atencion misma, y los
`rosa_logits` que devuelve `step_infer` SON esa fila. No se recalcula nada: lo que
mide el test es el valor que el modulo devuelve.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rwkv_pipeline.rosa_module import RosaAssociativeMemory

VOCAB, N_EMBD, RETR = 65536, 1024, 128  # los defaults reales del ckpt 0.4B
SEQ = 6


class _OneHotPerCall(nn.Module):
    """`proj_v` que devuelve la base canonica e_t, una por llamada (una por token)."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def forward(self, x):
        out = torch.zeros(x.shape[0], 1, RETR, dtype=x.dtype, device=x.device)
        out[:, 0, self.n] = 1.0
        self.n += 1
        return out


def _rosa():
    torch.manual_seed(1337)
    return RosaAssociativeMemory(vocab_size=VOCAB, hidden_dim=N_EMBD,
                                 retrieval_dim=RETR).eval()


def _hidden(t):
    g = torch.Generator().manual_seed(1337)
    return torch.randn(1, t, N_EMBD, generator=g)


def _with_readout(rosa):
    """Deja `rosa_logits` de `step_infer` igual a la fila de atencion, sin tocar la atencion."""
    rosa.proj_v = _OneHotPerCall()
    rosa.out_head = nn.Linear(RETR, RETR, bias=False)
    with torch.no_grad():
        rosa.out_head.weight.copy_(torch.eye(RETR))
    return rosa


def _observed_rows(rosa, hidden):
    """Filas de atencion tal como las devuelve `step_infer`, una por token."""
    memory, rows = None, []
    for t in range(hidden.shape[1]):
        rosa_logits, _, memory = rosa.step_infer(hidden[:, t:t + 1], memory)
        rows.append(rosa_logits[0, 0])
    return rows, memory


def _train_attn(rosa, hidden):
    """Matriz de atencion de `forward_train` y la mascara que usa."""
    T = hidden.shape[1]
    q, k = rosa.proj_q(hidden), rosa.proj_k(hidden)
    scale = 1.0 / (RETR ** 0.5)
    scores = torch.einsum("bid,bjd->bij", q, k) * scale
    mask = torch.tril(torch.ones(T, T, device=hidden.device), diagonal=0)
    return F.softmax(scores.masked_fill(mask == 0, -1e9), dim=-1)[0], mask


def test_forward_train_mask_keeps_the_diagonal():
    """El punto del diagnostico R1: `diagonal=0` conserva el token propio.

    Si alguien "corrige" R1 sacando el token actual de la diagonal, esto falla antes
    de que el cambio llegue a ningun lado.
    """
    _, mask = _train_attn(_rosa(), _hidden(SEQ))
    assert mask[2, 2] == 1, "la diagonal tiene que estar en la mascara"
    assert mask[2].tolist() == [1, 1, 1, 0, 0, 0], mask[2].tolist()
    assert mask[2, 3] == 0, "el futuro tiene que estar fuera"


def test_step_infer_weights_the_current_token():
    """El claim de R1, medido sobre el valor que `step_infer` devuelve: pesa si."""
    rosa = _with_readout(_rosa())
    train_rows, _ = _train_attn(rosa, _hidden(SEQ))
    rows, _ = _observed_rows(rosa, _hidden(SEQ))
    for t, row in enumerate(rows):
        self_infer = row[t].item()
        self_train = train_rows[t, t].item()
        print(f"  t={t}  self train={self_train:.6f}  infer={self_infer:.6f}")
        assert self_infer > 0.0, f"t={t}: sin peso propio, y train si lo tiene"
        assert self_infer == pytest.approx(self_train, abs=1e-5), (t, self_infer, self_train)


def test_step_infer_row_covers_exactly_the_past_and_present():
    """La fila t suma 1 sobre 0..t y es exactamente 0 en t+1 en adelante.

    El readout son 128 dims (el retrieval dim) y la fila viva ocupa las `t + 1`
    primeras; el resto tiene que ser 0 duro. Si el buffer se cortara en la capacidad
    preasignada en vez de en `:pos`, la softmax se repartiria sobre el `new_empty` sin
    inicializar y esos dims dejarian de ser 0.
    """
    rows, memory = _observed_rows(_with_readout(_rosa()), _hidden(SEQ))
    assert memory[2] == SEQ, "el buffer no consumio los T tokens"
    for t, row in enumerate(rows):
        assert row.numel() == RETR, f"t={t}: readout de {row.numel()}, deberia ser {RETR}"
        assert (row >= 0).all(), t
        assert row[:t + 1].sum().item() == pytest.approx(1.0, abs=1e-5), t
        leaked = row[t + 1:].abs().max().item()
        assert leaked == 0.0, f"t={t}: {leaked:.3e} de peso en posiciones futuras"
        print(f"  t={t}  suma sobre 0..t = {row[:t + 1].sum().item():.6f}  "
              f"cola = {leaked:.3e}")


def test_step_infer_row_equals_forward_train_row():
    """La matriz de atencion entera, fila a fila, es la misma en los dos caminos."""
    hidden = _hidden(SEQ)
    rosa = _with_readout(_rosa())
    train_rows, _ = _train_attn(rosa, hidden)
    rows, _ = _observed_rows(rosa, hidden)
    worst = 0.0
    for t, row in enumerate(rows):
        full = F.pad(row[:t + 1], (0, hidden.shape[1] - (t + 1)))  # el futuro es 0 en train
        d = (train_rows[t] - full).abs().max().item()
        worst = max(worst, d)
        print(f"  t={t}  max|d fila| = {d:.3e}")
    assert worst <= 1e-5, worst


def test_step_infer_logits_match_forward_train():
    """El modulo intacto, sin ningun sustituto: el error sobre los logits reales."""
    rosa, hidden = _rosa(), _hidden(SEQ)
    ref_logits, ref_gate = rosa.forward_train(hidden)
    memory, inc_logits, inc_gates = None, [], []
    for t in range(hidden.shape[1]):
        rosa_logits, gate, memory = rosa.step_infer(hidden[:, t:t + 1], memory)
        inc_logits.append(rosa_logits)
        inc_gates.append(gate)
    inc_logits, inc_gates = torch.cat(inc_logits, 1), torch.cat(inc_gates, 1)
    assert memory[2] == hidden.shape[1], "el buffer no consumio los T tokens"
    err_logits = (ref_logits - inc_logits).abs().max().item()
    err_gate = (ref_gate - inc_gates).abs().max().item()
    print(f"\n  max|dlogits| = {err_logits:.3e}  max|dgate| = {err_gate:.3e}")
    assert err_logits <= 1e-4, err_logits
    assert err_gate <= 1e-4, err_gate