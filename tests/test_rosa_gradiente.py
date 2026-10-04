#!/usr/bin/env python3
"""Canario del gradiente de ROSA: la rama debe estar en el camino Y no muerta.

    env -u PYTHONPATH -u VIRTUAL_ENV ./venv/bin/python -m pytest tests/test_rosa_gradiente.py -q

Dos Guards distintos, porque son dos fallos distintos:

1) "ROSA no esta en el camino" -> los logits del head con ROSA deben diferir de
   los del Linear pelado. Es el canario pedido: si son iguales, la rama no corre.

2) "ROSA esta en el camino pero muerta" -> el fallo real del 2026-10-01. La rama
   mezclaba bien pero el gate multiplicativo se cerraba (gate.mean 0.53 -> 6.6e-4
   en 4 pasos) y con el se llevaba el gradiente de proj_q/k/v y out_head:
   `rosa_grad_norm == 0.0` en 177 de los 300 pasos de out/rosa_head_corto. Un
   canario que solo mirara (1) no lo ve: los logits SI cambiaban al principio.
   Los dos tests de abajo fijan las invariantes que hacen que eso no vuelva:
   con out_head=0 la rama es no-op exacto en el forward pero out_head igual
   recibe gradiente (bootstrap), y con gate sin gradiente no hay multiplicador
   que se pueda cerrar.

Lo que este archivo NO puede guardar es la dinamica de 300 pasos: el colapso del
gate necesita el RWKV7 real en bf16 (n_embd=1024 hace que el gate de init.span
2.7e-5..1.0 entre tokens) y no se reproduce con MiniRWKV ni con tensores
sinteticos en CPU. Ese guardia es `rosa_grad_norm` por paso en la run.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rwkv_pipeline.rosa_module import RosaAssociativeMemory, RosaHead

VOCAB, N_EMBD, SEQ, RETRIEVAL = 1024, 64, 16, 128


def _head() -> RosaHead:
    torch.manual_seed(1337)
    return RosaHead(torch.nn.Linear(N_EMBD, VOCAB, bias=False),
                    retrieval_dim=RETRIEVAL)


def _hidden() -> torch.Tensor:
    g = torch.Generator().manual_seed(1337)
    return torch.randn(1, SEQ, N_EMBD, generator=g)


def _grad_norm(module) -> float:
    return sum(float(p.grad.detach().float().norm()) ** 2
               for p in module.parameters() if p.grad is not None) ** 0.5


def test_forward_with_rosa_differs_from_without():
    """Canario pedido: si los logits son iguales, ROSA no esta en el camino."""
    h = _head()
    x = _hidden()
    base = h.head(x)
    with_rosa = h(x)
    assert with_rosa.shape == base.shape
    assert not torch.equal(with_rosa, base)
    assert _grad_norm(h.rosa) == 0.0  # solo forward todavia


def test_every_rosa_param_is_in_the_graph():
    """Refuta explicitamente "el head nunca entra en el grafo": ningun .grad None."""
    h = _head()
    x = _hidden()
    logits = h(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                           torch.randint(0, VOCAB, (1, SEQ - 1)).reshape(-1))
    loss.backward()
    missing = [n for n, p in h.rosa.named_parameters() if p.grad is None]
    assert missing == []
    assert _grad_norm(h.rosa) > 0.0


def test_zero_init_out_head_is_exact_noop_but_still_learns():
    """out_head=0: silencio exacto en el forward, gradiente vivo igual.

    Es la invariante del zero-init de residual (ReZero / LoRA B=0) que usa
    rosa_train_corto: la rama arranca como no-op en vez de sumar ruido, pero no
    queda permanentemente muda. Si alguien la congela o la desconecta, esto falla.
    """
    h = _head()
    with torch.no_grad():
        h.rosa.out_head.weight.zero_()
    x = _hidden()
    assert torch.equal(h(x), h.head(x))

    logits = h(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                           torch.randint(0, VOCAB, (1, SEQ - 1)).reshape(-1))
    loss.backward()
    g = h.rosa.out_head.weight.grad
    assert g is not None and float(g.abs().sum()) > 0.0
    # el gate todavia no suma: su gradiente nace en 0 y recién revivira cuando
    # out_head deje de ser 0. Documenta el bootstrap, no un bug.
    assert h.rosa.gate.weight.grad is not None


def test_frozen_gate_cannot_close_itself():
    """El gate es el multiplicador que murio: sin gradiente no puede cerrarse."""
    h = _head()
    h.rosa.gate.weight.requires_grad_(False)
    h.rosa.gate.bias.requires_grad_(False)
    x = _hidden()
    logits = h(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                           torch.randint(0, VOCAB, (1, SEQ - 1)).reshape(-1))
    loss.backward()
    assert h.rosa.gate.weight.grad is None
    assert h.rosa.gate.bias.grad is None
    # y el resto de la rama sigue recibiendo gradiente sin el gate
    assert _grad_norm(h.rosa) > 0.0


def test_gate_stays_open_under_optimizer_steps():
    """12 pasos de AdamW como el run: con gate entrenable el multiplicador cae."""
    h = _head()
    params = list(h.rosa.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3, weight_decay=0.0)
    x = _hidden()
    g = torch.Generator().manual_seed(11)
    for _ in range(12):
        tgt = torch.randint(0, VOCAB, (1, SEQ - 1), generator=g).reshape(-1)
        logits = h(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    gate = torch.sigmoid(h.rosa.gate(x).detach())
    assert 0.2 < float(gate.mean()) < 0.8
    assert float(gate.min()) >= 0.0 and float(gate.max()) <= 1.0


def test_state_dict_keys_match_inference_module():
    """Lo que entrena el script tiene que ser lo que infer_kateto carga."""
    trained = dict(_head().rosa.state_dict())
    fresh = RosaAssociativeMemory(vocab_size=VOCAB, hidden_dim=N_EMBD,
                                  retrieval_dim=RETRIEVAL).state_dict()
    assert set(trained) == set(fresh)
    assert fresh["out_head.weight"].shape == (VOCAB, RETRIEVAL)
