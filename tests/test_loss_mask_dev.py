#!/usr/bin/env python3
"""El loss masker tiene que conocer el turno `developer` (<|im_dev|>).

    venv/bin/python -m pytest tests/test_loss_mask_dev.py -q

Bug de la card: `RWKV-PEFT/rwkvt/dataset/mask.py` enmascara solo desde
`<|im_user|>` (t1). Un turno `developer` abre con `<|im_dev|>`, que NO era t1;
al venir despues de un `<|im_start|>` previo (cur=True) el texto del slot
developer caia DENTRO del tramo conservado y recibia gradiente.

Bug de fondo que aparecio al armar el test: en el formato canonico que emite
`chat_template.render_turn` los turnos van PEGADOS (`<|im_end|><|im_user|>`), y
BPE fusiona el `<` del marcador con el `>` del turno anterior en un solo token.
Con eso, la secuencia exacta de `<|im_user|>` tampoco matcheaba y TODO turno
posterior al primero filtraba gradiente (34.450 ocurrencias en 18.815 muestras
del corpus). El fix: `create_mask`/`generate_mask` tambien matchean el marcador
sin su id inicial, y `dataset.py` le pasa `<|im_dev|>` como opener extra.

`create_mask` devuelve labels con el shift de dataset.py: el label de la posicion
`k` (k>=1) es `out[k-1]`; el token recibe gradiente iff `out[k-1] != -100`.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT), str(PROJECT / "RWKV-PEFT")):
    if p not in sys.path:
        sys.path.insert(0, p)

from rwkv.utils import PIPELINE  # noqa: E402
from rwkv_pipeline import chat_template  # noqa: E402

_MASK_PATH = PROJECT / "RWKV-PEFT" / "rwkvt" / "dataset" / "mask.py"
_spec = importlib.util.spec_from_file_location("kateto_mask", _MASK_PATH)
mask = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mask)

VOICE = "seco"
IGNORE = -100


@pytest.fixture(scope="module")
def pipe():
    return PIPELINE("rwkv6", "rwkv_vocab_v20230424")


def _markers(pipe):
    return (pipe.encode("<|im_user|>"), pipe.encode("<|im_start|>"),
            [pipe.encode("<|im_dev|>")])


def _find_opt(ids, sub, start=0):
    for i in range(start, len(ids) - len(sub) + 1):
        if ids[i:i + len(sub)] == sub:
            return i
    return None


def _find(ids, sub, start=0):
    i = _find_opt(ids, sub, start)
    assert i is not None, f"no encontre {sub} desde {start}"
    return i


def _find_marker(ids, full, start=0):
    """Ubica un marcador ChatML, tolerando que su `<` inicial este fusionado."""
    i = _find_opt(ids, full, start)
    if i is None:
        i = _find_opt(ids, full[1:], start)
    assert i is not None, f"no encontre el marcador {full} desde {start}"
    return i


def _grad(tok_idx, labels, ids):
    """True si el token en `tok_idx` recibe gradiente (label != -100)."""
    return tok_idx != 0 and int(labels[tok_idx - 1]) != IGNORE


def _traza_dev(pipe):
    return (chat_template.render_turn("che, corre esto", "dale, ya va", VOICE)
            + chat_template.render_turn("ahora corre los tests unitarios",
                                        "corriendo", VOICE, role="developer"))


def test_developer_slot_no_recibe_gradiente(pipe):
    """Formato canonico (turnos pegados): el slot developer sale -100, la respuesta se conserva."""
    trace = _traza_dev(pipe)
    assert "<|im_dev|>" in trace and "<|im_user|>" in trace

    ids = pipe.encode(trace)
    t1, t2, t1_alt = _markers(pipe)
    labels = mask.create_mask(torch.tensor(ids), t1, t2, len(ids), t1_alt)

    dev_open = _find_marker(ids, t1_alt[0])
    im_start_next = _find(ids, t2, dev_open)
    assert im_start_next > dev_open

    # Nada del slot developer (su texto, su <|im_end|>, el \n) recibe gradiente.
    for k in range(dev_open, im_start_next):
        assert not _grad(k, labels, ids), (
            f"posicion {k} del turno developer recibiria gradiente: {ids[k]!r}")

    # La respuesta del asistente que sigue SI se conserva (label == id real).
    ans = pipe.encode("corriendo")
    ans_start = _find(ids, ans, im_start_next)
    for k in range(ans_start, ans_start + len(ans)):
        assert int(labels[k - 1]) == ids[k], "la respuesta del asistente deberia conservarse"

    # Y la respuesta del turno user tambien sigue conservada.
    u = pipe.encode("dale, ya va")
    u_start = _find(ids, u)
    for k in range(u_start, u_start + len(u)):
        assert int(labels[k - 1]) == ids[k]


def test_segundo_turno_user_ya_no_filtra(pipe):
    """El bug de fondo: en turnos pegados, el 2do turno user filtraba gradiente."""
    trace = (chat_template.render_turn("che, corre esto", "dale, ya va", VOICE)
             + chat_template.render_turn("y los tests?", "verdes", VOICE))
    assert "<|im_end|><|im_user|>" in trace  # turnos pegados

    ids = pipe.encode(trace)
    t1, t2, t1_alt = _markers(pipe)
    labels = mask.create_mask(torch.tensor(ids), t1, t2, len(ids), t1_alt)

    u2 = pipe.encode("y los tests?")
    u2_start = _find_marker(ids, pipe.encode("<|im_user|>"), _find(ids, t2))
    for k in range(u2_start, _find(ids, t2, u2_start)):
        assert not _grad(k, labels, ids), (
            f"el 2do turno user filtra gradiente en {k}: {ids[k]!r}")
    a2 = pipe.encode("verdes")
    a2_start = _find(ids, a2, u2_start)
    for k in range(a2_start, a2_start + len(a2)):
        assert int(labels[k - 1]) == ids[k]


def test_formato_con_salto_sin_cambios(pipe):
    """Formato con `\\n` entre turnos: mismo mask que el algoritmo viejo."""
    trace = ("<|im_user|>che, corre esto<|im_end|>\n<|im_start|>seco\ndale, ya va"
             "<|im_end|>\n<|im_user|>y los tests?<|im_end|>\n<|im_start|>seco\nverdes<|im_end|>")
    ids = pipe.encode(trace)
    t1, t2, t1_alt = _markers(pipe)
    nuevo = mask.create_mask(torch.tensor(ids), t1, t2, len(ids), t1_alt)
    viejo = _create_mask_viejo(ids, t1, t2, len(ids))
    assert torch.equal(nuevo, viejo)


def test_sin_t1_alt_el_bug_existe(pipe):
    """Documenta el bug: sin el marker <|im_dev|> el slot developer recibe gradiente."""
    ids = pipe.encode(_traza_dev(pipe))
    t1, t2, t1_alt = _markers(pipe)
    dev_open = _find_marker(ids, t1_alt[0])
    im_start_next = _find(ids, t2, dev_open)

    roto = mask.create_mask(torch.tensor(ids), t1, t2, len(ids))  # sin t1_alt
    leak = [k for k in range(dev_open, im_start_next) if _grad(k, roto, ids)]
    assert leak, "se esperaba fuga de gradiente sin t1_alt (el bug original)"


def test_generate_mask_se_tambien_respeta_im_dev(pipe):
    """El masker 'se' (generate_mask) comparte el fix."""
    ids = pipe.encode(_traza_dev(pipe))
    t1, t2, t1_alt = _markers(pipe)
    labels = mask.generate_mask(torch.tensor(ids), t1, t2, len(ids), t1_alt)
    dev_open = _find_marker(ids, t1_alt[0])
    im_start_next = _find(ids, t2, dev_open)
    for k in range(dev_open, im_start_next):
        assert not _grad(k, labels, ids)
    ans = pipe.encode("corriendo")
    ans_start = _find(ids, ans, im_start_next)
    for k in range(ans_start, ans_start + len(ans)):
        assert int(labels[k - 1]) == ids[k]


def test_dataset_gate_cubre_im_dev():
    """El gate de dataset.py elige los delimitadores ChatML tambien con <|im_dev|>."""
    from rwkvt.dataset import dataset as ds

    assert "<|im_dev|>" in ds.MARCAS_MASK_CTX
    ctx_dev = "<|im_dev|>ahora corre los tests<|im_end|>\n<|im_start|>seco\ncorriendo<|im_end|>"
    assert any(m in ctx_dev for m in ds.MARCAS_MASK_CTX)
    ctx_user = "<|im_user|>che<|im_end|>\n<|im_start|>seco\nhola<|im_end|>"
    assert any(m in ctx_user for m in ds.MARCAS_MASK_CTX)
    ctx_legacy = "User: che\nAssistant: hola"
    assert not any(m in ctx_legacy for m in ds.MARCAS_MASK_CTX)


def _create_mask_viejo(seq, token1, token2, min_len):
    """Copia literal del create_mask previo al fix (match exacto, sin tails)."""
    s, t1, t2 = list(seq), list(token1), list(token2)
    n = min(len(s), min_len)
    starts1 = {i for i in range(max(n - len(t1) + 1, 0))
               if t1 and s[i:i + len(t1)] == t1} if t1 else set()
    starts2 = {i for i in range(max(n - len(t2) + 1, 0))
               if t2 and s[i:i + len(t2)] == t2} if t2 else set()
    keep = [False] * len(s)
    cur = False
    for i in range(n):
        if i in starts1:
            cur = False
        elif i in starts2:
            cur = True
        keep[i] = cur
    if not any(keep[1:min_len]):
        for j in range(1, min(n, min_len)):
            keep[j] = True
    return torch.tensor([s[j + 1] if (j + 1 < min_len and keep[j + 1]) else IGNORE
                         for j in range(len(s) - 1)], dtype=torch.long)
