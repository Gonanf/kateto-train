#!/usr/bin/env python3
"""K3: el style tune adapta el tensor de estilo y NO toca los pesos.

    python3 -m pytest tests/test_style_tune.py -q          # este python no tiene torch
    venv/bin/python -m pytest tests/test_style_tune.py -q  # este si

El test NO importa torch (mismo truco que scripts/test_fix126.py): todo lo que
toca tensores corre en un subproceso con el python que si tiene torch, y aca se
afirman sus exit codes y su salida. Asi el comando del gate verifica de verdad
con cualquiera de los dos interpreters, en vez de dar un verde de skips.

Sin GPU, sin red, sin modelo real: los fixtures son tensores chicos (32 de
embedding, 2 capas) generados en el subproceso.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CLI = REPO / "scripts" / "style_tune_kateto.py"

RC_ERROR = 3  # scripts/style_tune_kateto.py:RC_ERROR

# El head de estilo tiene que ser (4, 8, 8) = (n_embd/head_size, head_size,
# head_size) con n_embd=32 y --head-size 8. Mismo shape que produce `init`.
N_HEAD, HEAD_SIZE = 4, 8
PARAM_POR_TENSOR = N_HEAD * HEAD_SIZE * HEAD_SIZE
N_LAYERS = 2
N_PESOS_BASE = 1 + N_LAYERS * 2  # emb + key/value por capa

FIXTURES = f"""
import sys, torch
torch.manual_seed(0)
C = 32
base = {{"emb.weight": torch.randn(16, C)}}
for i in range({N_LAYERS}):
    base[f"blocks.{{i}}.att.key.weight"] = torch.randn(C, C)
    base[f"blocks.{{i}}.att.value.weight"] = torch.randn(C, C)
torch.save(base, sys.argv[1])

# head de estilo entrenado (el que deja `--peft state`), con valores propios
forma = ({N_HEAD}, {HEAD_SIZE}, {HEAD_SIZE})
torch.save({{f"blocks.{{i}}.att.time_state": torch.full(forma, 0.25 * (i + 1))
            for i in range({N_LAYERS})}}, sys.argv[2])

# head con una forma que NO calza: `apply` tiene que negarse
torch.save({{"blocks.0.att.time_state": torch.full((2, 16, 16), 1.0)}}, sys.argv[3])
"""

# La garantia del K3: `apply` escribe el .pth nuevo y los pesos base salen
# identicos bit a bit. Solo puede Comparar eso quien tiene torch.
COMPARA = """
import sys, torch
base = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
head = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
out  = torch.load(sys.argv[3], map_location="cpu", weights_only=True)
S = ".att.time_state"
pesos = [k for k in base if not k.endswith(S)]
assert pesos, "el fixture base no tiene pesos"
for k in pesos:
    assert k in out, f"apply borro el peso {k}"
    assert torch.equal(out[k], base[k]), f"apply cambio el peso {k}"
estilo = [k for k in out if k.endswith(S)]
assert len(estilo) == len(head), f"estilo {len(estilo)} != head {len(head)}"
for k, v in head.items():
    assert torch.equal(out[k], v), f"el tensor de estilo {k} no quedo como en el head"
print(f"OK pesos_base={len(pesos)} estilo={len(estilo)}")
"""


def _tiene_torch(py: str) -> bool:
    return subprocess.run([py, "-c", "import torch"], capture_output=True).returncode == 0


def _py_con_torch() -> str:
    for c in (REPO / "venv/bin/python", REPO / "venv-unsloth-qwen/bin/python", sys.executable):
        if c.exists() and _tiene_torch(str(c)):
            return str(c)
    pytest.skip(
        "ningun python con torch (probo venv/, venv-unsloth-qwen/ y sys.executable)",
        allow_module_level=True,
    )


PY = _py_con_torch()


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, str(CLI), *args], capture_output=True, text=True, cwd=str(REPO), timeout=120
    )


def _py(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, "-c", code, *args], capture_output=True, text=True, timeout=120
    )


def _campo(stdout: str, nombre: str) -> str:
    m = re.search(rf"^{nombre}: (.+)$", stdout, re.M)
    assert m, f"no encontre '{nombre}:' en la salida:\n{stdout}"
    return m.group(1).strip()


@pytest.fixture()
def fx(tmp_path):
    """base.pth (pesos solos), head.pth (estilo), head_malo.pth (forma que no calza)."""
    paths = {
        "base": tmp_path / "base.pth",
        "head": tmp_path / "head.pth",
        "malo": tmp_path / "head_malo.pth",
        "init": tmp_path / "init.pth",
        "out": tmp_path / "out.pth",
        "out_malo": tmp_path / "out_malo.pth",
    }
    r = _py(FIXTURES, str(paths["base"]), str(paths["head"]), str(paths["malo"]))
    assert r.returncode == 0, r.stdout + r.stderr
    return paths


def test_info_sin_head_falla_legible(fx):
    """Un ckpt de solo pesos: error que se lee, no KeyError."""
    r = _cli("info", str(fx["base"]))
    salida = r.stdout + r.stderr

    assert r.returncode == RC_ERROR, salida
    assert "Traceback" not in salida
    assert "KeyError" not in salida
    assert "time_state" in salida  # dice que clave espera
    assert "init" in salida  # y como se arregla


def test_init_y_info_misma_forma_y_params(fx):
    """`init` y `info` tienen que contar lo mismo, sobre el tensor que escribio init."""
    r = _cli("init", str(fx["base"]), "--out", str(fx["init"]), "--head-size", str(HEAD_SIZE))
    assert r.returncode == 0, r.stdout + r.stderr

    i = _cli("info", str(fx["init"]))
    assert i.returncode == 0, i.stdout + i.stderr

    forma, params = f"({N_HEAD}, {HEAD_SIZE}, {HEAD_SIZE})", str(N_LAYERS * PARAM_POR_TENSOR)
    assert _campo(i.stdout, "forma") == forma == _campo(r.stdout, "forma")
    assert _campo(i.stdout, "params") == params == _campo(r.stdout, "params")
    assert _campo(i.stdout, "tensores") == str(N_LAYERS)
    assert _campo(i.stdout, "capas") == f"[{', '.join(str(x) for x in range(N_LAYERS))}]"


def test_apply_no_toca_los_pesos_base(fx):
    """La garantia del K3: los pesos base salen identicos bit a bit."""
    r = _cli("apply", str(fx["base"]), "--head", str(fx["head"]), "--out", str(fx["out"]),
             "--head-size", str(HEAD_SIZE))
    assert r.returncode == 0, r.stdout + r.stderr

    c = _py(COMPARA, str(fx["base"]), str(fx["head"]), str(fx["out"]))
    assert c.returncode == 0, c.stdout + c.stderr
    assert f"pesos_base={N_PESOS_BASE} estilo={N_LAYERS}" in c.stdout

    i = _cli("info", str(fx["out"]))  # el .pth nuevo se describe solo
    assert i.returncode == 0, i.stdout + i.stderr
    assert _campo(i.stdout, "tensores") == str(N_LAYERS)


def test_apply_que_no_calza_no_escribe_nada(fx):
    """Un head con otra forma se rechaza entero y no deja un .pth a medias."""
    r = _cli("apply", str(fx["base"]), "--head", str(fx["malo"]), "--out", str(fx["out_malo"]),
             "--head-size", str(HEAD_SIZE))
    salida = r.stdout + r.stderr

    assert r.returncode == RC_ERROR, salida
    assert "Traceback" not in salida
    assert "!= (4, 8, 8)" in salida, salida  # dice las dos formas, no solo que fallo
    assert not fx["out_malo"].exists(), "apply escribio un .pth apesar del error"