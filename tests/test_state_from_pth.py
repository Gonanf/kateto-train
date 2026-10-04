#!/usr/bin/env python3
"""B3: el .pth nativo -> el byte-stream que llama.cpp acepta.

    env -u PYTHONPATH -u VIRTUAL_ENV ./venv/bin/python -m pytest tests/test_state_from_pth.py -q

El `.pth` real de `out/rwkv_states/seco/` es de un RWKV-7 0.4B (24 capas, n_embd
1024) y el GGUF de `out/` es un 2.9B (32 capas, n_embd 2560): no calzan, asi que
el camino de fabricacion se prueba con un `.pth` sintetico con las formas que el
modelo pide. Lo que se prueba ahi es el layout, no la semantica.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rwkv_pipeline.state_bytes import parse_state_bytes
from rwkv_pipeline.state_from_pth import diagnostico, estado_a_bytes, leer_hparams

PTH_REAL = Path(os.environ.get("KATETO_STATE_PTH", "out/rwkv_states/seco/rwkv-3.pth"))
GGUF = Path(os.environ.get("KATETO_STATE_GGUF", "out/kateto-rwkv29-Q4_K_M.gguf"))
DUMP = Path(os.environ.get("KATETO_STATE_DUMP", "/tmp/estado-orig.bin"))

# Hiperparametros reales de out/kateto-rwkv29-Q4_K_M.gguf (ver test_state_bytes.py).
N_LAYER = 32
N_EMBD = 2560
HEAD = 64
N_HEAD = N_EMBD // HEAD
N_EMBD_R = 2 * N_EMBD
N_EMBD_S = N_EMBD * HEAD
SIZE_ESPERADO = 8 + 4 + 8 + 8 + N_LAYER * 12 + N_LAYER * N_EMBD_R * 4 \
    + N_LAYER * 12 + N_LAYER * N_EMBD_S * 4


def _saltar_sin_fichero(p: Path) -> None:
    if not p.exists():
        pytest.skip(f"falta {p}")


@pytest.fixture(scope="module")
def hp():
    _saltar_sin_fichero(GGUF)
    return leer_hparams(GGUF)


@pytest.fixture(scope="module")
def pth_sintetico(tmp_path_factory, hp):
    """.pth con las formas que el modelo pide, relleno con valores distinguibles."""
    _saltar_sin_fichero(GGUF)
    rng = np.random.default_rng(7)
    sd = {
        f"blocks.{il}.att.time_state": torch.from_numpy(
            rng.standard_normal((hp["n_head"], hp["head_size"], hp["head_size"]), dtype=np.float32)
        )
        for il in range(hp["n_layer"])
    }
    p = tmp_path_factory.mktemp("pth") / "estado.pth"
    torch.save(sd, p)
    return p


# ------------------------------------------------------- el veredicto (real)


def test_el_pth_real_no_calza_con_el_gguf(hp):
    """El resultado de B3 con los numeros: 24 capas de (16,64,64) contra 32 de (40,64,64)."""
    _saltar_sin_fichero(PTH_REAL)
    dx = diagnostico(PTH_REAL, GGUF)
    assert dx["n_tensores_pth"] == 24, "el .pth de voz trae 24 time_state"
    assert dx["shape_pth"] == (16, 64, 64)
    assert dx["todos_mismo_shape"] and dx["capas_pth_contiguas"]
    assert (hp["n_layer"], hp["n_embd"], hp["head_size"]) == (32, 2560, 64)
    assert not dx["calza"], "1024 != 2560: no puede calzar"
    assert dx["n_capas_ok"] == 0


def test_el_pth_real_no_se_puede_fabricar():
    """estado_a_bytes se niega en vez de inventar tensores."""
    _saltar_sin_fichero(PTH_REAL)
    with pytest.raises(ValueError, match="no calzan"):
        estado_a_bytes(PTH_REAL, modelo_gguf=GGUF)


def test_report_sale_con_3():
    """`--report` sobre el .pth real: exit 3, que es lo que lee el script."""
    _saltar_sin_fichero(PTH_REAL)
    _saltar_sin_fichero(GGUF)
    r = subprocess.run(
        [sys.executable, "-m", "rwkv_pipeline.state_from_pth",
         "--pth", str(PTH_REAL), "--modelo", str(GGUF), "--report"],
        cwd=PROJECT, capture_output=True, text=True,
    )
    assert r.returncode == 3, r.stdout[-2000:]
    assert "NO CALZA" in r.stdout
    assert "(16, 64, 64)" in r.stdout and "(2560, 64)" in r.stdout


# ------------------------------------------------- la fabrica (pth sintetico)


def test_hparams_del_gguf(hp):
    assert hp["arquitectura"] == "rwkv7"
    assert (hp["n_layer"], hp["n_embd"], hp["head_size"], hp["n_head"]) == (
        N_LAYER, N_EMBD, HEAD, N_HEAD,
    )
    assert hp["n_embd_r"] == N_EMBD_R
    assert hp["n_embd_s"] == N_EMBD_S


def test_el_buffer_sintetico_tiene_el_tamano_del_layout(pth_sintetico):
    """Mismo tamano que el dump real de llama.cpp para este modelo."""
    _saltar_sin_fichero(GGUF)
    buf = estado_a_bytes(pth_sintetico, modelo_gguf=GGUF)
    assert len(buf) == SIZE_ESPERADO == 21627676


def test_los_tensores_van_a_s_y_en_orden(pth_sintetico):
    """Cada time_state del .pth queda en el s de su capa, sin reordenar."""
    _saltar_sin_fichero(GGUF)
    buf = estado_a_bytes(pth_sintetico, modelo_gguf=GGUF)
    datos = parse_state_bytes(buf)
    assert datos["cell_count"] == 1 and datos["seq_id"] == 0
    assert len(datos["capas"]) == N_LAYER
    sd = torch.load(pth_sintetico, map_location="cpu", weights_only=True)
    for il in range(N_LAYER):
        capa = datos["capas"][il]
        assert capa["p"] is None, "rwkv7 no tiene p"
        assert capa["r"].shape == (1, N_EMBD_R)
        assert capa["s"].shape == (1, N_EMBD_S)
        esperado = sd[f"blocks.{il}.att.time_state"].numpy().reshape(-1)
        np.testing.assert_array_equal(capa["s"][0], esperado)
    print(f"\nsintetico OK: {len(buf)} bytes, {N_LAYER} capas")


def test_r_queda_en_ceros_y_eso_dice(pth_sintetico):
    """El token shift no esta en el .pth: r va en cero, y el modulo lo declara."""
    _saltar_sin_fichero(GGUF)
    datos = parse_state_bytes(estado_a_bytes(pth_sintetico, modelo_gguf=GGUF))
    for il, capa in enumerate(datos["capas"]):
        assert not capa["r"].any(), f"capa {il}: r deberia estar en cero"


def test_los_descriptores_de_fila_calzan_con_el_dump_real(pth_sintetico):
    """Si hay dump de B1, nuestro buffer tiene los mismos tipos y anchos de fila."""
    if not DUMP.exists():
        pytest.skip(f"no hay dump de estado en {DUMP}; lo genera script/probar-estado-python.sh")
    _saltar_sin_fichero(GGUF)
    a = parse_state_bytes(DUMP.read_bytes())
    b = parse_state_bytes(estado_a_bytes(pth_sintetico, modelo_gguf=GGUF))
    for x, y in zip(a["capas"], b["capas"]):
        assert x["r"].shape == y["r"].shape
        assert x["s"].shape == y["s"].shape
    assert a["cell_count"] == b["cell_count"] and len(a["capas"]) == len(b["capas"])
