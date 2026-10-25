"""fix-125: guard de estado, filtro [INST] por fuente y cortes de generacion por ids.

Los helpers mas importantes (corte por secuencia de ids, guarda n-grama y el
predicado de [INST]) son puros (sin torch), asi que se testean con `python3`
pelado — el gate de pytest corre sin torch.

build-train-set.py tiene guion en el nombre, asi que se carga por ruta con
importlib; _stop_helpers.py vive en rwkv_pipeline/ y no importa torch.
"""
import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
PIPELINE = REPO / "rwkv_pipeline"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bt = _load("build_train_set", REPO / "kateto-medicion" / "build-train-set.py")
_stop = _load("stop_helpers", PIPELINE / "_stop_helpers.py")

IM_END = [61, 125, 1949, 96, 7463, 125, 63]  # encode('<|im_end|>') medido


# --- punto 2: filtro [INST] por fuente ---------------------------------------
def test_inst_predicate_por_fuente():
    assert _bt.tiene_inst("texto con [INST] adentro")
    assert _bt.tiene_inst("chau [/INST]")
    assert not _bt.tiene_inst("hola mundo <|im_end|>")


# --- punto 3: corte por ids --------------------------------------------------
def test_ends_with_ids_marcador_completo():
    assert _stop.ends_with([1, 2] + IM_END, IM_END)
    assert _stop.ends_with(IM_END, IM_END)


def test_ends_with_ids_rechaza_incompleto():
    # 6 de 7 (el caso degenerado: el stop por string nunca disparaba) NO corta por ids
    assert not _stop.ends_with([1, 2] + IM_END[:-1], IM_END)
    assert not _stop.ends_with([], IM_END)
    assert not _stop.ends_with([61, 125], IM_END)


def test_repetition_ngram_consecutivo():
    n = 3
    rep = [1, 2, 3] * 6
    # el ultimo [1,2,3] ya aparecio 5 veces antes -> 6 ocurrencias consecutivas
    assert _stop.count_consecutive_ngram_repeats(rep, n) == 5
    assert _stop.count_consecutive_ngram_repeats([1, 2, 3, 4, 5, 6], n) == 0
    assert _stop.count_consecutive_ngram_repeats([1] * 10, 1) == 9


def test_repetition_corta_a_ngram_repeat_max():
    # 12 ocurrencias consecutivas del mismo trigrama deben superar el umbral default
    rep = [7, 8, 9] * 12
    assert _stop.count_consecutive_ngram_repeats(rep, 3) + 1 >= 12