#!/usr/bin/env python3
"""R16 - la whitelist de terminos tecnicos: que recupera y que no.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python scripts/pipelines/medir_spanglish.py

Corre el gate DOS veces sobre los mismos datos: una con la regla de R16 SIN la
whitelist (o sea, la regla como estaba) y una con la whitelist puesta, y cuenta
los rechazos de `spanglish` en las dos. Todo numero del reporte sale de aca.

No hay una segunda copia de la lista: el "antes" se arma parcheando
`RX_TECNICOS` por una regex que no matchea nunca, y las dos pasadas llaman a la
misma `ref.validate` que usa el generador y la purga.

Los datos de entrada son de solo lectura; lo unico que escribe este script es
`scripts/pipelines/reporte-spanglish.json`.
"""

from __future__ import annotations

import contextlib
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

PIPELINES = Path(__file__).resolve().parent
REPO = PIPELINES.parent.parent
for _p in (PIPELINES, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kateto_purge as kp  # noqa: E402
import gen_toolcalling_dataset as ref  # noqa: E402

REPORT_PATH = PIPELINES / "reporte-spanglish.json"
MAX_EJEMPLOS = 3

# La whitelist vacia. Es TODO lo que hace falta para volver al "antes": la regla
# sigue siendo la de `RX_SPANGLISH`, sin excepciones.
SIN_WHITELIST = re.compile(r"(?!x)x")

CORPUS: dict[str, list[Path]] = {
    # El train set crudo y los shards de la corrida anterior.
    "purga_entrada": [REPO / "data" / "rwkv_kateto_base_train_plus.jsonl"]
    + [Path(p) for p in sorted(glob.glob(str(REPO / "out/dataset/shards/parcial-w*.jsonl")))],
    # Los rechazos que guardo la purga anterior.
    "v2_rechazos": [PIPELINES / "smoke" / "teacher-pool-v2" / "rechazos.jsonl"],
}


@contextlib.contextmanager
def reglas_antes() -> Iterator[None]:
    """Saca la whitelist. Al salir, todo queda como estaba."""
    saved = ref.RX_TECNICOS
    ref.RX_TECNICOS = SIN_WHITELIST
    try:
        yield
    finally:
        ref.RX_TECNICOS = saved


def _filas(rutas: list[Path]) -> list[str]:
    """Los textos del corpus, leidos UNA vez: el train son 60 MB y el script los
    recorre cuatro veces (antes, despues, whitelist, ejemplos)."""
    filas: list[str] = []
    for ruta in rutas:
        with ruta.open(encoding="utf-8") as fh:
            filas += [json.loads(linea)["text"] for linea in fh if linea.strip()]
    return filas


def voces_de(texto: str) -> list[str]:
    return [n.strip() for n, _ in kp.VOICE_TURN_RE.findall(texto)] or ["seco"]


def spanglish_de(texto: str) -> set[str]:
    """Las palabras de spanglish que el gate le atribuye a la linea. El gate es el
    de verdad (`ref.validate`), no una copia de la regla: asi el "antes" y el
    "despues" se miden con la misma puerta que usa la purga."""
    palabras: set[str] = set()
    for voz in voces_de(texto):
        for err in ref.validate(texto, voice=voz):
            if err.startswith("spanglish("):
                palabras.add(err.removeprefix("spanglish(").rstrip(")"))
    return palabras


def medir(filas: list[str]) -> dict[str, Any]:
    """Pasada completa: una vez sin whitelist, una vez con ella.

    Las dos consultas van en el MISMO bucle a proposito: la linea es la unidad de
    conteo y asi no hay forma de que las dos pasadas se desincronicen.
    """
    rechazos = {"antes": 0, "despues": 0}
    recuperadas = 0
    palabras_aceptadas: Counter = Counter()
    ejemplos: dict[str, list[str]] = {}
    for texto in filas:
        with reglas_antes():
            antes = spanglish_de(texto)
        despues = spanglish_de(texto)
        rechazos["antes"] += bool(antes)
        rechazos["despues"] += bool(despues)
        nuevas = antes - despues
        if not nuevas:
            continue
        recuperadas += 1
        for palabra in nuevas:
            palabras_aceptadas[palabra] += 1
            if len(ejemplos.setdefault(palabra, [])) < MAX_EJEMPLOS:
                ejemplos[palabra].append(texto[:260])
    return {
        "lineas": len(filas),
        "rechazos_spanglish": rechazos,
        "lineas_recuperadas": recuperadas,
        "palabras_que_ahora_se_aceptan": dict(palabras_aceptadas.most_common()),
        "ejemplos": ejemplos,
    }


def medir_whitelist(filas: list[str]) -> dict[str, dict[str, int]]:
    """Frecuencia de cada término técnico en el TURNO DE LA VOZ, que es donde la
    regla se aplica. Es la medicion que justifies la lista: son las apariciones que
    la whitelist pone a salvo para siempre, y las que se romperian si la lista
    negra creciera sin el criterio del dueño."""
    rx = ref.RX_TECNICOS
    en_voz: Counter = Counter()
    rechaza: Counter = Counter()
    for texto in filas:
        for voz in kp.turnos_voz(texto):
            for m in rx.finditer(voz):
                en_voz[m.group(0).lower()] += 1
        for palabra in spanglish_de(texto):
            rechaza[palabra] += 1
    return {
        t.lower(): {"turnos_de_voz": en_voz[t], "rechazos": rechaza[t]}
        for t in ref.TERMINOS_TECNICOS
    }


def ejemplos_whitelist(filas: list[str], termino: str, n: int = MAX_EJEMPLOS) -> list[str]:
    """Muestra de turnos de voz donde aparece el término: el texto que la regla
    aceptaba por casualidad y que hoy acepta a propósito. El recorte se centra en
    la aparición, para que el ejemplo muestre el término en su contexto."""
    rx = re.compile(rf"\b{re.escape(termino)}\b", re.I)
    vistos: list[str] = []
    for texto in filas:
        for voz in kp.turnos_voz(texto):
            limpio = " ".join(voz.split())
            m = rx.search(limpio)
            if not m:
                continue
            fragmento = limpio[max(0, m.start() - 90): m.end() + 90]
            if not any(fragmento in v for v in vistos):
                vistos.append(fragmento)
            if len(vistos) >= n:
                return vistos
    return vistos


# Lo que el dueño NO autorizó como prestamo.
NO_TECNICOS = ("worries", "worry", "okay", "okey", "different", "differente")


def medir_no_tecnicos(filas: list[str]) -> dict[str, dict[str, int]]:
    """Las palabras que el dueño descartó explícitamente, medidas. `worries`/`okay`
    tienen traducción de uso ('preocupación', 'dale'), asi que no son prestamos
    técnicos; `differente` no es inglés sino una variante del castellano. La
    medicion dice donde aparecen y que NO las rechaza el motivo `spanglish`."""
    en_voz: Counter = Counter()
    rechaza: Counter = Counter()
    rx = re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, NO_TECNICOS)), re.I)
    for texto in filas:
        for voz in kp.turnos_voz(texto):
            for m in rx.finditer(voz):
                en_voz[m.group(0).lower()] += 1
        for palabra in spanglish_de(texto):
            rechaza[palabra] += 1
    return {p: {"turnos_de_voz": en_voz[p], "rechazos_spanglish": rechaza[p]}
            for p in NO_TECNICOS}


def main() -> int:
    corpus: dict[str, Any] = {}
    filas_por_corpus: dict[str, list[str]] = {}
    for nombre, rutas in CORPUS.items():
        faltan = [p for p in rutas if not p.exists()]
        if faltan:
            print(f"saltando {nombre}: falta {faltan[0]}", flush=True)
            continue
        print(f"midiendo {nombre} ({len(rutas)} archivos)...", flush=True)
        filas = _filas(rutas)
        medido = medir(filas)
        medido["archivos"] = [str(p.relative_to(REPO)) for p in rutas]
        corpus[nombre] = medido
        filas_por_corpus[nombre] = filas

    faltan_shards = not glob.glob(str(REPO / "out/dataset/shards/parcial-w*.jsonl"))

    total = {k: sum(c["rechazos_spanglish"][k] for c in corpus.values())
             for k in ("antes", "despues")}
    palabras: Counter = Counter()
    for c in corpus.values():
        palabras.update(c["palabras_que_ahora_se_aceptan"])
    recuperadas = sum(c["lineas_recuperadas"] for c in corpus.values())
    ejemplos = {p: e for c in corpus.values() for p, e in c["ejemplos"].items()}

    entrada = filas_por_corpus.get("purga_entrada", [])
    whit = medir_whitelist(entrada)
    apariciones = sum(d["turnos_de_voz"] for d in whit.values())
    solape = [t for t in ref.TERMINOS_TECNICOS if ref.RX_SPANGLISH.fullmatch(t)]
    a_pista = sorted(((t, d["turnos_de_voz"]) for t, d in whit.items()),
                     key=lambda x: -x[1])[:3]

    reporte = {
        "meta": {
            "brief": "R16 - el spanglish no va: whitelist de terminos tecnicos",
            "generado_por": "scripts/pipelines/medir_spanglish.py",
            "regla": "scripts/gen_toolcalling_dataset.py::spanglish_en (la de validate)",
            "whitelist": "scripts/gen_toolcalling_dataset.py::TERMINOS_TECNICOS",
            "reglas_de_antes": (
                "la MISMA regla con `RX_TECNICOS` parcheado por una regex que no "
                "matchea: la lista negra intacta, sin excepciones"
            ),
            "criterio_de_la_whitelist": (
                "entra solo si en castellano rioplatense se dice asi y no hay "
                "traduccion de uso. No entra la palabra en ingles con equivalente "
                "corriente: por eso `worries`, `okay` y `okey` siguen rechazados y "
                "`differente` -que no es ingles- no esta en la lista"
            ),
            "conteo": "lineas rechazadas por el motivo `spanglish`",
        },
        "por_motivo_spanglish": total,
        "lineas_recuperadas": recuperadas,
        "palabras_que_ahora_se_aceptan": dict(palabras.most_common()),
        "ejemplos": ejemplos,
        "notas": {
            **({"shards": "no hay shards en `out/dataset/shards/`: el corpus de "
                          "entrada es el train set crudo y solo el"} if faltan_shards else {}),
            "recuperadas_0": (
                "la whitelist no recupera ninguna linea porque NO se solapa con la "
                "lista negra: `RX_SPANGLISH` no tiene ningun termino tecnico. Eso es "
                "el resultado correcto, no un fallo: la regla de R16 sigue tan dura "
                "como antes y la whitelist queda como excepcion explicita para cuando "
                "la lista negra crezca"
            ) if recuperadas == 0 else {},
        },
        "por_corpus": corpus,
        "medicion_whitelist": {
            "sobre": "purga_entrada",
            "que_es": (
                "apariciones de cada termino en el TURNO DE LA VOZ, donde la regla se "
                "aplica, y rechazos que la lista negra le atribuye hoy. Son las "
                "apariciones que la whitelist pone a salvo"
            ),
            "total_apariciones_en_voz": apariciones,
            "solape_con_la_lista_negra": solape,
            "top": [{"termino": t, "turnos_de_voz": n} for t, n in a_pista],
            "terminos": whit,
            "ejemplos": {t: ejemplos_whitelist(entrada, t) for t, _ in a_pista},
        },
        "palabras_descartadas": {
            "que_es": (
                "lo que el dueño NO autorizó como préstamo técnico. `worries`, "
                "`okay`, `okey` y `worry` tienen traducción de uso ('preocupación', "
                "'dale'), asi que siguen rejecting; `different` es inglés pero con "
                "traducción corriente, y `differente` no es inglés sino una variante "
                "del castellano: ninguna va en la whitelist"
            ),
            "terminos": medir_no_tecnicos(entrada),
        },
    }
    REPORT_PATH.write_text(json.dumps(reporte, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"reporte: {REPORT_PATH}", flush=True)
    print(f"  spanglish  {total['antes']:6d} -> {total['despues']:6d}"
          f"  (recuperadas: {recuperadas})", flush=True)
    print(f"  whitelist  {apariciones:6d} apariciones en turnos de voz, hoy a salvo",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())