#!/usr/bin/env python3
"""R1 — Purga de muestras contaminadas del train set de Kateto.

Pipeline de distilabel: lee el train set y los shards, anota cada linea con las
reglas de purga que dispara, y escribe la salida limpia + el reporte.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python pipelines/kateto_purge.py

El formato del dato no se toca:
    <|im_user|>texto<|im_end|>\n<|im_start|>{voz}\nrespuesta<|im_end|>

Plan B documentado en `reporte-purga-v2.json` (`-v1.json` es la evidencia
del bug de la rama `token_repetido`, ya corregido): distilabel 1.5.3 si instalo y su
modelo de Steps sirve, con el matiz de que los Steps conservan el numero de filas
(anotan columnas) y el filtrado real ocurre en el Step sink.
"""

# Sin `from __future__ import annotations`: distilabel detects el parametro de
# entrada de un Step por identidad sobre el Annotated real, y con PEP 563 la
# anotacion llega como string y el Step se rechaza.
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from distilabel.typing import StepColumns, StepOutput

REPO = Path(__file__).resolve().parent.parent
TRAIN_PATH = REPO / "data" / "rwkv_kateto_base_train_plus.jsonl"
SHARD_GLOB = "parcial-w*.jsonl"
SHARD_DIR = REPO / "out" / "dataset" / "shards"
OUT_DIR = Path("/home/chaos/harness-run/kateto-purge")
REPORT_PATH = REPO / "pipelines" / "reporte-purga-v2.json"
# R1 queda como evidencia del bug: nunca se sobreescribe.
V1_REPORT_PATH = REPO / "pipelines" / "reporte-purga.json"

TRAIN_KEY = "train_set"
SHARDS_KEY = "shards"
SOURCES = (TRAIN_KEY, SHARDS_KEY)

OUT_FILES = {
    TRAIN_KEY: "rwkv_kateto_base_train_plus.purged.jsonl",
    SHARDS_KEY: "shards.purged.jsonl",
}

# Columnas que se preservan en la salida limpia, por fuente. El formato del dato
# no cambia: solo se descartan lineas.
KEEP_COLS = {
    TRAIN_KEY: ("text",),
    SHARDS_KEY: ("text", "scenario", "cat", "voice"),
}

# ponytail: marcadores de turno tal cual vienen en el dato; 4 tokens, no un
# parser de chat. Si el formato cambia, este es el unico lugar a tocar.
TURN_RE = re.compile(r"<\|im_(?:user|start)\|>(.*?)<\|im_end\|>", re.S)
VOICE_TURN_RE = re.compile(r"<\|im_start\|>([^\n]*)\n(.*?)<\|im_end\|>", re.S)
SIMIL_RE = re.compile(r"\bcomo (?:cuando|si|el|la|un|una)\b", re.I)
# `KATET+O+` del brief es notacion de regex: KATET seguido de una o mas O.
PLACEHOLDER_RE = re.compile(r"KATETO+", re.I)
ANDAMIAJE_RE = re.compile(r"Diagnóstico:|Cierre:")
CONTRATO_RE = re.compile(r"^\s*(?:VOS|PERSONA):", re.M)
EXCLAMACION_RE = re.compile(r"[¡!]")
# La rama repetida necesita dos ajustes que R1 no tenia: limpiar los marcadores
# del formato (pegados al token, tapan la palabra degenerada) y exigir letra, para
# que `1000` no cuente y `...`/`---`/`===` no cuenten nunca.
MARCADOR_RE = re.compile(r"<\|[a-z_]+\|>")
REPETIDO_RE = re.compile(r"([a-záéíóúüñ])\1{2,}$", re.I)
# La rama de R1 tal cual (pegaba solo puntuacion). Vive solo para medir cuantas
# lineas recupera el arreglo; no participa de la decision de purgar.
REPETIDO_V1_RE = re.compile(r"(\S)\1{2,}$")

# La voz de asistente: aclaracion, disclaimer, leccion y moralina. Frases
# sueltas, no un clasificador. Se comparan sobre texto normalizado (minusculas,
# sin acentos, apostrofe ASCII) asi que `estas` y `esta` son el mismo patron y
# el apostrofo tipografico no se escapa. Una sola tabla: la reusa la medicion de
# `pipelines/medir_higiene.py`, asi que agregar un patron es agregar un string.
CHATLA_PATRONES: tuple[str, ...] = (
    # aclaracion, leccion, moralina
    "es importante",
    "tenes que tener en cuenta",
    "cabe aclarar",
    "en resumen",
    "en conclusion",
    "no dudes en",
    "recuerda que",
    "deberias",
    "espero que",
    # juicio / valoracion
    "no esta bien",
    # `esta mal` SALIO de aca (R12, triage #7): como substring suelto matchea
    # cualquier argumentacion normal ("El capitalismo esta mal", "no esta mal
    # para 27B"): las 618 lineas que rechazaba sola eran discusiones, no
    # moralinas. Exigir contexto moral ("esta mal que", "lo que esta mal es")
    # tampoco servia — sobre esas 618 matcheaba 10 veces y casi ninguna era
    # moralina. El juicio moral ya lo cubren las de arriba, que no matchean
    # discusion.
    # autodeclaracion de asistente
    "como asistente",
    "como ia",
    "como modelo de lenguaje",
    # rechazo y limitacion de capacidad
    "no puedo ayudarte",
    "no puedo proporcionar",
    "no puedo generar",
    "no tengo acceso",
    "lo siento, pero",
    "lo siento",
    # rechazo en ingles dentro de un turno en espanol
    "as an ai",
    "i can't",
    "i cannot",
    "i'm unable",
    "i'm sorry",
    "i must refuse",
)

MAX_EJEMPLOS = 3
EJEMPLO_CHARS = 400


# --------------------------------------------------------------------------
# Parsers de turno
# --------------------------------------------------------------------------
def turnos(text: str) -> list[str]:
    """Contenido de todos los turnos (de usuario y de voz), en orden."""
    return TURN_RE.findall(text)


def turnos_voz(text: str) -> list[str]:
    """Respuestas de los turnos de la VOZ (sin el nombre de la voz)."""
    return [resp for _, resp in VOICE_TURN_RE.findall(text)]


def sin_acentos(texto: str) -> str:
    """Minusculas, sin diacríticos y con el apostrofo tipografico ya ASCII."""
    plano = unicodedata.normalize("NFKD", texto.lower().replace("’", "'"))
    return "".join(c for c in plano if not unicodedata.combining(c))


def _conteo_similes(respuestas: list[str]) -> list[int]:
    return [len(SIMIL_RE.findall(r)) for r in respuestas]


# --------------------------------------------------------------------------
# Reglas de purga. Una funcion por regla. True = descartar.
# --------------------------------------------------------------------------
def _token_repetido(text: str) -> bool:
    """Una PALABRA que termina con 3+ del mismo caracter (`holaaaa`, `KATETOOOOO`).

    `REPETIDO_RE` exige letra al final, asi que los tokens que son solo
    puntuacion se descartan solos: `...`, `---` y `===` nunca cuentan.
    """
    return any(REPETIDO_RE.search(tok) for tok in MARCADOR_RE.sub(" ", text).split())


def _token_repetido_v1(text: str) -> bool:
    """La rama de R1, sin limpiar marcadores: solo para medir la recuperacion."""
    return any(REPETIDO_V1_RE.search(tok) for tok in text.split())


def _clausula_placeholder(text: str) -> list[str]:
    """Cual de las dos ramas de `placeholder_degenerado` dispara, o ninguna.

    La rama `KATETO` mira SOLO los turnos de la voz, igual que `simil_repetido` y
    `charla_de_asistente` de esta misma tabla (R12). Antes corria sobre el texto
    entero y el usuario llamando por su nombre al personaje —"Che Kateto, estas?"—
    era indistinguishable del placeholder: medido sobre las 95.133 lineas de
    entrada de la purga, 3.829 descartes-tenian el `KATETO` solo en el turno del
    usuario. La rama `token_repetido` sigue sobre el texto entero a proposito: una
    palabra degenerada (`KATETOOOOO`, `holaaaa`) lo es donde este, y achicarla
    seria volver al bug que R1b ya cerro.
    """
    ramas = []
    if any(PLACEHOLDER_RE.search(v) for v in turnos_voz(text)):
        ramas.append("KATETO")
    if _token_repetido(text):
        ramas.append("token_repetido")
    return ramas


def placeholder_degenerado(text: str) -> bool:
    """`KATET+O+` o un token que termina con 3+ del mismo caracter."""
    return bool(_clausula_placeholder(text))


def andamiaje_en_dialogo(text: str) -> bool:
    """Etiquetas de receta (`Diagnóstico:` / `Cierre:`) dentro de un turno."""
    return any(ANDAMIAJE_RE.search(t) for t in turnos(text))


def contrato_en_dialogo(text: str) -> bool:
    """Etiquetas de contrato (`VOS:` / `PERSONA:`) al inicio de linea de un turno."""
    return any(CONTRATO_RE.search(t) for t in turnos(text))


def exclamacion_prosodia(text: str) -> bool:
    """Signos de exclamacion, que delatan la prosodia escrita de la plantilla."""
    return bool(EXCLAMACION_RE.search(text))


def simil_repetido(text: str) -> bool:
    """Similes `como (cuando|si|el|la|un|una)` repetidos en los turnos de la voz.

    Descarta si hay 2+ en turnos consecutivos, o si hay >= 2 en una muestra con
    <= 3 turnos de voz. Un simil suelto se cuenta pero no descarta.
    """
    conteos = _conteo_similes(turnos_voz(text))
    if sum(conteos) < 2:
        return False
    if len(conteos) <= 3:
        return True
    return any(
        conteos[i] >= 1 and conteos[i + 1] >= 1 for i in range(len(conteos) - 1)
    )


def charla_disparadores(text: str) -> set[str]:
    """Que patrones de `CHATLA_PATRONES` dispara el texto.

    Solo mira turnos de la VOZ: el usuario diciendo "no puedo ayudarte" es el
    usuario, no la voz de asistente que hay que sacar del dataset.
    """
    out: set[str] = set()
    for turno in turnos_voz(text):
        plano = sin_acentos(turno)
        out.update(p for p in CHATLA_PATRONES if p in plano)
    return out


def charla_de_asistente(text: str) -> bool:
    """Turnos de la VOZ que suenan a asistente: aclaracion, disclaimer, leccion
    o moralina (`como asistente`, `lo siento, pero`, `es importante`)."""
    return bool(charla_disparadores(text))


REGLAS: dict[str, Any] = {
    "placeholder_degenerado": placeholder_degenerado,
    "andamiaje_en_dialogo": andamiaje_en_dialogo,
    "contrato_en_dialogo": contrato_en_dialogo,
    "exclamacion_prosodia": exclamacion_prosodia,
    "simil_repetido": simil_repetido,
    "charla_de_asistente": charla_de_asistente,
}

# Escape hatch de mutacion para los tests: PURGE_MUTAR=<regla> invierte ese
# predicado al importar, y la suite tiene que caer.
if _mutar := os.environ.get("PURGE_MUTAR"):
    _orig = REGLAS[_mutar]
    REGLAS[_mutar] = lambda t: not _orig(t)  # noqa: E731


def evaluar(text: str) -> list[str]:
    """Nombres de todas las reglas que dispara el texto (no cortocircuita)."""
    return [nombre for nombre, fn in REGLAS.items() if fn(text)]


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------
def leer_train() -> list[dict[str, Any]]:
    with TRAIN_PATH.open(encoding="utf-8") as fh:
        return [{"text": ln["text"], "_fuente": TRAIN_KEY} for ln in map(json.loads, fh)]


def leer_shards() -> list[dict[str, Any]]:
    files = sorted(SHARD_DIR.glob(SHARD_GLOB))
    rows: list[dict[str, Any]] = []
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for ln in map(json.loads, fh):
                rows.append({**ln, "_fuente": SHARDS_KEY})
    return rows


# --------------------------------------------------------------------------
# Steps de distilabel
# --------------------------------------------------------------------------
from distilabel.pipeline import Pipeline  # noqa: E402
from distilabel.steps import (  # noqa: E402
    LoadDataFromDicts,
    Step,
    StepInput,
)
from pydantic import PrivateAttr  # noqa: E402


class AnotarReglas(Step):
    """Anota cada linea con `_reglas`: las reglas de purga que dispara.

    No descarta nada: distilabel conserva el numero de filas entre Steps, asi que
    el filtrado real ocurre en el sink.
    """

    @property
    def inputs(self) -> "StepColumns":
        return ["text"]

    @property
    def outputs(self) -> "StepColumns":
        return ["_reglas"]

    def process(self, *inputs: StepInput) -> "StepOutput":
        for rows in inputs:
            for row in rows:
                row["_reglas"] = evaluar(row["text"])
            yield rows


class EscribirPurgado(Step):
    """Sink: descarta, cuenta por regla y escribe la salida limpia de una fuente.

    Cada invocacion del pipeline maneja una sola fuente y deja su fragmento de
    reporte en `_stats-<fuente>.json`; `main()` los mergea. El sink corre en su
    propio proceso, asi que los contadores no pueden vivir en memoria del
    orquestador.
    """

    fuente: str

    _fh: Any = PrivateAttr(default=None)
    _entrada: int = PrivateAttr(default=0)
    _salida: int = PrivateAttr(default=0)
    _por_regla: Counter = PrivateAttr(default_factory=Counter)
    _multi: int = PrivateAttr(default=0)
    _similes: dict = PrivateAttr(
        default_factory=lambda: {p: Counter() for p in ("antes", "despues")}
    )
    _ejemplos: dict = PrivateAttr(default_factory=lambda: defaultdict(list))
    _clausulas: Counter = PrivateAttr(default_factory=Counter)
    _recuperadas: int = PrivateAttr(default=0)
    _recuperadas_vivas: int = PrivateAttr(default=0)

    @property
    def inputs(self) -> "StepColumns":
        return ["text", "_reglas"]

    @property
    def outputs(self) -> "StepColumns":
        return []

    def _handle(self, fuente: str):
        if fuente not in self._handles:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            self._handles[fuente] = (OUT_DIR / OUT_FILES[fuente]).open(
                "w", encoding="utf-8"
            )
        return self._handles[fuente]

    def _handle(self):
        if self._fh is None:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            self._fh = (OUT_DIR / OUT_FILES[self.fuente]).open("w", encoding="utf-8")
        return self._fh

    def process(self, inputs: StepInput) -> "StepOutput":
        for row in inputs:
            reglas = row["_reglas"]
            self._entrada += 1

            respuestas = turnos_voz(row["text"])
            self._similes["antes"]["turnos"] += len(respuestas)
            self._similes["antes"]["similes"] += sum(_conteo_similes(respuestas))

            # Lo que la rama de R1 descartaba y la corregida no: lo recupera el
            # arreglo. De esas, las que no cae por ninguna otra regla vuelven al
            # train set.
            if "placeholder_degenerado" not in reglas and _token_repetido_v1(
                row["text"]
            ):
                self._recuperadas += 1
                if not reglas:
                    self._recuperadas_vivas += 1

            if reglas:
                self._salida += 1
                if len(reglas) > 1:
                    self._multi += 1
                if "placeholder_degenerado" in reglas:
                    for rama in _clausula_placeholder(row["text"]):
                        self._clausulas[rama] += 1
                for regla in reglas:
                    self._por_regla[regla] += 1
                    if len(self._ejemplos[regla]) < MAX_EJEMPLOS:
                        self._ejemplos[regla].append(
                            {
                                "fuente": self.fuente,
                                "reglas": reglas,
                                "text": row["text"][:EJEMPLO_CHARS],
                            }
                        )
                continue

            self._handle().write(
                json.dumps(
                    {k: row[k] for k in KEEP_COLS[self.fuente]}, ensure_ascii=False
                )
                + "\n"
            )
            self._similes["despues"]["turnos"] += len(respuestas)
            self._similes["despues"]["similes"] += sum(_conteo_similes(respuestas))
        yield inputs

    def unload(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"_stats-{self.fuente}.json").write_text(
            json.dumps(self._fragmento(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        super().unload()

    def _fragmento(self) -> dict[str, Any]:
        return {
            "fuente": self.fuente,
            "entrada": self._entrada,
            "descartadas": self._salida,
            "mantenidas": self._entrada - self._salida,
            "por_regla": dict(self._por_regla),
            "multiples_reglas": self._multi,
            "placeholder_por_clausula": dict(self._clausulas),
            "similes": {p: dict(c) for p, c in self._similes.items()},
            "recuperadas": {
                "rama_v1_disparaba_y_v2_no": self._recuperadas,
                "vivas_ganadas": self._recuperadas_vivas,
            },
            "ejemplos": {r: self._ejemplos[r] for r in self._ejemplos},
        }


# --------------------------------------------------------------------------
def purgar_fuente(fuente: str, rows: list[dict[str, Any]], cache: Path) -> None:
    """Una invocacion del pipeline por fuente.

    Converger los dos generadores en un Step (`[a, b] >> step`) cuelga el
    BatchManager de distilabel 1.5.3: con dos predecesores los lotes se bufferean
    por `seq_no` y el flush final no converge. Dos corridas encadenadas con el
    mismo DAG evitan eso y ademas bajan el pico de memoria a la mitad.
    """
    with Pipeline(name=f"kateto-purge-{fuente}", cache_dir=cache) as pipeline:
        load = LoadDataFromDicts(name=f"load_{fuente}", data=rows, batch_size=500)
        anotar = AnotarReglas(name="anotar_reglas", input_batch_size=500)
        sink = EscribirPurgado(
            name="escribir_purgado", fuente=fuente, input_batch_size=500
        )
        load >> anotar >> sink
    pipeline.run(use_cache=False)


def _ratio(similes: dict) -> float | None:
    turnos = similes.get("turnos", 0)
    if not turnos:
        return None
    return round(similes.get("similes", 0) / turnos, 4)


def _v1_clausulas() -> dict:
    """`placeholder_por_clausula` del reporte de R1, o {} si ya no esta."""
    if not V1_REPORT_PATH.exists():
        return {}
    return json.loads(V1_REPORT_PATH.read_text(encoding="utf-8")).get(
        "placeholder_por_clausula", {}
    )


def armar_reporte(fragmentos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from importlib.metadata import version

    shards = sorted(SHARD_DIR.glob(SHARD_GLOB))
    ejemplos: dict[str, list] = {r: [] for r in REGLAS}
    for fuente in SOURCES:
        for regla, xs in fragmentos.get(fuente, {}).get("ejemplos", {}).items():
            ejemplos.setdefault(regla, [])
            ejemplos[regla] = (ejemplos[regla] + xs)[:MAX_EJEMPLOS]

    v1 = _v1_clausulas()

    reporte: dict[str, Any] = {
        "backend": "distilabel",
        "distilabel_version": version("distilabel"),
        "python": sys.version.split()[0],
        "nota_plan_b": (
            "distilabel 1.5.3 SI instalo y sus Steps funcionan, asi que no hizo "
            "falta la version stdlib. Dos ajustes obligatorios: (1) no existe "
            "TextIO ni ningun lector/escritor de JSONL, la carga es con "
            "LoadDataFromDicts y la escritura la hace el Step sink; (2) los "
            "Steps conservan el numero de filas, asi que las reglas anotan una "
            "columna `_reglas` y el filtrado real ocurre en el sink; (3) "
            "converger dos generadores en un Step cuelga el BatchManager, por "
            "eso el pipeline corre una vez por fuente."
        ),
        "entradas": {
            "train_set": {
                "path": str(TRAIN_PATH.relative_to(REPO)),
                "lineas": fragmentos.get(TRAIN_KEY, {}).get("entrada", 0),
            },
            "shards": {
                "path": f"{SHARD_DIR.relative_to(REPO)}/{SHARD_GLOB}",
                "archivos": len(shards),
                "lineas": fragmentos.get(SHARDS_KEY, {}).get("entrada", 0),
            },
            "total_lineas": sum(f.get("entrada", 0) for f in fragmentos.values()),
        },
        "salidas": {
            "train_set": {
                "path": str(OUT_DIR / OUT_FILES[TRAIN_KEY]),
                "lineas_mantenidas": fragmentos.get(TRAIN_KEY, {}).get("mantenidas", 0),
                "lineas_descartadas": fragmentos.get(TRAIN_KEY, {}).get("descartadas", 0),
            },
            "shards": {
                "path": str(OUT_DIR / OUT_FILES[SHARDS_KEY]),
                "lineas_mantenidas": fragmentos.get(SHARDS_KEY, {}).get("mantenidas", 0),
                "lineas_descartadas": fragmentos.get(SHARDS_KEY, {}).get("descartadas", 0),
            },
            "total_mantenidas": sum(f.get("mantenidas", 0) for f in fragmentos.values()),
        },
        "por_regla": {
            regla: {
                "train_set": fragmentos.get(TRAIN_KEY, {})
                .get("por_regla", {})
                .get(regla, 0),
                "shards": fragmentos.get(SHARDS_KEY, {})
                .get("por_regla", {})
                .get(regla, 0),
                "total": sum(
                    fragmentos.get(f, {}).get("por_regla", {}).get(regla, 0)
                    for f in SOURCES
                ),
            }
            for regla in REGLAS
        },
        "descartadas_por_multiples_reglas": {
            "train_set": fragmentos.get(TRAIN_KEY, {}).get("multiples_reglas", 0),
            "shards": fragmentos.get(SHARDS_KEY, {}).get("multiples_reglas", 0),
            "total": sum(f.get("multiples_reglas", 0) for f in fragmentos.values()),
        },
        "placeholder_por_clausula": {
            f: fragmentos.get(f, {}).get("placeholder_por_clausula", {})
            for f in SOURCES
        },
        "recuperadas_vs_v1": {
            clave: {
                **{
                    f: fragmentos.get(f, {})
                    .get("recuperadas", {})
                    .get(clave, 0)
                    for f in SOURCES
                },
                "total": sum(
                    fragmentos.get(f, {}).get("recuperadas", {}).get(clave, 0)
                    for f in SOURCES
                ),
            }
            for clave in ("rama_v1_disparaba_y_v2_no", "vivas_ganadas")
        },
        "similes_por_turno_voz": {
            periodo: {
                f: _ratio(fragmentos.get(f, {}).get("similes", {}).get(periodo, {}))
                for f in SOURCES
            }
            for periodo in ("antes", "despues")
        },
        "ejemplos": ejemplos,
"notas": {
            "placeholder_por_clausula": (
                "La rama `token_repetido` de R1 estaba anulada por dos cosas a la "
                "vez: el marcador del formato pegado al token, y `(\\S)\\1{2,}$` "
                "que no exige letra. Una palabra degenerada al final de un turno "
                "NUNCA cumplia el patron, asi que la rama disparaba sobre "
                "puntuacion y markdown (`...`, `---`, `===`). Ahora limpia los "
                "marcadores antes de tokenizar y exige letra al final del token: "
                "caza `holaaaa` y `KATETOOOOO`, y `...`/`---`/`===` no cuentan "
                "mas. `recuperadas_vs_v1` cuenta el delta y `comparacion_v1` lo "
                "contrasta contra el reporte de R1."
            ),
            "lineas_de_entrada": (
                "Los archivos crecian mientras corria la purga (ola kateto-ola170 "
                "activa), asi que las entradas son mayores que las del brief."
            ),
        },
    }

    if v1:
        reporte["comparacion_v1"] = {
            "reporte": str(V1_REPORT_PATH.relative_to(REPO)),
            "token_repetido": {
                f: {
                    "v1": v1.get(f, {}).get("token_repetido", 0),
                    "v2": fragmentos.get(f, {})
                    .get("placeholder_por_clausula", {})
                    .get("token_repetido", 0),
                }
                for f in SOURCES
            },
            "KATETO": {
                f: {
                    "v1": v1.get(f, {}).get("KATETO", 0),
                    "v2": fragmentos.get(f, {})
                    .get("placeholder_por_clausula", {})
                    .get("KATETO", 0),
                }
                for f in SOURCES
            },
        }

    return reporte


def main() -> int:
    cache = Path(
        os.environ.get("PURGE_CACHE_DIR", "/run/media/chaos/secundario/purge-cache")
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fuente in SOURCES:
        (OUT_DIR / f"_stats-{fuente}.json").unlink(missing_ok=True)

    for fuente, rows in (
        (TRAIN_KEY, leer_train()),
        (SHARDS_KEY, leer_shards()),
    ):
        print(f"purgando {fuente}: {len(rows)} lineas", flush=True)
        purgar_fuente(fuente, rows, cache)
        del rows

    fragmentos = {
        f: json.loads((OUT_DIR / f"_stats-{f}.json").read_text(encoding="utf-8"))
        for f in SOURCES
    }
    REPORT_PATH.write_text(
        json.dumps(armar_reporte(fragmentos), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"reporte: {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
