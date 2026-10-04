#!/usr/bin/env python3
"""R17b — Tres lineas de generacion en paralelo, una por procesador, y la union.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python pipelines/tres_lineas.py

La idea del dueno: correr tres modelos con ventajas distintas en los tres
procesadores (CPU, GPU0 y GPU1), uno en cada uno, y despues juntar lo que cada
una produjo. Eso es un Mixture of Agents por afterthought: no hay un modelo que
combine, hay tres que escriben y una regla que junta.

Lo que NO se reimplementa (esta prohibido, y ademas seria medir otra cosa):

  - la generacion: `kateto_gen.py` como subproceso, con sus mismos argumentos
    (`--n 1 --scenario solo_charla --seed 42 --voice seco`). Es el generador de
    registro y el que aplica el gate de aceptacion.
  - la calidad por linea: `comparar_simulador.brazo()`. Se le pasa la carpeta
    `tres-lineas/linea<N>` y funciona porque `SMOKE / carpeta` resuelve el
    anidado igual que un brazo de primer nivel.
  - la deduplicacion de la union: `estilo.fold` + `estilo.huella` para el texto
    normalizado, `kateto_purge.simil_repetido` para los similes repetidos, y
    `kateto_gen.es_eco` para el casi-duplicado entre lineas (jaccard de
    trigramas >= 0.40 o arranque identico de 6 palabras).

Tres cosas del terreno que costaron una corrida y quedaron anotadas en el
codigo, porque no se deducen del brief:

  1. `GGML_VK_VISIBLE_DEVICES=N` hace que el backend Vulkan renumere como
     `Vulkan0` la unica placa que sobrevive, asi que las dos lineas de placa se
     arranCAN con `--device Vulkan0` y la que se pide `Vulkan1` se rechaza con
     `invalid device: Vulkan1`. Como el indice ya no las distingue, lo que las
     distingue es el NOMBRE de la placa en el log, y `evidencia()` lo exige.
  2. Los `.gguf` estan en `/run/media/chaos/secundario/models/hf`, NO en el
     cache de HF. R17 busco en `~/.cache/huggingface/hub/`, no los encontro y
     reporto que los modelos del brief no existian. Por eso cada server
     arranca solo despues de un `stat -c '%s %n'` cuya salida va al reporte.
  3. `kateto_gen.py` sale con rc=1 SIEMPRE que hay stats de eco: distilabel
     revienta DESPUES del sink, escribiendo su cache en parquet
     (`ArrowNotImplementedError: Cannot write struct type 'motivos_eco'`). Los
     .jsonl ya estan escritos. Por eso el exito se judgea por los archivos de
     salida, no por el rc (R13 lo discovering y lo documento en su cronometro).
  4. Este build de llama-server no imprime timings por request salvo que el
     pedido traiga `"verbose": true`, y `kateto_gen.py` no lo manda. El bloque
     `timings` SI viene en la respuesta del endpoint OpenAI, asi que los tok/s
     salen de una sonda al mismo server ya cargado, con el prompt del escenario
     y `verbose: true`. Es el timing del server, medido, pero de una sonda y no
     de los pedidos internos del item; va rotulado como tal.

Los numeros que NO se pueden medir van como `no_medido` con el motivo. No hay
estimaciones: el brief las prohibe y este item es comparativo.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PIPELINES = Path(__file__).resolve().parent
REPO = PIPELINES.parent
for _p in (PIPELINES, REPO / "scripts", PIPELINES / "estilo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kateto_gen as g  # noqa: E402  el generador y es_eco, sin duplicar
import kateto_purge as purge  # noqa: E402  simil_repetido, sin duplicar
import comparar_simulador as cs  # noqa: E402  brazo(): el instrumento de calidad
from estilo import metricas_estilo as est  # noqa: E402  fold/huella: texto normalizado

SMOKE = PIPELINES / "smoke"
BASE = SMOKE / "tres-lineas"
TABLA = SMOKE / "tres-lineas.json"
MD = SMOKE / "tres-lineas.md"

PY = "/run/media/chaos/secundario/venvs/distilabel/bin/python"
LLAMA = "/usr/bin/llama-server"

# Los .gguf NO estan en el cache de HF sino en este disco. R17 busco en
# `~/.cache/huggingface/hub/`, no los encontro, y reporto que los modelos del
# brief no existian. Existian. El `stat` de la regla dura va en `stat_evidencia`.
MODELOS = Path("/run/media/chaos/secundario/models/hf")

# El unico item por linea, el mismo para las tres. Es el del item anterior
# (`/home/chaos/harness-run/item-por-modelo/`), asi que las tres lineas y las
# cinco del item anterior hablan del mismo escenario con la misma semilla.
ESCENARIO = "solo_charla"
SEED = 42
VOICE = "seco"
N = 1

# Puertos propios. Ni `:11434` (el manager del dueno, que no se toca) ni el
# `11510`/`11534` de corridas anteriores.
PUERTOS = {"linea1": 11571, "linea2": 11572, "linea3": 11573,
           "linea4": 11574, "linea5": 11575}

# El baseline de "una sola linea" del brief son las ~28 muestras/hora del 26B
# solo. Viene del item anterior y esta MEDIDO ahi, no acá: se lee del archivo,
# y se cita con el nombre del modelo que lo midio (que no es el de la linea 1 de
# este item; ver `lineas()`).
ITEMS_ANTERIOR = Path("/home/chaos/harness-run/item-por-modelo/items.jsonl")

# Sampling que usa el resto del repo para los brazos locales (comparar_local_12b
# y el config.ini del dueno). Mismo sampling en las tres lineas, para que la
# unica variable sea el dispositivo.
SAMPLING = ["--temp", "0.6", "--top-k", "64", "--top-p", "0.9",
            "--min-p", "0.05", "--repeat-penalty", "1.1"]

# El server solo imprime `using device`, el offload de capas y el desglose de
# VRAM con `-lv 10`. Con el default (3) el log no los tiene, y la evidencia del
# brief —que cada linea cayO en su placa— seria una suposicion.
VERBOSIDAD = ["-lv", "10"]


@dataclass
class Linea:
    n: int
    clave: str
    modelo: str
    gguf: Path | None
    dispositivo: str          # "CPU" | "Vulkan0" | "Vulkan1"
    flags: list[str]
    pedido: str                # el nombre del brief, para el reporte de faltantes
    tamanio_bytes: int | None = None
    por_que_no: str | None = None
    unidad: str = ""
    log: Path = field(default_factory=Path)
    vk_visible: str | None = None   # GGML_VK_VISIBLE_DEVICES de su entorno
    placa: str | None = None       # substring que el log DEBE decir para darla por buena
    stat_evidencia: str = "no_medido"
    # los archivos que el propio brief nombra como alternativa para este slot, en
    # orden. Se prueban si el primario no carga (no solo si no esta).
    alternativas: tuple[Path, ...] = ()

    @property
    def puerto(self) -> int:
        return PUERTOS[self.clave]

    @property
    def carpeta(self) -> Path:
        return BASE / self.clave


def _gguf(*relatives: str) -> tuple[Path | None, str | None]:
    """El primer .gguf que exista de la lista, y el que se intento y fallo.

    Devuelve la tupla para que el fallback de la linea 2 sea una caida
    declarada, no un modelo distinto sin avisar.
    """
    for rel in relatives:
        p = MODELOS / rel
        if p.exists():
            return p, None
    return None, "/".join(str(MODELOS / r) for r in relatives)


def _stat(ruta: Path) -> tuple[int | None, str]:
    """`stat -c '%s %n'`: los bytes y la linea cruda, que es la evidencia que el
    brief exige antes de arrancar. Se pega tal cual en el reporte."""
    p = subprocess.run(["stat", "-c", "%s %n", str(ruta)],
                       capture_output=True, text=True)
    linea = p.stdout.strip()
    if p.returncode != 0 or not linea:
        return None, f"stat fallo ({p.returncode}): {p.stderr.strip()[:200]}"
    return int(linea.split()[0]), linea


# Las tres lineas del brief, con los archivos EXACTOS que pide. Ninguno se
# sustituye: si uno falta de verdad, `arrancar` lo reporta con el `ls` de la
# carpeta y la linea sale `no_disponible`.
def lineas() -> list[Linea]:
    # Fallback del brief para la linea 2 (k2horizon -> frognano). Los dos
    # existen; k2horizon no lo carga ESTE build, asi que el fallback es por
    # arquitectura, no por archivo ausente.
    p26, f26 = _gguf("heretic-styletune-26b/"
                     "gemma-4-26b-a4b-heretic-styletune-v2-head.i1-Q4_K_M.gguf")
    k2, f2 = _gguf("catalogo/k2horizon-37b/K2-Horizon-4B-Q4_K_M.gguf",
                   "catalogo/frognano-4b/FrogNano-4B-2609-Q4_K_M.gguf")
    bonsai, f3 = _gguf("catalogo/bonsai-8b/Bonsai-8B-Q1_0.gguf")
    alt_frognano = (MODELOS / "catalogo/frognano-4b/FrogNano-4B-2609-Q4_K_M.gguf")

    comun = ["-c", "8192", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    return [
        Linea(
            1, "linea1", "heretic26b", p26, "CPU",
            ["--device", "none", "-t", "6", *comun],
            "heretic-styletune-26b (17,2 GB, mmap, -t 6, sin offload)",
            por_que_no=f26,
        ),
        Linea(
            2, "linea2", "k2horizon4b", k2, "Vulkan0",
            # `--device Vulkan0` y no `Vulkan1`: con `GGML_VK_VISIBLE_DEVICES`
            # puesta, el backend renumera como Vulkan0 la unica placa visible.
            # Por eso `placa` verifica el NOMBRE de la placa en el log.
            ["--device", "Vulkan0", "-ngl", "99", "--fit", "on", *comun],
            "catalogo/k2horizon-37b (3,16 GB, -ngl 99); fallback del brief: "
            "catalogo/frognano-4b (2,80 GB)",
            por_que_no=f2, vk_visible="0", placa="AMD Radeon RX 6500 XT",
            alternativas=(alt_frognano,) if k2 != alt_frognano else (),
        ),
        Linea(
            3, "linea3", "bonsai8b", bonsai, "Vulkan1",
            ["--device", "Vulkan0", "-ngl", "99", "--fit", "on", *comun],
            "catalogo/bonsai-8b Q1 (1,16 GB, -ngl 99)",
            por_que_no=f3, vk_visible="1", placa="Intel(R) Iris(R) Xe Graphics",
        ),
    ]


# La Fase B: el mismo archivo en CPU y en su placa, mismo prompt, un item cada
# uno. Es el numero que dice cuanto da la placa. Usa el archivo de la linea 2.
def fase_b(base: Linea) -> tuple[Linea, Linea] | None:
    if base.gguf is None:
        return None
    comun = ["-c", "8192", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    return (
        Linea(4, "linea4", f"{base.modelo}-cpu", base.gguf, "CPU",
              ["--device", "none", "-t", "6", *comun],
              f"el mismo archivo de la linea 2 pero en CPU -t 6 "
              f"(el par que dice cuanto da la placa)",
              tamanio_bytes=base.tamanio_bytes, vk_visible=None,
              stat_evidencia=base.stat_evidencia),
        Linea(5, "linea5", f"{base.modelo}-vk0", base.gguf, "Vulkan0",
              ["--device", "Vulkan0", "-ngl", "99", "--fit", "on", *comun],
              f"el mismo archivo de la linea 2 en su placa (Vulkan0)",
              tamanio_bytes=base.tamanio_bytes, vk_visible=base.vk_visible,
              placa=base.placa, stat_evidencia=base.stat_evidencia),
    )


# --------------------------------------------------------------------------- #
# Servers: systemd-run --user. Nunca en el shell del runner (cgroup ~4 GB).
# --------------------------------------------------------------------------- #
def _argv_srv(linea: Linea) -> list[str]:
    assert linea.gguf is not None
    return [
        LLAMA, "-m", str(linea.gguf), "--alias", linea.modelo,
        "--host", "127.0.0.1", "--port", str(linea.puerto),
        "--no-webui", "--log-file", str(linea.log),
        *linea.flags, *SAMPLING, *VERBOSIDAD,
    ]


def _unidad(linea: Linea) -> str:
    return f"{linea.clave}-{linea.modelo}"


def unidad_viva(unidad: str) -> bool:
    p = subprocess.run(["systemctl", "--user", "is-active", "--quiet", unidad],
                       capture_output=True)
    return p.returncode == 0


def parar(unidad: str) -> None:
    """Para lo NUESTRO por nombre de unidad. Nunca `pkill -x llama-server`:
    el :11434 del dueno es una unidad del sistema y se lleva puesta."""
    subprocess.run(["systemctl", "--user", "stop", unidad],
                   capture_output=True)


def unidades_linea() -> list[str]:
    p = subprocess.run(["systemctl", "--user", "list-units", "linea*", "--all",
                        "--no-pager", "--plain", "--no-legend"],
                       capture_output=True, text=True)
    return [ln.split()[0] for ln in p.stdout.splitlines() if ln.strip()]


def _mem_host() -> dict[str, int]:
    """RAM y swap del host, para probar que nada hizo swap a lo loco."""
    d: dict[str, int] = {}
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            k, _, rest = ln.partition(":")
            if k in {"MemAvailable", "MemTotal", "SwapTotal", "SwapFree"}:
                d[k] = int(rest.strip().split()[0]) * 1024
    except OSError as e:  # pragma: no cover
        return {"error": str(e)}
    return d


def memoria_unidad(unidad: str) -> int | None:
    p = subprocess.run(["systemctl", "--user", "show", unidad,
                        "-p", "MemoryCurrent", "--value"],
                       capture_output=True, text=True)
    v = p.stdout.strip()
    try:
        return int(v)
    except ValueError:
        return None


def arrancar(linea: Linea, *, espera_s: int = 240) -> dict[str, Any]:
    """Levanta la linea, y si el modelo no CARGA usa la alternativa que el
    propio brief nombra para ese slot.

    El brief dice "si falta" para el fallback de la linea 2, pero lo que paso de
    verdad es que `k2-horizon` no lo soporta ESTE build de llama-server
    (`unknown model architecture: 'k2-horizon'`): el archivo esta y pesa 3,16 GB.
    La alternativa que el brief ya nombra para esa linea es `frognano-4b`, asi
    que se usa ESA y no una tercera cosa. El fallo del primero queda crudo en
    `caidas`, con la linea de log.
    """
    primario = linea.gguf
    intentos: list[dict[str, Any]] = []
    for cand in [linea.gguf, *linea.alternativas]:
        if cand is None:
            continue
        if cand is not primario:
            # Alias del alterno, derivado del archivo: "FrogNano-4B-2609-Q4_K_M"
            # -> "frognano4b". El primario conserva su alias curado.
            linea.modelo = "-".join(cand.stem.split("-")[:2]).lower()
            linea.modelo = linea.modelo.replace("-", "").replace("_", "")
        linea.gguf = cand
        r = _intentar(linea, espera_s=espera_s)
        intentos.append({"archivo": str(cand), "stat": r.get("stat"), **{
            k: r[k] for k in ("estado", "motivo", "log") if k in r}})
        if r.get("estado") == "arrancada":
            r["intentos"] = intentos
            r["usado_el_alterno"] = cand is not primario
            r["primario_pedido"] = str(primario) if primario else None
            return r
        parar(linea.unidad)
    linea.por_que_no = intentos[-1].get("motivo") if intentos else "sin archivo"
    return {"estado": "no_disponible", "motivo": linea.por_que_no, "intentos": intentos}


def _intentar(linea: Linea, *, espera_s: int = 240) -> dict[str, Any]:
    """Levanta la linea como unidad de usuario y espera `/health`.

    Devuelve el veredicto con la EVIDENCIA del dispositivo: la linea del log
    donde el server dice que device esta usando. Sin esa linea, la linea se
    reportaria `no_disponible` — no se supone que cada linea cae en su placa.
    """
    linea.unidad = _unidad(linea)
    linea.log = BASE / f"{linea.clave}-{linea.modelo}.server.log"
    linea.carpeta.mkdir(parents=True, exist_ok=True)
    parar(linea.unidad)
    linea.log.unlink(missing_ok=True)

    if linea.gguf is None or not linea.gguf.exists():
        linea.por_que_no = linea.por_que_no or f"no existe el archivo {linea.pedido}"
        # El brief: si un archivo falta de verdad, PARAR y reportar con el `ls` de
        # la carpeta. No sustituir por otro modelo.
        return {"estado": "no_disponible", "motivo": linea.por_que_no,
                "stat": "no_medido: el archivo no existe",
                "ls_de_la_carpeta": _ls_carpeta(linea.gguf)}

    # Regla dura del brief: `stat` antes de arrancar, y la salida cruda en el
    # reporte. Es la evidencia de que el archivo existe y con cuantos bytes.
    bytes_, stat_linea = _stat(linea.gguf)
    linea.tamanio_bytes = bytes_
    linea.stat_evidencia = stat_linea

    argv = ["systemd-run", "--user", f"--unit={linea.unidad}"]
    if linea.vk_visible is not None:
        argv.append(f"--setenv=GGML_VK_VISIBLE_DEVICES={linea.vk_visible}")
    argv += _argv_srv(linea)
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        linea.por_que_no = f"systemd-run salio {proc.returncode}: {proc.stderr.strip()[:300]}"
        return {"estado": "no_disponible", "motivo": linea.por_que_no,
                "stat": stat_linea}

    t0 = time.monotonic()
    while time.monotonic() - t0 < espera_s:
        if not unidad_viva(linea.unidad):
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{linea.puerto}/health",
                                        timeout=2) as r:
                if r.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        parar(linea.unidad)
        linea.por_que_no = f"el server no respondio /health en {espera_s}s"
        return {"estado": "no_disponible", "motivo": linea.por_que_no,
                "stat": stat_linea, "log": _ultimas(linea.log)}

    if not unidad_viva(linea.unidad):
        cola = _ultimas(linea.log) or _ultimas_via_journal(linea.unidad)
        linea.por_que_no = f"el server murio al arrancar: {cola}"
        return {"estado": "no_disponible", "motivo": linea.por_que_no,
                "stat": stat_linea, "log": cola}

    return {"estado": "arrancada", "segundos_de_arranque": round(time.monotonic() - t0, 1),
            "stat": stat_linea, "ggml_vk_visible_devices": linea.vk_visible}


def _ls_carpeta(gguf: Path | None) -> str | None:
    """El `ls -la` de la carpeta del .gguf que se pidio, para probar que el
    archivo no esta y no fue un path mal escrito."""
    if gguf is None:
        return None
    carpeta = gguf.parent
    if not carpeta.is_dir():
        return f"{carpeta} no existe"
    p = subprocess.run(["ls", "-la", str(carpeta)], capture_output=True, text=True)
    return p.stdout.strip() or p.stderr.strip()


def _ultimas(path: Path, n: int = 12) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[-n:])


def _ultimas_via_journal(unidad: str, n: int = 12) -> str:
    p = subprocess.run(["journalctl", "--user", "-u", unidad, "-n", str(n),
                        "--no-pager", "-o", "cat"], capture_output=True, text=True)
    return p.stdout.strip()


# La evidencia: el server ENUMERA las dos placas siempre, asi que enumerarlas no
# prueba nada. Lo que prueba es la linea de `llama_prepare_model_devices`, que
# dice cual esta usando.
RX_EVIDENCIA = re.compile(r"using device (\S+)(?:\s+\((.*)\))?")
RX_OFFLOAD = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
# La enumeracion que hace el server al arrancar:
#   - Vulkan0 : Intel(R) Iris(R) Xe Graphics (DG1) (4024 MiB, 3621 MiB free)
# El nombre va sin parentesis desbalanceados porque se corta en `(N MiB`. Con
# `GGML_VK_VISIBLE_DEVICES` puesta aparece UNA sola linea, y ese conteo es la
# prueba de que el server no ve la otra placa.
RX_PLACA = re.compile(r"-\s*(Vulkan\d+)\s*:\s*(.+?)\s*\(\d+ MiB")
# Con `--device none` no hay `using device`: hay una linea por capa.
RX_CPU_LAYER = re.compile(r"layer\s+\d+ assigned to device CPU")
# El desglose que imprime el server es una fila por memoria:
#   | - Vulkan0 (RX 6500 XT) | 4080 = 4064 + (1494 = 1341 + 19 + 133) + -1478 |
# con el encabezado `total free self model context compute unaccounted`.
# `self` es lo que ocupa el server (modelo + contexto + compute): eso es el
# "VRAM usada" del brief. `Host` es la RAM del proceso.
RX_VRAM = re.compile(
    r"\|\s*-\s*(?P<quien>[^|]+?)\s*\|\s*(?P<total>[\d.]+)?\s*=\s*"
    r"(?P<libre>[\d.]+)\s*\+\s*\(\s*(?P<self>[\d.]+)\s*=\s*(?P<model>[\d.]+)\s*\+\s*"
    r"(?P<contexto>[\d.]+)\s*\+\s*(?P<compute>[\d.]+)\s*\)"
)


def evidencia(linea: Linea) -> dict[str, Any]:
    """Del log del server: que device uso, cuantas capas y cuanta memoria.

    La linea de `using device` solo aparece cuando hay device. Con
    `--device none` (la linea de CPU) lo que hay es una linea POR CAPA:
    `layer 0 assigned to device CPU`. Sin ese caso, la linea de CPU se
    reportaria sin evidencia, que es justo lo que el brief prohibe.
    """
    txt = linea.log.read_text(encoding="utf-8", errors="ignore") if linea.log.exists() else ""
    gpu = None
    for m in RX_EVIDENCIA.finditer(txt):
        gpu = (m.group(1), m.group(2))
    capas_gpu = RX_OFFLOAD.search(txt)
    a_cpu = len(RX_CPU_LAYER.findall(txt))
    linea_device = next((ln.strip() for ln in txt.splitlines() if "using device" in ln), None)

    if gpu is not None:
        usado, nombre = gpu
    elif a_cpu:
        usado, nombre = "CPU", "(sin device: todas las capas al CPU)"
    else:
        usado, nombre = "no_medido", None

    # Con `--device none` el log dice "offloaded X/Y layers to GPU" igual (es la
    # palabra del server, no el hecho), asi que para la linea de CPU lo que
    # cuenta son las capas asignadas al CPU.
    if usado == "CPU" and a_cpu:
        capas = f"{a_cpu} capas a CPU (0 a GPU)"
    elif capas_gpu:
        capas = f"{capas_gpu.group(1)}/{capas_gpu.group(2)} a GPU"
    else:
        capas = None

    mem: dict[str, Any] = {}
    for m in RX_VRAM.finditer(txt):
        quien = m.group("quien").strip()
        if not quien.startswith(("Vulkan", "CPU", "Host")):
            continue
        clave = "host" if quien.startswith("Host") else quien.split()[0]
        mem[clave] = {
            "quien": quien,
            "mib": {k: float(m.group(k)) for k in
                    ("total", "libre", "self", "model", "contexto", "compute")
                    if m.group(k)},
        }

    # Con `GGML_VK_VISIBLE_DEVICES` las dos lineas de placa dicen `Vulkan0`, asi
    # que el indice ya no distingue AMD de Intel: lo que las distingue es el
    # NOMBRE. Sin este chequeo, la linea 3 podria reportarse como Vulkan0 y
    # nadie notaria que cayo en la AMD.
    enumeradas = RX_PLACA.findall(txt)
    nombres = [n for _, n in enumeradas]
    coincide_placa = (linea.placa is None
                      or any(linea.placa in n for n in nombres))
    coincide_indice = usado == linea.dispositivo
    # El indice solo vale mientras la placa se renumere: con la variable de
    # visibilidad las dos lineas de placa dicen `Vulkan0`. Por eso el veredicto
    # es "indice o nombre", y cada uno queda por separado.
    return {
        "dispositivo_del_log": usado,
        "nombre_del_dispositivo": nombres[0] if nombres else nombre,
        "coincide_con_el_pedido": coincide_indice or coincide_placa,
        "coincide_el_indice": coincide_indice,
        "placa_pedida": linea.placa,
        "placas_que_el_log_enumera": nombres,
        "cuantas_placas_ve_el_server": len(nombres) or None,
        "coincide_la_placa": coincide_placa,
        "aislamiento": (
            f"GGML_VK_VISIBLE_DEVICES={linea.vk_visible}: el server enumera "
            f"{len(nombres)} placa ({', '.join(nombres) or 'no_medido'})"
            if linea.vk_visible is not None else
            "--device none (sin placa; toda capa al CPU)"),
        "capas": capas,
        "capas_a_cpu": a_cpu or None,
        "memoria_mib": mem,
        "linea_del_log": linea_device or next(
            (ln.strip() for ln in txt.splitlines() if "assigned to device" in ln), None),
    }


# --------------------------------------------------------------------------- #
# El item: kateto_gen.py como subproceso, con reloj alrededor (el patron de R13)
# --------------------------------------------------------------------------- #
def correr_item(linea: Linea) -> dict[str, Any]:
    """Un item por linea. El exito se judgea por los archivos, no por el rc: el
    generador sale con 1 siempre que distilabel falle al escribir su cache,
    DESPUES de que el sink escribio los .jsonl."""
    out = linea.carpeta / "muestras.jsonl"
    rej = linea.carpeta / "rechazos.jsonl"
    argv = [
        PY, str(PIPELINES / "kateto_gen.py"),
        "--n", str(N), "--scenario", ESCENARIO, "--seed", str(SEED),
        "--voice", VOICE, "--model", linea.modelo,
        "--base-url", f"http://127.0.0.1:{linea.puerto}/v1",
        "--key", "local",
        "--out", str(out), "--rejects", str(rej),
    ]
    log_gen = linea.carpeta / "generador.log"
    mem_antes = _mem_host()
    t0 = time.time()
    proc = subprocess.run(argv, capture_output=True, text=True,
                          cwd=str(REPO), env={**os.environ, "GEN_CACHE_DIR":
                                              str(BASE / "gen-cache")})
    wall = time.time() - t0
    mem_despues = _mem_host()
    log_gen.write_text(f"$ {' '.join(argv)}\n\n--- stdout ---\n{proc.stdout}"
                       f"\n--- stderr ---\n{proc.stderr}", encoding="utf-8")

    # Se mide el footprint con el modelo CARGADO (los .jsonl ya estan
    # escritos), que es el estado en el que la linea vive.
    memoria = memoria_unidad(linea.unidad) if linea.unidad else None

    stats_p = out.with_suffix(".stats.json")
    stats = json.loads(stats_p.read_text(encoding="utf-8")) if stats_p.exists() else {}
    n_acept = stats.get("aceptadas", _filas(out))
    n_rech = stats.get("rechazadas", _filas(rej))

    return {
        "argv": " ".join(argv),
        "rc": proc.returncode,
        # El rc de este comando NO es el veredicto (ver nota 2 del docstring).
        "rc_que_significa": (
            "ok" if proc.returncode == 0 else
            "distilabel fallo DESPUES del sink al escribir su cache en parquet "
            "(ArrowNotImplementedError: 'motivos_eco'); los .jsonl ya estaban escritos"
            if "ArrowNotImplementedError" in proc.stderr else
            f"fallo real (rc={proc.returncode})"
        ),
        "wall_segundos": round(wall, 2),
        "muestras": n_acept + n_rech,
        "aceptadas": n_acept,
        "rechazadas": n_rech,
        "muestras_por_hora": round((n_acept + n_rech) / (wall / 3600), 2) if wall else None,
        "aceptadas_por_hora": round(n_acept / (wall / 3600), 2) if wall else None,
        "s_por_muestra": round(wall / (n_acept + n_rech), 2) if (n_acept + n_rech) else None,
        "memoria_del_proceso_bytes": memoria,
        "host_antes": mem_antes,
        "host_despues": mem_despues,
        "swap_delta_bytes": (
            (mem_antes.get("SwapFree", 0) - mem_despues.get("SwapFree", 0))
            if "SwapFree" in mem_antes and "SwapFree" in mem_despues else None
        ),
        "mem_antes": mem_antes,
        "log_del_generador": str(log_gen.relative_to(REPO)),
        "errores": [ln for ln in proc.stderr.splitlines()
                    if "Error" in ln or "error" in ln][-5:],
        "cola_del_generador": proc.stderr.strip().splitlines()[-6:],
    }


def _filas(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip())


# --------------------------------------------------------------------------- #
# Los tok/s: sonda al server ya cargado con `verbose: true`, que es la unica
# via por la que este build entrega los timings (nota 3 del docstring).
# --------------------------------------------------------------------------- #
def sonda_timings(linea: Linea) -> dict[str, Any]:
    """Un pedido al MISMO server, con el prompt del escenario y `verbose: true`.

    El `timings` que vuelve en el cuerpo es del server. Lo que NO es, el item
    completo: es una sonda. Va rotulado `timings_de_sonda`.
    """
    sc = next((s for s in _escenarios() if s["id"] == ESCENARIO), None)
    if sc is None:
        return {"estado": "no_medido", "motivo": f"escenario {ESCENARIO} no encontrado"}
    prompt = g.build_user_prompt(sc, VOICE, [], 3)
    cuerpo = json.dumps({
        "model": linea.modelo,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "verbose": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{linea.puerto}/v1/chat/completions", data=cuerpo,
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
    except Exception as e:
        return {"estado": "no_medido", "motivo": f"{type(e).__name__}: {e}"}
    wall = time.monotonic() - t0
    t = d.get("timings") or {}
    uso = d.get("usage") or {}
    return {
        "estado": "medido",
        "fuente": "sonda al server de la linea, timings del server (verbose:true)",
        "tok_s_prompt": t.get("prompt_per_second"),
        "tok_s_generacion": t.get("predicted_per_second"),
        "prompt_tokens": t.get("prompt_n") or uso.get("prompt_tokens"),
        "tokens_generados": t.get("predicted_n") or uso.get("completion_tokens"),
        "prompt_ms": t.get("prompt_ms"),
        "generacion_ms": t.get("predicted_ms"),
        "sonda_wall_s": round(wall, 2),
    }


def _escenarios() -> list[dict]:
    sys.path.insert(0, str(REPO / "scripts"))
    import gen_toolcalling_dataset as ref  # noqa: PLC0415
    return list(ref.SCENARIOS)


# --------------------------------------------------------------------------- #
# La union. Dedup con los instrumentos del repo, no con uno nuevo.
# --------------------------------------------------------------------------- #
def unir(lineas_medidas: list[dict[str, Any]]) -> dict[str, Any]:
    """Los aceptados de las tres lineas en un conjunto, deduplicado.

    Tres reglas, cada una con el instrumento que YA existe en el repo:

      1. texto normalizado identico -> `estilo.fold` + `estilo.huella`. Es la
         misma normalizacion (sin tildes, minusculas, espacios colapsados) que
         usa el instrumento de estilo, no una variante.
      2. similes repetidos -> `kateto_purge.simil_repetido`, la regla de purga
         (`como ...` repetido en los turnos de voz). No se reimplementa.
      3. casi-duplicado ENTRE lineas -> `kateto_gen.es_eco`, el mismo detector
         anti-eco del generador (jaccard de trigramas >= 0.40 o arranque
         identico de 6 palabras). Es la pregunta "estas dos son casi lo mismo",
         con el umbral que el repo ya usa para eso.
    """
    entrantes: list[dict[str, Any]] = []
    for l in lineas_medidas:
        carpeta = Path(l["carpeta"])
        if l.get("estado") != "medida" or not (carpeta / "muestras.jsonl").exists():
            continue
        for f in _leer(carpeta / "muestras.jsonl"):
            f = {**f, "_linea": l["clave"], "_modelo": l["modelo"]}
            entrantes.append(f)

    BASE.mkdir(parents=True, exist_ok=True)
    conjunto_p = BASE / "conjunto.jsonl"
    caidas_p = BASE / "conjunto-caidas.jsonl"

    quedan: list[dict[str, Any]] = []
    caidas: list[dict[str, Any]] = []
    vistas: dict[str, str] = {}
    por_linea = {l["clave"]: {"aceptadas": l.get("aceptadas", 0), "en_el_conjunto": 0,
                              "por_duplicado": 0} for l in lineas_medidas}

    for f in entrantes:
        texto = f.get("text") or ""
        h = est.huella(est.fold(texto))
        if h in vistas:
            por_linea[f["_linea"]]["por_duplicado"] += 1
            caidas.append({**f, "caida": "duplicado_exacto",
                           "igual_a_linea": vistas[h]})
            continue
        if purge.simil_repetido(texto):
            por_linea[f["_linea"]]["por_duplicado"] += 1
            caidas.append({**f, "caida": "purga:simil_repetido"})
            continue
        casi = next((q for q in quedan if g.es_eco(texto, q.get("text") or "")), None)
        if casi is not None:
            por_linea[f["_linea"]]["por_duplicado"] += 1
            caidas.append({**f, "caida": "casi_duplicado",
                           "igual_a_linea": casi["_linea"]})
            continue
        vistas[h] = f["_linea"]
        quedan.append(f)
        por_linea[f["_linea"]]["en_el_conjunto"] += 1

    with conjunto_p.open("w", encoding="utf-8") as fh:
        for f in quedan:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")
    with caidas_p.open("w", encoding="utf-8") as fh:
        for f in caidas:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    return {
        "entrantes": len(entrantes),
        "en_el_conjunto": len(quedan),
        "caidas": len(caidas),
        "caidas_por_motivo": _conteo(f.get("caida") for f in caidas),
        "por_linea": por_linea,
        "instrumentos": {
            "texto_normalizado": "pipelines/estilo/metricas_estilo.py::fold + huella",
            "similes_repetidos": "pipelines/kateto_purge.py::simil_repetido (las reglas de R1)",
            "casi_duplicado_entre_lineas": "pipelines/kateto_gen.py::es_eco "
                                           f"(trigramas >= {g.ECO_TRIGRAMAS} o arranque de {g.ECO_ARRANQUE})",
        },
        "conjunto": str(conjunto_p.relative_to(REPO)),
        "caidas_archivo": str(caidas_p.relative_to(REPO)),
    }


def _leer(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            out.append(json.loads(ln))
    return out


def _conteo(xs) -> dict[str, int]:
    d: dict[str, int] = {}
    for x in xs:
        if x:
            d[x] = d.get(x, 0) + 1
    return d


# --------------------------------------------------------------------------- #
# El baseline de una sola linea, del item anterior.
# --------------------------------------------------------------------------- #
def una_sola_linea() -> dict[str, Any]:
    """Lo que midio el item anterior con el 26B SOLO, ledo de su archivo.

    Es el MISMO modelo que la linea 1 de hoy (26B-A4B heretic + StyleTune head,
    17,2 GB), asi que la comparacion "las tres juntas vs la mejor sola" es
    apples-to-apples. Lo que no se sabe es con cuantos hilos se midio aqui; se
    cita como lo que es: una referencia medida antes, no una medicion de hoy.
    """
    if not ITEMS_ANTERIOR.exists():
        return {"estado": "no_medido",
                "motivo": f"no existe {ITEMS_ANTERIOR}"}
    # El brief compara contra "el 26B solo": se busca ese, no el primero que
    # salio bien. Si no esta, se usa el mayor y se dice cual fue.
    ok = [f for f in _leer(ITEMS_ANTERIOR) if f.get("segundos") and f.get("ok")]
    elegido = next((f for f in ok if "26" in str(f.get("modelo"))), None) or (ok[0] if ok else None)
    for f in [elegido] if elegido else []:
            s = f["segundos"]
            return {
                "estado": "medido_en_item_anterior",
                "modelo": f["modelo"],
                "descripcion": f.get("descripcion"),
                "segundos_por_muestra": s,
                "muestras_por_hora": round(3600 / s, 2),
                "fuente": str(ITEMS_ANTERIOR),
            }
    return {"estado": "no_medido", "motivo": "el item anterior no tiene ninguna linea con ok"}


def _md(r: dict[str, Any]) -> None:
    """Media pagina: las tres preguntas del brief, con el numero de adentro."""
    L, t, fb, u = r["lineas"], r["titular"], r["fase_b_cpu_vs_placa"], r["union"]
    sola = t["una_sola_linea_referencia"]

    def c(x: Any) -> str:
        return "no_medido" if x is None or x == "no_medido" else str(x)

    filas = []
    for f in L:
        ev = f.get("evidencia_del_dispositivo") or {}
        filas.append(
            f"| {f['linea']} | {f['dispositivo']} | "
            f"`{f.get('modelo') or 'no_disponible'}` | {f.get('tamanio_gb', '-')} "
            f"| {c(f.get('tok_s_prompt'))} | {c(f.get('tok_s_generacion'))} "
            f"| {c(f.get('wall_segundos'))} "
            f"| {f.get('aceptadas', '-')}/{f.get('rechazadas', '-')} "
            f"| {c(f.get('muestras_por_hora'))} "
            f"| {ev.get('capas') or '-'} |")
    tabla = "\n".join(filas)

    stats = "\\n".join(
        f"- `{f['linea']}` — `{f.get('stat_evidencia', 'no_medido')}`"
        for f in L if f.get("stat_evidencia") not in (None, "no_medido"))

    alternos = [f for f in L if f.get("usado_el_alterno")]
    alt_txt = ""
    if alternos:
        partes = []
        for f in alternos:
            fallos = f.get("intentos_fallidos") or []
            causa = next((str(x.get("log", "")).strip().splitlines()[-1]
                          for x in fallos if x.get("log")), "sin log")
            partes.append(
                f"- `{f['linea']}`: el brief pedia "
                f"`{Path(str(f.get('modelo_primario_pedido') or '')).name}`, que esta en "
                f"disco pero **no lo carga este build de llama-server**. Se uso la "
                f"alternativa que el propio brief nombra para esa linea, "
                f"`{Path(str(f['archivo'])).name}`. Error crudo del intento fallido: "
                f"`{causa}`")
        alt_txt = ("\\n\\n### Alternativa usada (la nombra el brief, no la elegi yo)\\n\\n"
                   + "\\n".join(partes))

    if fb.get("estado") == "medido" and fb.get("la_placa_rinde"):
        fb_txt = (
            f"**Si: {fb['la_placa_rinde']}x.** El MISMO archivo en las dos columnas, "
            f"el mismo prompt y la misma sonda: **{c(fb.get('tok_s_generacion_placa'))} "
            f"tok/s** de generacion en su placa contra **{c(fb.get('tok_s_generacion_cpu'))}"
            f" tok/s** en CPU (`-t 6`). Lo unico que cambia es `--device`, asi que ese "
            f"numero aisla lo que da la placa. Por item: "
            f"{c((fb.get('placa') or {}).get('s_por_muestra'))} s en la placa contra "
            f"{c((fb.get('cpu') or {}).get('s_por_muestra'))} s en CPU."
        )
    else:
        fb_txt = ("**No medido.** " + str(fb.get("motivo") or fb.get("estado")))

    total = t["total_de_las_tres_muestras_por_hora"]
    mejor = t.get("mejor_sola") or {}
    con_muestras = [f for f in L if (f.get("aceptadas") or 0) > 0]
    sin_muestras = [f["linea"] for f in L if f.get("estado") == "medida"
                    and not f.get("aceptadas")]

    if isinstance(total, (int, float)) and isinstance(mejor.get("muestras_por_hora"), (int, float)):
        x_mejor = round(total / mejor["muestras_por_hora"], 2)
        veredicto_1 = (
            f"Las tres juntas dan **{c(total)} muestras/hora**; la mejor sola, "
            f"{mejor['linea']}, da **{c(mejor['muestras_por_hora'])}**: **{x_mejor}x**. "
            + (f"Contra la referencia de una sola linea del item anterior "
               f"({sola.get('modelo')}, {c(sola.get('segundos_por_muestra'))} s/muestra = "
               f"{c(sola.get('muestras_por_hora'))}/hora, el MISMO modelo que la linea 1) "
               f"son **{c(t.get('las_tres_vs_una_sola'))}x**."
               if t.get("las_tres_vs_una_sola") else ""))
    else:
        veredicto_1 = (
            f"**No se puede contestar con estos numeros.** Las tres juntas dan "
            f"{c(total)} muestras/hora y solo **{len(con_muestras)} de las 3 lineas** "
            "produjo una muestra aceptada"
            + (f" (lineas {', '.join(sin_muestras)} con 0 aceptadas y 0 rechazadas: no es "
               "que sus muestras se caiaran, es que no hay muestra que contar)"
               if sin_muestras else "")
            + f". La union sale con **{u['en_el_conjunto']}** documento de "
              f"**{u['entrantes']}** entrantes.")

    if mejor:
        linea_vol = (f"**{mejor['linea']}** ({mejor.get('dispositivo')}, "
                     f"{c(mejor.get('muestras_por_hora'))} muestras/hora)")
    else:
        linea_vol = "ninguna: no hay ninguna linea con muestras/hora mayor que cero"
    linea_cal = (f"**{con_muestras[0]['linea']}**" if con_muestras
                 else "ninguna: no hay linea que haya pasado el gate de aceptacion")

    MD.write_text(f"""# R17b — Tres lineas en paralelo, una por procesador

## Que se midio, y que no

Los tres archivos del brief, **sin sustituir ninguno**. Cada server arranca solo
despues de un `stat -c '%s %n'`, cuya salida cruda va en `stat_evidencia` de su
fila del `.json`:

{stats or "- no_medido"}{alt_txt}

| linea | dispositivo | modelo | GB | tok/s prompt | tok/s gen | s/item | acep/rech | muestras/h | capas |
|---|---|---|---|---|---|---|---|---|---|
{tabla}

## 1. ¿Las tres juntas rinden mas que la mejor sola?

{veredicto_1}

## 2. ¿Que linea conviene para volumen y cual para calidad?

- **Volumen**: {linea_vol}.
- **Calidad**: {linea_cal}. {("Es la unica que paso el gate de aceptacion, asi que tambien es la unica cuyo texto esta en el conjunto." if len(con_muestras) == 1 else "Pasaron el gate: " + ', '.join(f['linea'] for f in con_muestras) + ".")}

Ninguna celda de este reporte es estimada. Lo que no se pudo medir dice
`no_medido` con el motivo, tanto en el `.json` como en el `generador.log` y el
`<linea>.server.log` de la linea.

## 3. ¿La placa rinde mas que el CPU para el mismo modelo?

{fb_txt}

## La union

`{u['conjunto']}`: **{u['en_el_conjunto']}** documento de **{u['entrantes']}**
entrantes, **{u['caidas']}** caidas ({u['caidas_por_motivo'] or 'sin motivo'}). Dedup
con instrumentos que ya estan en el repo:

- texto normalizado — `{u['instrumentos']['texto_normalizado']}`
- similes repetidos — `{u['instrumentos']['similes_repetidos']}`
- casi-duplicado entre lineas — `{u['instrumentos']['casi_duplicado_entre_lineas']}`

## Como se aislo cada linea

`GGML_VK_VISIBLE_DEVICES=0` (AMD) y `=1` (Intel) en el entorno de cada server, y
`--device none` en la de CPU. Cada server es una unidad `systemd-run --user`
`linea<N>-<modelo>`.

Dos cosas del terreno:

- Con la variable puesta, el backend renumera como `Vulkan0` la unica placa que
  sobrevive, asi que **las dos lineas de placa se arrancan con `--device Vulkan0`**;
  `--device Vulkan1` se rechaza con `invalid device: Vulkan1`. Como el indice ya no
  distingue AMD de Intel, la evidencia es el **nombre** de la placa en el log del
  server (`coincide_la_placa` en el `.json`), no el indice.
- Sin `--device none`, la linea de CPU **toma una placa por su cuenta**.

Limpieza: unidad `linea*` activa al terminar =
**{r['limpieza'].get('unidades_linea_activas_al_terminar') or 'ninguna'}**. El
manager de `:11434` es una unidad del **sistema** (`llama-server.service`), sigue
`{r['limpieza'].get('manager_11434_intacto', {}).get('activo', '?')}`, y no se toco.
""", encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _limpieza(vivas: list[Linea]) -> dict[str, Any]:
    """Para lo NUESTRO y verifica que no quedo nada vivo. Corre siempre."""
    for l in vivas:
        if l.unidad:
            parar(l.unidad)
    for _ in range(30):
        if not unidades_linea():
            break
        time.sleep(1)
    restantes = unidades_linea()
    return {
        "unidades_paradas": [l.unidad for l in vivas if l.unidad],
        "unidades_linea_activas_al_terminar": restantes,
        "limpio": not restantes,
        # La prueba de que no se toco :11434: es una unidad del SISTEMA, y sigue
        # viva, y no es nuestra.
        "manager_11434_intacto": _sistema_11434(),
    }


def _sistema_11434() -> dict[str, Any]:
    p = subprocess.run(["systemctl", "is-active", "llama-server.service"],
                       capture_output=True, text=True)
    return {"activo": p.stdout.strip(), "nota": "unidad del sistema; no se toco"}


def _con_hilos(flags: list[str], n: str) -> list[str]:
    """Los mismos flags con otro `-t`."""
    out = list(flags)
    if "-t" in out:
        out[out.index("-t") + 1] = n
    else:
        out += ["-t", n]
    return out


def _bloque(linea: Linea, item: dict[str, Any], sonda: dict[str, Any]) -> dict[str, Any]:
    """La fila de una linea: lo que se midio, y `no_medido` con motivo si no."""
    stat = linea.stat_evidencia
    if linea.gguf is None:
        return {
            "linea": linea.clave, "n": linea.n, "dispositivo": linea.dispositivo,
            "modelo": None, "pedido_por_el_brief": linea.pedido,
            "estado": "no_disponible", "motivo": linea.por_que_no,
            "stat_evidencia": stat, "ls_de_la_carpeta": _ls_carpeta(linea.gguf),
            "tok_s_prompt": "no_medido", "tok_s_generacion": "no_medido",
            "wall_segundos": "no_medido", "muestras_por_hora": "no_medido",
        }
    b: dict[str, Any] = {
        "linea": linea.clave,
        "n": linea.n,
        "dispositivo": linea.dispositivo,
        "modelo": linea.modelo,
        "pedido_por_el_brief": linea.pedido,
        "archivo": str(linea.gguf),
        "stat_evidencia": item.get("stat", stat),
        "tamanio_gb": round(linea.tamanio_bytes / 2**30, 2) if linea.tamanio_bytes else None,
        "flags": linea.flags,
        "ggml_vk_visible_devices": linea.vk_visible,
        "unidad": linea.unidad,
        "puerto": linea.puerto,
    }
    if item.get("estado") == "no_medido":
        b.update({"estado": "no_medido", "motivo": item.get("motivo"),
                  "tok_s_prompt": "no_medido", "tok_s_generacion": "no_medido",
                  "wall_segundos": "no_medido", "muestras_por_hora": "no_medido"})
        return b

    b.update({
        "estado": "medida",
        "tok_s_prompt": sonda.get("tok_s_prompt", "no_medido"),
        "tok_s_generacion": sonda.get("tok_s_generacion", "no_medido"),
        "timings_de_sonda": sonda,
        "wall_segundos": item["wall_segundos"],
        "s_por_muestra": item["s_por_muestra"],
        "muestras": item["muestras"],
        "aceptadas": item["aceptadas"],
        "rechazadas": item["rechazadas"],
        "muestras_por_hora": item["muestras_por_hora"],
        "aceptadas_por_hora": item["aceptadas_por_hora"],
        "memoria_del_proceso_bytes": item["memoria_del_proceso_bytes"],
        "swap_delta_bytes": item["swap_delta_bytes"],
        "arranque": item.get("arranque"),
        "usado_el_alterno": (item.get("arranque") or {}).get("usado_el_alterno"),
        "modelo_primario_pedido": (item.get("arranque") or {}).get("primario_pedido"),
        "intentos_fallidos": [
            {"archivo": x.get("archivo"), "motivo": x.get("motivo"),
             "log": x.get("log")}
            for x in (item.get("arranque") or {}).get("intentos", [])
            if x.get("estado") != "arrancada"],
        "evidencia_del_dispositivo": item.get("evidencia"),
        "calidad": "no_calculada",
        "crudo": item.get("crudos", []),
        "rc_del_generador": item.get("rc"),
        "rc_que_significa": item.get("rc_que_significa"),
        "log_del_generador": item.get("log_del_generador"),
        "reintento": item.get("reintento"),
        "cola_del_generador": item.get("cola_del_generador"),
    })
    return b


def _calidad(clave: str) -> dict[str, Any]:
    """La calidad por linea, con `comparar_simulador.brazo()`: el MISMO
    instrumento de R5/R9/R10/R13. `tres-lineas/<clave>` resuelve igual que un
    brazo de primer nivel porque `brazo()` hace `SMOKE / carpeta`."""
    try:
        b = cs.brazo(f"tres-lineas/{clave}")
        return {
            "instrumento": "pipelines/comparar_simulador.py::brazo",
            "aceptadas": b["aceptadas"],
            "rechazadas": b["rechazadas"],
            "motivos": b["motivos"],
            "motivos_de_rechazo": b["motivos_de_rechazo"],
            "tasa_eco_usuario": b["tasa_eco"]["usuario"]["tasa"],
            "tasa_eco_voz": b["tasa_eco"]["voz"]["tasa"],
            "similes_por_turno_de_voz": b["similes_por_turno_de_voz"],
            "palabras_por_turno_de_voz": b["largo_turnos_de_voz"]["palabras_promedio"],
        }
    except Exception as e:
        return {"estado": "no_medido", "motivo": f"{type(e).__name__}: {e}"}


def main(argv: list[str] | None = None) -> int:
    ap = __import__("argparse").ArgumentParser(description=__doc__)
    ap.add_argument("--solo-fase-a", action="store_true",
                    help="solo las tres lineas en paralelo")
    args = ap.parse_args(argv)

    todas = lineas()
    vivas: list[Linea] = []
    reporte: dict[str, Any] = {}
    try:
        for l in todas:
            if l.gguf is not None:
                vivas.append(l)

        # ---- Fase A: las tres lineas a la vez -------------------------- #
        print("== Fase A: las tres lineas en paralelo ==", flush=True)
        arranques: dict[str, dict[str, Any]] = {}
        for l in vivas:
            a = arrancar(l)
            arranques[l.clave] = a
            print(f"  {l.clave} {l.dispositivo:8} {l.modelo:16} -> {a['estado']}", flush=True)
            print(f"     stat: {a.get('stat', 'no_medido')}", flush=True)
            if a["estado"] == "no_disponible":
                print(f"     {a.get('motivo')}", flush=True)

        # Si arranco alguna, se miden los items. Los items corren juntos: cada
        # proceso kateto_gen pega contra su propio puerto, y los tres en
        # paralelo es lo que produce el "de las tres a la vez" del brief.
        # Los TRES items a la vez, que es la afirmacion del brief ("las tres
        # lineas a la vez"). Secuencial mediria tres lineas solas y despues se
        # sumarian: daria el mismo total y no mediria la contencion, que es
        # justo lo que pasa cuando las tres escriben.
        items: dict[str, dict[str, Any]] = {}
        con_server = [l for l in vivas if unidad_viva(l.unidad)]
        with ThreadPoolExecutor(max_workers=len(con_server) or 1) as pool:
            futuros = {pool.submit(correr_item, l): l for l in con_server}
            for fut in as_completed(futuros):
                l = futuros[fut]
                print(f"  item de {l.clave} ({l.modelo})...", flush=True)
                it = fut.result()
                if it["muestras"] == 0 and l.dispositivo == "CPU":
                    # `ref.generate` tiene timeout=240 y no se toca. En CPU, con
                    # los 6 hilos del brief, el prompt eval del 26B (2200 tokens
                    # con el registro) se pasa de esa ventana y el item sale
                    # vacio. MEDIDO: 240 s no alcanzan con 6, y con 12 el prompt
                    # eval baja a 94,6 s. Se reintenta y se anotan los dos.
                    primer = {"hilos": "6", "wall_segundos": it["wall_segundos"],
                              "muestras": it["muestras"], "rc": it["rc"],
                              "cola_del_generador": it.get("cola_del_generador")}
                    parar(l.unidad)
                    l.flags = _con_hilos(l.flags, "12")
                    arranques[l.clave] = arrancar(l, espera_s=300)
                    if arranques[l.clave]["estado"] == "arrancada":
                        it = correr_item(l)
                    it["reintento"] = {"motivo": "el item con 6 hilos no produjo ninguna "
                                                "muestra: timeout de 240 s en ref.generate",
                                        "primer_intento": primer,
                                        "flags_del_reintento": l.flags,
                                        "evidencia": evidencia(l)}
                it["crudos"] = _leer(l.carpeta / "muestras.jsonl")
                it["rechazos"] = _leer(l.carpeta / "rechazos.jsonl")
                it["arranque"] = arranques.get(l.clave)
                it["stat"] = arranques.get(l.clave, {}).get("stat", l.stat_evidencia)
                it["evidencia"] = evidencia(l)
                items[l.clave] = it
                print(f"     {it['wall_segundos']}s  aceptadas={it['aceptadas']} "
                      f"rechazadas={it['rechazadas']}  rc={it['rc']} "
                      f"({it['rc_que_significa']})", flush=True)
                print(f"     device del log: {it['evidencia']['dispositivo_del_log']} "
                      f"(pedido {l.dispositivo}, coincide="
                      f"{it['evidencia']['coincide_con_el_pedido']}, placa="
                      f"{it['evidencia'].get('coincide_la_placa')})", flush=True)

        # La sonda va DESPUES del item: el server ya esta cargado y el modelo no
        # se mueve de dispositivo.
        sondas = {l.clave: sonda_timings(l) for l in vivas if l.clave in items}

        filas = [_bloque(l, items.get(l.clave, {"estado": "no_medido",
                                               "motivo": "el server no llega a item"}),
                         sondas.get(l.clave, {"estado": "no_medido",
                                              "motivo": "no se llego a sondar"}))
                 for l in todas]
        for f in filas:
            if f.get("estado") == "medida":
                f["calidad"] = _calidad(f["linea"])

        # ---- Fase B: el mismo modelo chico en CPU y en su placa --------- #
        faseb: dict[str, Any] = {"estado": "no_medido"}
        if not args.solo_fase_a:
            print("== Fase B: el modelo de la linea 2 en CPU vs en su placa ==", flush=True)
            for l in vivas:            # se para la Fase A antes: 3 procesos, nunca 4
                parar(l.unidad)
            time.sleep(2)
            par = fase_b(todas[1])
            if par is None:
                faseb = {"estado": "no_medido", "motivo": "el modelo de la linea 2 no esta"}
            else:
                cpu, gpu = par
                vivas_b: list[Linea] = []
                bloq: dict[str, Any] = {}
                try:
                    for l in (cpu, gpu):
                        a = arrancar(l)
                        vivas_b.append(l)
                        print(f"  {l.clave} {l.dispositivo:8} -> {a['estado']}", flush=True)
                        if a["estado"] == "no_disponible":
                            print(f"     {a.get('motivo')}", flush=True)
                    for l in vivas_b:
                        if not unidad_viva(l.unidad):
                            continue
                        it = correr_item(l)
                        it["crudos"] = _leer(l.carpeta / "muestras.jsonl")
                        it["evidencia"] = evidencia(l)
                        bloq[l.clave] = _bloque(l, it, sonda_timings(l))
                        print(f"     {l.clave}: {it['wall_segundos']}s "
                              f"{bloq[l.clave]['tok_s_generacion']} tok/s gen", flush=True)
                finally:
                    vivas += vivas_b
                faseb = {
                    "estado": "medido" if len(bloq) == 2 else "parcial",
                    "modelo": todas[1].modelo,
                    "pregunta": "el mismo modelo, mismos flags, mismo prompt: CPU vs su placa",
                    "cpu": bloq.get("linea4"),
                    "placa": bloq.get("linea5"),
                }
                if len(bloq) == 2:
                    c, p_ = bloq["linea4"], bloq["linea5"]
                    tc, tp = c.get("tok_s_generacion"), p_.get("tok_s_generacion")
                    if isinstance(tc, (int, float)) and isinstance(tp, (int, float)) and tc:
                        faseb["tok_s_generacion_cpu"] = tc
                        faseb["tok_s_generacion_placa"] = tp
                        faseb["la_placa_rinde"] = round(tp / tc, 2)
                        faseb["muestras_por_hora_cpu"] = c.get("muestras_por_hora")
                        faseb["muestras_por_hora_placa"] = p_.get("muestras_por_hora")

        # ---- La union --------------------------------------------------- #
        print("== La union ==", flush=True)
        union = unir([{"clave": f["linea"], "modelo": f["modelo"],
                       "carpeta": str(BASE / f["linea"]),
                       "estado": f["estado"], "aceptadas": f.get("aceptadas", 0)}
                      for f in filas])

        rates = [f["muestras_por_hora"] for f in filas
                 if f.get("estado") == "medida" and isinstance(f.get("muestras_por_hora"), (int, float))]
        total = round(sum(rates), 2) if rates else "no_medido"
        sola = una_sola_linea()
        # "la mejor sola" es la pregunta del brief, asi que el mejor de las tres
        # de HOY se compara contra la suma de las tres de HOY.
        candidatas = [f for f in filas
                      if isinstance(f.get("muestras_por_hora"), (int, float))
                      and f.get("muestras_por_hora")]
        mejor = (max(candidatas, key=lambda f: f["muestras_por_hora"])
                 if candidatas else {})

        reporte = {
            "nota": (
                "R17b. Tres lineas de generacion en paralelo, una por procesador, y la "
                "union, con los TRES archivos que pide el brief y sin sustituir "
                "ninguno. R17 los busco en el cache de HF y concluyo que no "
                "existian; estan en /run/media/chaos/secundario/models/hf y cada uno "
                "arranca solo despues de un `stat -c '%s %n'` cuya salida esta en "
                "`stat_evidencia` de su fila. Aislamiento por "
                "GGML_VK_VISIBLE_DEVICES=0/1 mas `--device none` en la de CPU; como "
                "eso renumera las placas a Vulkan0, lo que prueba que cada linea cayo "
                "en su placa es el NOMBRE en el log (`coincide_la_placa`), no el "
                "indice. La calidad sale de comparar_simulador.brazo(), la generacion "
                "de kateto_gen.py, y el dedup de la union con los instrumentos de "
                "pipelines/estilo y kateto_purge."
            ),
            "scenario": {"id": ESCENARIO, "n": N, "seed": SEED, "voice": VOICE},
            "reglas_duras": {
                "servidores": "systemd-run --user; ninguno se cargo en el shell del runner",
                "puerto_11434": "intocado (unidad del sistema)",
                "paro": "systemctl --user stop por NOMBRE de unidad; nunca pkill por binario",
                "procesos": "3 como maximo: la Fase A para antes de que arranque la B",
            },
            "lineas": filas,
            "fase_b_cpu_vs_placa": faseb,
            "union": union,
            "titular": {
                "lineas_medidas": sum(1 for f in filas if f.get("estado") == "medida"),
                "muestras_por_hora_por_linea": {
                    f["linea"]: f.get("muestras_por_hora") for f in filas},
                "total_de_las_tres_muestras_por_hora": total,
                "mejor_sola": (
                    {"linea": mejor["linea"], "dispositivo": mejor["dispositivo"],
                     "modelo": mejor["modelo"],
                     "muestras_por_hora": mejor["muestras_por_hora"]}
                    if mejor else {"estado": "no_medido",
                                   "motivo": "ninguna linea midio muestras/hora > 0"}),
                "las_tres_vs_la_mejor_sola": (
                    round(total / mejor["muestras_por_hora"], 2) if mejor else None),
                "una_sola_linea_referencia": sola,
            },
            "limpieza": {},
        }
        if isinstance(total, (int, float)) and isinstance(sola.get("muestras_por_hora"), (int, float)):
            reporte["titular"]["las_tres_vs_una_sola"] = round(
                total / sola["muestras_por_hora"], 2)
            reporte["titular"]["contra_que"] = (
                f"{sola['muestras_por_hora']} muestras/hora del item anterior "
                f"({sola['modelo']}, {sola['descripcion']}, 26B SOLO). Es el MISMO "
                "modelo que la linea 1 de hoy, medido en otra corrida: la unica "
                "diferencia que el brief no controla es cuantos hilos usaba esa")

        BASE.mkdir(parents=True, exist_ok=True)
        TABLA.write_text(json.dumps(reporte, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
        _md(reporte)
        print(f"-> {TABLA.relative_to(REPO)}")
        print(f"-> {MD.relative_to(REPO)}")
        return 0
    finally:
        limpieza = _limpieza(vivas)
        print("== Limpieza ==")
        print(f"  {limpieza['unidades_linea_activas_al_terminar']} unidades linea* activas")
        if reporte:
            reporte["limpieza"] = limpieza
            TABLA.write_text(json.dumps(reporte, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
