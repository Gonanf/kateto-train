#!/usr/bin/env python3
"""K5 — sistema donde vive Kateto: streaming, comedia generativa y verificacion.

Cuatro subcomandos sobre los mismos dos servidores llama.cpp:

    salud     probe de /health + /v1/models + /props + completion real
    comedia   turno de comedia con el registro inyectado (prompt engineering)
    stream    captura audio -> transcripcion -> inferencia -> emision (TTS)
    feedback  verificacion con feedback estructurado (reglas R*/M* + juez LLM)

Why stdlib-only: el repo ya tiene `venv/bin/python` con numpy/torch, pero los
scripts de produccion (`eval_http_backend.py`, `kateto/safety/`) son stdlib puro
para que corran con el python del sistema. Este archivo mantiene esa regla:
solo `urllib`, `subprocess`, `wave`, `re`, `json`.

Why NO se reimplementa nada de esto:
  - la inferencia HTTP la hace `scripts/eval_http_backend.py::generate_http`
  - el TTS lo hace `rwkv_pipeline/piper_tts.py::PiperTTSProvider`
  - el filtro pre-TTS lo hace `kateto/safety/content_filter.py::ContentFilter`
  - el registro lo trae `prompts/prompt-compartido.md` VERBATIM
Todo lo que hay aca es pegamento, CLIs y las reglas de feedback.

    python scripts/k5_sistema.py salud
    python scripts/k5_sistema.py comedia --voz jane "me trabé con el deploy"
    python scripts/k5_sistema.py feedback --prompt "..." --respuesta "..."
    python scripts/k5_sistema.py stream --turnos 1        # necesita microfono
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from scripts.eval_http_backend import generate_http  # noqa: E402
from rwkv_pipeline.piper_tts import PiperTTSProvider, split_sentences  # noqa: E402
from kateto.safety.content_filter import ContentFilter  # noqa: E402

REGISTRO_PATH = REPO / "prompts" / "prompt-compartido.md"

# --------------------------------------------------------------------------- #
# 1. Servidores. Los dos del brief, con el nombre que reporta /v1/models.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Server:
    nombre: str
    url: str
    modelo: str


SERVERS: tuple[Server, ...] = (
    Server("neohorse", "http://127.0.0.1:11662", "neohorse"),
    Server("e2b", "http://127.0.0.1:11661", "e2b"),
)

# --------------------------------------------------------------------------- #
# 2. Voces. Los tres perfiles de prompts/tics-por-voz.md.
# --------------------------------------------------------------------------- #
VOCES: dict[str, dict] = {
    "jane": {
        "desc": "seca y corta. Al punta, sin buildup, corta el loop sin culpar.",
        "max_len": 48,
        "stop": ["\n\n"],
    },
    "doktor": {
        "desc": "grupo que ayuda igual. Protesta primero, oferta despues.",
        "max_len": 72,
        "stop": ["\n\n"],
    },
    "conquest": {
        "desc": "ceremonial. Se toma en serio el Tearoom y el relato.",
        "max_len": 96,
        "stop": ["\n\n"],
    },
}

VOZ_POR_DEFECTO = "jane"


def registro() -> str:
    """`prompts/prompt-compartido.md` VERBATIM.

    Igual que `pipelines/kateto_gen.py::registro()`: el archivo es la
    especificacion, copiarlo al codigo lo desalinea en silencio en la primera
    edicion del doc.
    """
    return REGISTRO_PATH.read_text(encoding="utf-8")


def system_block(voz: str) -> str:
    """El bloque que va al prompt: registro + la voz elegida.

    `test_voz` (comedia.md) define COMO suena cada una; el resto del registro es
    comun. La voz va ultimo porque es lo que el modelo tiene que retaining al
    final del contexto.
    """
    p = VOCES.get(voz)
    if p is None:
        raise KeyError(f"voz desconocida: {voz!r} (hay: {', '.join(VOCES)})")
    return f"{registro()}\n\nTu voz es {voz}: {p['desc']}"


def prompt_turno(historial: list[tuple[str, str]], pregunta: str) -> str:
    """Arma el prompt crudo. Sin chat template: `/completion` toma texto plano.

    Por que no `/v1/chat/completions`: los dos modelos que sirven hoy
    (NeoHorse-1-4B, gemma-4-E2B-it) son instruct con su propio template, pero
    Kateto se entrena con el markers de RWKV (`<|im_end|>`), no con el de ellos.
    Un prompt crudo con el registro arriba es el unico formato que las dos
    partes comparten; si se cambia el modelo servido, este es el unico punto
    a tocar.
    """
    partes = [registro()]
    for user, kateto in historial:
        partes.append(f"{user}\n{kateto}")
    partes.append(f"{pregunta}\n")
    return "\n\n".join(partes)


# --------------------------------------------------------------------------- #
# 3. `salud`
# --------------------------------------------------------------------------- #
def _get_json(url: str, timeout: int = 10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def probe_server(s: Server, *, probar_inferencia: bool = True) -> dict:
    """Un probe por servidor. Nunca levanta: un server caido es un dato, no un error.

    Reporta las TRES capas por separado porque mienten distinto:
      - /health   el proceso responde
      - /v1/models el modelo esta cargado (o no)
      - completion  la inferencia anda de verdad (o esta colgada)
    """
    out: dict = {
        "servidor": s.nombre,
        "url": s.url,
        "health": False,
        "modelo_cargado": False,
        "inferencia": False,
        "detalle": {},
    }
    t0 = time.time()

    try:
        out["health"] = _get_json(f"{s.url}/health").get("status") == "ok"
    except Exception as e:
        out["detalle"]["health"] = f"{type(e).__name__}: {e}"

    try:
        m = _get_json(f"{s.url}/v1/models")
        # El repo ya documento las DOS formas de respuesta (eval_http_backend
        # `wait_for_model`): router da `data[].id`, llama-server suelto da
        # `models[].name`. Parsear solo una hace que el probe mienta.
        nombres = [x.get("id") for x in m.get("data", [])]
        nombres += [x.get("name") or x.get("model") for x in m.get("models", [])]
        out["detalle"]["nombres"] = [n for n in nombres if n]
        out["modelo_cargado"] = s.modelo in out["detalle"]["nombres"]
        meta = (m.get("data") or [{}])[0].get("meta") or {}
        if meta:
            out["detalle"]["n_ctx"] = meta.get("n_ctx")
            out["detalle"]["params"] = meta.get("n_params")
    except Exception as e:
        out["detalle"]["modelos"] = f"{type(e).__name__}: {e}"

    try:
        p = _get_json(f"{s.url}/props")
        out["detalle"]["n_ctx_total"] = p.get("default_generation_settings", {}).get("n_ctx") or p.get("n_ctx")
        out["detalle"]["slots"] = p.get("total_slots")
        out["detalle"]["ckpt"] = Path(p.get("model_path", "?")).name
    except Exception as e:
        out["detalle"]["props"] = f"{type(e).__name__}: {e}"

    if probar_inferencia and out["health"] and out["modelo_cargado"]:
        try:
            r = generate_http(s.url, s.modelo, "Che.", max_len=8, temperature=0.0,
                              timeout_s=60)
            out["inferencia"] = True
            out["detalle"]["tokens"] = r["n_tokens"]
            out["detalle"]["s"] = r["elapsed_s"]
        except Exception as e:
            out["detalle"]["inferencia"] = f"{type(e).__name__}: {e}"

    out["sano"] = out["health"] and out["modelo_cargado"] and out["inferencia"]
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def cmd_salud(args) -> int:
    res = [probe_server(s, probar_inferencia=not args.sin_inferencia) for s in SERVERS]

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(f"{'servidor':10} {'url':26} {'health':7} {'modelo':8} {'infer':6} {'s':>6}")
        for r in res:
            print(f"{r['servidor']:10} {r['url']:26} "
                  f"{str(r['health']):7} {str(r['modelo_cargado']):8} "
                  f"{str(r['inferencia']):6} {r['elapsed_s']:>6}")
            for k, v in r["detalle"].items():
                print(f"           {k}: {v}")
            if not r["sano"]:
                for k, v in r["detalle"].items():
                    if isinstance(v, str) and ("Error" in v or "error" in v):
                        print(f"           ! {k}: {v}")
        sanos = sum(r["sano"] for r in res)
        print(f"\n{sanos}/{len(res)} sanos")

    return 0 if all(r["sano"] for r in res) else 1


# --------------------------------------------------------------------------- #
# 4. `comedia`
# --------------------------------------------------------------------------- #
# Turno mudo. MEDIDO 2026-10-07 con neohorse: 2 de 6 llamadas con el mismo
# prompt y la misma seed devolvieron 1 token y nada (`n_tokens: 1`,
# `stopping_word: ''` -> EOS inmediato, no un stop). El seed NO lo arregla: con
# `cache_prompt: False` mas batching continuo el muestreo de llama.cpp no es
# bit-determinista, y la misma seed dio 6 salidas distintas. Por eso el
# reintento cambia la seed en vez de repetir la llamada identica.
TURNOS_MUDOS_UMBRAL = 2      # menos de 2 caracteres es un turno mudo
TURNOS_MUDOS_REINTENTOS = 3  # intentos totales (1 original + 2 reintentos)


def hablar(pregunta: str, voz: str = VOZ_POR_DEFECTO, *,
           historial: list[tuple[str, str]] | None = None,
           server: Server | None = None,
           max_len: int | None = None,
           temperatura: float = 0.85,
           seed: int = 1337) -> dict:
    """Un turno. Devuelve el texto y la traza (para `feedback` y para tests).

    Reintenta si el turno sale mudo. El guard va ACA y no en los callers porque
    `comedia`, `stream` y `feedback --desde-comedia` pasan todos por acá.
    """
    s = server or SERVERS[0]
    p = VOCES[voz]
    hist = list(historial or [])
    reintentos = 0

    while True:
        r = generate_http(
            s.url, s.modelo, prompt_turno(hist, pregunta),
            max_len=max_len or p["max_len"],
            temperature=temperatura,
            top_p=0.90,
            seed=seed + reintentos,
            stop=p["stop"],
            repeat_penalty=1.15,
        )
        # `cache_prompt: False` ya viene en generate_http (obligatorio, estado
        # sucio entre llamadas). Aca solo se corta el prefijo que el modelo a
        # veces repite.
        txt = r["text"].strip()
        if txt.startswith(pregunta):
            txt = txt[len(pregunta):].strip()

        if len(txt) >= TURNOS_MUDOS_UMBRAL or reintentos >= TURNOS_MUDOS_REINTENTOS - 1:
            break
        reintentos += 1
        print(f"[k5] turno mudo ({s.nombre}), reintento {reintentos}", file=sys.stderr)

    return {"texto": txt, "voz": voz, "servidor": s.nombre, "reintentos": reintentos,
            "raw": {k: v for k, v in r.items() if k != "raw"}}


def cmd_comedia(args) -> int:
    voz = args.voz
    hist: list[tuple[str, str]] = []
    server = next((s for s in SERVERS if s.nombre == args.servidor), None)
    if server is None:
        print(f"servidor desconocido: {args.servidor}", file=sys.stderr)
        return 2

    if args.interactivo:
        print(f"[k5] {voz} | {server.nombre} | escribí y dale Enter "
              f"(Ctrl-C o linea vacia para salir)", file=sys.stderr)
        while True:
            try:
                q = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(file=sys.stderr)
                break
            if not q:
                break
            r = hablar(q, voz, historial=hist, server=server)
            print(r["texto"])
            hist.append((q, r["texto"]))
        return 0

    if not args.pregunta:
        print("hace falta --pregunta (o --interactivo)", file=sys.stderr)
        return 2

    r = hablar(args.pregunta, voz, server=server)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print(r["texto"])
    return 0


# --------------------------------------------------------------------------- #
# 5. `stream` — captura -> ASR -> inferencia -> emision
# --------------------------------------------------------------------------- #
ASR_POR_DEFECTO = ("whisper-cli", "main", "whisper", "whisper-cpp")

# ffmpeg: 16k mono, que es lo que come cualquier ASR y lo que piper no necesita
# (piper resamplea solo desde su propio input).
FFMPEG_CAP = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "alsa",
              "-i", "default", "-ac", "1", "-ar", "16000"]


@dataclass
class Turno:
    """Todo lo que produjo un turno. Sirve de log de la sesion."""
    t: float
    wav: str
    transcribe: str
    texto: str
    wavs: list[str] = field(default_factory=list)
    bleep: list[str] = field(default_factory=list)


def capturar(dest: Path, segundos: float, *, device: str = "default") -> Path:
    """Graba con ffmpeg. Devuelve el wav. Sin numpy: ffmpeg hace el PCM."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg no esta en el PATH (necesario para capturar)")
    cmd = [c if c != "default" or device == "default" else device
           for c in FFMPEG_CAP]
    if device != "default":
        cmd[cmd.index("alsa") + 1] = device
    cmd += ["-t", str(segundos), str(dest)]
    subprocess.run(cmd, check=True)
    return dest


def transcribir(wav: Path, cmd_template: str | None) -> str:
    """Transcribe vía un comando externo. Plantilla con `{wav}`.

    Por que NO hay ASR adentro: en esta caja no hay whisper/whisper-cli
    (verificado 2026-10-07) y los dos llama-server no hacen audio. Agregar un
    modelo de ASR adentro seria 1GB de dependencias; la plantilla deja el
    binario como decision del operador y el resto del pipeline igual se puede
    ejercitar con `--texto`.
    """
    if cmd_template:
        cmd = cmd_template.replace("{wav}", str(wav))
    else:
        exe = next((b for b in ASR_POR_DEFECTO if shutil.which(b)), None)
        if exe is None:
            raise RuntimeError(
                f"no hay ASR: ni {', '.join(ASR_POR_DEFECTO)} ni --asr-cmd. "
                f"Para ejercitar la cadena sin microfono: --texto \"...\"")
        cmd = f"{exe} -l es -f {wav} -nt -otxt -of -"

    out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ASR fallo ({out.returncode}): {out.stderr[-400:]}")
    # whisper escribe timestamps [00:00.000 --> ...]; el -of - sale en stdout.
    txt = re.sub(r"^\[.*?\]\s*$", "", out.stdout, flags=re.M).strip()
    return re.sub(r"\s+", " ", txt)


def emitir(texto: str, outdir: Path, voz: str, *, sonar: bool = False,
           tts: PiperTTSProvider | None = None) -> tuple[list[str], list[str]]:
    """Filtra por oracion, sintetiza y opcionalmente reproduce. Emision incremental.

    Devuelve (wavs emitidos, oraciones bleepeadas). El filtro va POR ORACION y
    ANTES de sintetizar, que es el contrato que `content_filter.py` deja escrito
    en su docstring ("MUST call ContentFilter.check() on each text chunk BEFORE
    synthesis"). HARD devuelve texto vacio -> se calla la frase entera.
    """
    tts = tts or PiperTTSProvider()
    outdir.mkdir(parents=True, exist_ok=True)
    wavs: list[str] = []
    bleeps: list[str] = []

    for i, sent in enumerate(split_sentences(texto)):
        limpio, modified = ContentFilter.check(sent)
        if modified and not limpio:
            bleeps.append(sent)
            print(f"  [filtro HARD] silenciada: {sent[:60]}", file=sys.stderr)
            continue
        if modified:
            print(f"  [filtro SOFT] {sent[:40]} -> {limpio[:60]}", file=sys.stderr)
        p = tts.synth_sentence(limpio, outdir / f"turno_{i:03d}.wav")
        wavs.append(str(p))
        print(f"  wav {p} ({_dur(p):.2f}s)", file=sys.stderr)
        if sonar:
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(p)],
                           check=False)
    return wavs, bleeps


def _dur(p: Path) -> float:
    with wave.open(str(p)) as w:
        return w.getnframes() / w.getframerate()


def cmd_stream(args) -> int:
    voz = args.voz
    server = next((s for s in SERVERS if s.nombre == args.servidor), SERVERS[0])
    outdir = Path(args.out)
    hist: list[tuple[str, str]] = []
    turnos: list[Turno] = []
    tmp = Path(tempfile.mkdtemp(prefix="k5-stream-"))

    for n in range(args.turnos):
        print(f"[k5] turno {n + 1}/{args.turnos}", file=sys.stderr)
        if args.texto:
            dicho = args.texto
            wav = ""
            print(f"  transcripcion (inyectada): {dicho}", file=sys.stderr)
        else:
            wav = str(capturar(tmp / f"turno_{n:03d}.wav", args.segundos,
                               device=args.dispositivo))
            dicho = transcribir(Path(wav), args.asr_cmd)
            print(f"  transcripcion: {dicho}", file=sys.stderr)

        if not dicho:
            print("  (nada dicho, se saltea)", file=sys.stderr)
            continue

        r = hablar(dicho, voz, historial=hist, server=server)
        print(f"  kateto: {r['texto']}", file=sys.stderr)
        wavs, bleeps = emitir(r["texto"], outdir / f"turno_{n:03d}", voz,
                              sonar=args.sonar)
        hist.append((dicho, r["texto"]))
        turnos.append(Turno(t=time.time(), wav=wav, transcribe=dicho,
                            texto=r["texto"], wavs=wavs, bleep=bleeps))

    if args.json:
        print(json.dumps([t.__dict__ for t in turnos], ensure_ascii=False, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# 6. `feedback` — verificacion con feedback estructurado
# --------------------------------------------------------------------------- #
# Cada check apunta a una regla REAL de prompts/registro-kateto.md (R*) o
# prompts/comedia.md (M*). No hay taxonomia inventada: la numeracion ya existe y
# docs/FUENTES.md la traza.
@dataclass
class Regla:
    id: str
    que: str
    patron: "re.Pattern[str]"
    gravedad: str  # "grave" | "aviso"


REGLAS: tuple[Regla, ...] = (
    Regla("R4/M6", "prosodia escrita entre corchetes/asteriscos/parentesis",
          re.compile(r"\[[^\]]{0,40}\]|\*[^*\n]{0,40}\*|\((?:risa|pausa|tono|suspiro|ri)[^)]*\)",
                     re.IGNORECASE), "grave"),
    Regla("R8", "nunca exclamacion",
          re.compile(r"!"), "aviso"),
    Regla("R22", "explica el chiste o recapitula",
          re.compile(r"(el chiste es que|lo que quiero decir es|b\xe9sicamente|"
                     r"en otras palabras|como te decia antes)", re.IGNORECASE), "grave"),
    # "(?:te )?" es opcional a proposito: MEDIDO 2026-10-07, el modelo servido
# devuelve "en qué puedo ayudarte" (sin "te") y esa forma se escapaba.
Regla("R21", "cierre de servicio (pregunta de aprobacion)",
          re.compile(r"(\u00bfquer[eé]s que|\u00bfte sirve|\u00bfalgo m[aá]s|"
                     r"en qu[eé] (?:te )?puedo ayudar|\u00bfnecesit[aá]s algo)",
                     re.IGNORECASE), "grave"),
    Regla("M9/R19", "formula servil de apertura",
          re.compile(r"^\s*(\u00a1?claro|\u00a1?hola|por supuesto|entendido|"
                     r"\u00a1?excelente pregunta|con gusto)", re.IGNORECASE), "grave"),
    # El mismo `think_leak` de config/behavior_patterns.yaml §2.2. Se reimplementa
    # en vez de leer el YAML porque este archivo no depende de PyYAML (stdlib-only)
    # y porque el feedback necesita el id de regla, que el YAML no tiene.
    Regla("think_leak", "razonamiento filtrado a la respuesta visible",
          re.compile(r"(<think>|</think>|\bthe user\b|\bLet me think\b)", re.IGNORECASE),
          "grave"),
    Regla("M6", "emoji", re.compile("[\U0001F300-\U0001FAFF☀-➿]"), "grave"),
    Regla("R19", "ataca identidad o cuerpo",
          re.compile(r"(tu (?:familia|madre|padre)|gordo|retrasad|bolita|paragua)\b",
                     re.IGNORECASE), "grave"),
)

JUEZ_PROMPT = """Sos un evaluador de un comediante rioplatense.
Kateto es seco, con autodepreciacion. Sin emoji, sin exclamaciones, sin notacion
de tono entre parentesis o corchetes, sin explicar el chiste, sin preguntar si
queres que haga algo.

Estimulo: {pregunta}
Turno: {turno}

Puntua 0-3 cada eje: ritmo, anclaje (rioplatense real, no translationese), seco."""

JUEZ_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "ritmo": {"type": "integer", "minimum": 0, "maximum": 3},
        "anclaje": {"type": "integer", "minimum": 0, "maximum": 3},
        "seco": {"type": "integer", "minimum": 0, "maximum": 3},
        "notas": {"type": "array", "maxItems": 4,
                  "items": {"type": "string", "maxLength": 60}},
    },
    "required": ["ritmo", "anclaje", "seco", "notas"],
}

# Techo alto a proposito: neohorse rellena con espacios y con 180 truncaba el
# JSON. `n_predict` es un techo, no un costo — el modelo corta en EOS igual.
JUEZ_MAX_TOKENS = 600


def feedback_deterministico(texto: str) -> dict:
    """Lo que un regex puede verificar sin modelo. Siempre corre, sin red."""
    violaciones = [
        {"regla": r.id, "que": r.que, "cita": (m.group(0) or "")[:60],
         "gravedad": r.gravedad}
        for r in REGLAS for m in [r.patron.search(texto)] if m
    ]
    graves = sum(v["gravedad"] == "grave" for v in violaciones)
    return {
        "veredicto": "ajuste" if graves else "ok",
        "violaciones": violaciones,
        "puntajes": {
            "limites": max(0, 3 - graves),
            "limpieza": max(0, 3 - sum(v["gravedad"] == "aviso" for v in violaciones)),
        },
    }


def feedback_juez(pregunta: str, turno: str, server: Server, *,
                  timeout_s: int = 120) -> dict:
    """Juez LLM. Best-effort: un juez caido NO es un veredicto de paso.

    La salida va forzada por `json_schema` de llama.cpp, no por regex sobre texto
    libre. MEDIDO 2026-10-07: asking for JSON in the prompt no funcionaba —
    neohorse gastaba los 180 tokens en un `<think>` y no emitia nada parseable
    (0/1). Con la gramatica el shape sale garantizado, y `notas` lleva
    `maxLength` porque sin tope el modelo encadenaba espacios hasta agotar el
    presupuesto y devolvia JSON truncado. 6/6 con este schema en los dos
    servidores.
    """
    payload = {
        "model": server.modelo,
        "prompt": JUEZ_PROMPT.format(pregunta=pregunta, turno=turno),
        "n_predict": JUEZ_MAX_TOKENS,
        "temperature": 0.3,
        "cache_prompt": False,
        "json_schema": JUEZ_SCHEMA,
    }
    try:
        req = urllib.request.Request(
            f"{server.url}/completion", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            d = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    txt = (d.get("content") or "").strip()
    if not txt:
        return {"ok": False, "error": "el juez no devolvio nada"}
    try:
        j = json.loads(txt)
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"JSON invalido: {e}", "crudo": txt[:200]}
    return {"ok": True, "juez": j}


def verificar(prompt: str, respuesta: str, *, voz: str = VOZ_POR_DEFECTO,
              server: Server | None = None, con_juez: bool = False) -> dict:
    """Feedback estructurado. El nucleo del contrato de K5."""
    det = feedback_deterministico(respuesta)
    acciones = [f"{v['regla']}: sacar {v['cita']!r}" for v in det["violaciones"]]
    out = {
        "voz": voz,
        "servidor": (server or SERVERS[0]).nombre,
        "prompt": prompt,
        "respuesta": respuesta,
        "deterministico": det,
        "acciones": acciones,
    }
    if con_juez:
        out["juez"] = feedback_juez(prompt, respuesta, server or SERVERS[0])
    out["veredicto"] = det["veredicto"]
    return out


def cmd_feedback(args) -> int:
    respuesta = args.respuesta
    if args.desde_comedia:
        respuesta = hablar(args.prompt, args.voz).get("texto", "")
    server = next((s for s in SERVERS if s.nombre == args.servidor), SERVERS[0])
    r = verificar(args.prompt, respuesta, voz=args.voz, server=server,
                  con_juez=args.juez)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0 if r["veredicto"] == "ok" else 1


# --------------------------------------------------------------------------- #
# 7. CLI
# --------------------------------------------------------------------------- #
def _comun(p: argparse.ArgumentParser) -> None:
    p.add_argument("--servidor", default=SERVERS[0].nombre, choices=[s.nombre for s in SERVERS],
                   help="que modelo contesta (default: %(default)s)")
    p.add_argument("--voz", default=VOZ_POR_DEFECTO, choices=list(VOCES),
                   help="perfil de voz de prompts/tics-por-voz.md (default: %(default)s)")
    p.add_argument("--json", action="store_true", help="salida JSON")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="k5_sistema", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("salud", help="probe de los servidores de modelo")
    s.add_argument("--sin-inferencia", action="store_true",
                   help="no genera: solo /health + /v1/models + /props")
    _comun(s)
    s.set_defaults(func=cmd_salud)

    c = sub.add_parser("comedia", help="turno de comedia con el registro inyectado")
    c.add_argument("pregunta", nargs="?", default="", help="estimulo")
    c.add_argument("--interactivo", action="store_true", help="loop de turnos por stdin")
    _comun(c)
    c.set_defaults(func=cmd_comedia)

    t = sub.add_parser("stream", help="audio -> transcripcion -> inferencia -> emision")
    t.add_argument("--turnos", type=int, default=1)
    t.add_argument("--segundos", type=float, default=5.0, help="duracion de cada captura")
    t.add_argument("--dispositivo", default="default", help="dispositivo de entrada de ffmpeg")
    t.add_argument("--asr-cmd", default=None,
                   help="comando de transcripcion con {wav}, ej: 'whisper-cli -l es -f {wav} -otxt -of -'")
    t.add_argument("--texto", default=None, help="transcripcion inyectada: corre la cadena sin microfono")
    t.add_argument("--sonar", action="store_true", help="reproducir con ffplay")
    t.add_argument("--out", default="out/k5-stream", help="directorio de wavs")
    _comun(t)
    t.set_defaults(func=cmd_stream)

    f = sub.add_parser("feedback", help="verificacion con feedback estructurado")
    f.add_argument("--prompt", required=True, help="el estimulo")
    f.add_argument("--respuesta", default="", help="el turno a evaluar")
    f.add_argument("--desde-comedia", action="store_true",
                   help="generar la respuesta con el modelo y evaluarla")
    f.add_argument("--juez", action="store_true", help="sumar puntajes de un juez LLM")
    _comun(f)
    f.set_defaults(func=cmd_feedback)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())