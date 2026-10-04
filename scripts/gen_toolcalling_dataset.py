#!/usr/bin/env python
"""Generador del dataset de tool calling + conversacion multi-turno (bloque A/D).

Objetivo (§3 y §4 del spec): conversaciones multi-turno en registro argentino donde
la voz usa las tools REALES de Kateto y de video-rag, con errores y rechazos reales.

DISENO EN DOS PARTES, y la parte que importa es la segunda:

  1. `generate()` — pide la muestra al teacher (cualquier endpoint OpenAI-compatible;
     por defecto freellmapi en :3001).
  2. `validate()` — RECHAZA la muestra si no cumple. Esta parte es la que evita
     repetir el desastre del dataset actual, donde el 7.5% de los objetivos eran
     tool calls del agente Hermes y 3.7% tenia voz argentina.

`--dry-run` valida muestras de ejemplo sin llamar al teacher: sirve para probar el
validador sin depender de que haya teacher disponible.

Uso:
    python scripts/gen_toolcalling_dataset.py --dry-run
    python scripts/gen_toolcalling_dataset.py --n 200 --model auto --out out/dataset/tc_v1.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Modelo default por modo de teacher. cmd esta fuera del CLI (prohibido).
DEFAULT_MODEL: dict[str, str] = {
    "http": "poolside-laguna-s-2.1",
    "opencode": "opencode/muse-spark-1.3-contributor-free",
    "cline": "meta/muse-spark-1.3-contributor-free",
}

# --------------------------------------------------------------------------- #
# 1. Catalogo REAL de tools (verificado en codigo, no en docs)
# --------------------------------------------------------------------------- #

# Kateto builtin: 18 (kateto/voices/tools.py, extraidas de los esquemas)
KATETO_TOOLS: dict[str, dict] = {
    "read_file":        {"required": ["path"], "optional": []},
    "write_file":       {"required": ["path", "content"], "optional": []},
    "delete_file":      {"required": ["path"], "optional": []},
    "run_command":      {"required": ["command"], "optional": []},
    "send_event":       {"required": ["event_name", "data"], "optional": ["target"]},  # runtime: data OBLIGATORIO
    "list_events":      {"required": [], "optional": []},
    "enable_plugin":    {"required": ["name"], "optional": []},
    "disable_plugin":   {"required": ["name"], "optional": []},
    "list_plugins":     {"required": [], "optional": []},
    "create_skill":     {"required": ["name", "content"], "optional": []},
    "update_skill":     {"required": ["name", "content"], "optional": []},
    "create_workflow":  {"required": ["name", "voice", "content"], "optional": []},
    "update_workflow":  {"required": ["name", "voice", "content"], "optional": []},
    "update_soul":      {"required": ["name", "content"], "optional": []},
    "refine_memory":    {"required": ["dept", "op", "evidence"], "optional": ["fact"]},
    "request_generation": {"required": ["target_voice", "prompt"], "optional": ["dept"]},
    "schedule_event":   {"required": ["event_name"], "optional": ["delay", "interval", "cron", "target_voice", "dept"]},
    "get_current_time": {"required": [], "optional": []},
}

# video-rag MCP: 6 (repo ~/proyectos/video-rag, src/mcp.rs)
VIDEORAG_TOOLS: dict[str, dict] = {
    "search":          {"required": ["query"], "optional": ["video", "k"]},
    "list_videos":     {"required": [], "optional": []},
    "describe":        {"required": ["video"], "optional": []},
    "describe_images": {"required": ["prompt", "images"], "optional": []},
    "answer":          {"required": ["video", "query"], "optional": ["k"]},
    "summarize":       {"required": ["video"], "optional": []},
}

# Juegos: chess por prompt, voyager_* dinamicas con GATE DOBLE
GAME_TOOLS: dict[str, dict] = {
    "voyager_craft_stick": {"required": [], "optional": [], "gate": "minecraft + skill habilitado"},
    "voyager_mine_diamond": {"required": [], "optional": [], "gate": "minecraft + skill habilitado"},
}

ALL_TOOLS = {**KATETO_TOOLS, **VIDEORAG_TOOLS, **GAME_TOOLS}

# --------------------------------------------------------------------------- #
# Tipos REALES de los argumentos, extraidos del runtime que los ejecuta
# (kateto/voices/tools.py, VoiceToolExecutor, 2026-09-15). Solo builtin: las
# tools de MCP (video-rag) y las de juego no tienen esquema local verificable,
# asi que ahi no se chequea tipo. Un `data` string donde va un objeto, o un
# `delay` string donde va un numero, son fallos de esquema que el runtime
# rechaza en preflight_tool_arguments().
# --------------------------------------------------------------------------- #
TOOL_TYPES: dict[str, dict[str, str]] = {
    "read_file":          {"path": "string"},
    "write_file":         {"path": "string", "content": "string"},
    "delete_file":        {"path": "string"},
    "run_command":        {"command": "string"},
    "send_event":         {"event_name": "string", "data": "object", "target": "string"},
    "enable_plugin":      {"name": "string"},
    "disable_plugin":     {"name": "string"},
    "create_skill":       {"name": "string", "content": "string"},
    "update_skill":       {"name": "string", "content": "string"},
    "create_workflow":    {"name": "string", "voice": "string", "content": "string"},
    "update_workflow":    {"name": "string", "voice": "string", "content": "string"},
    "update_soul":        {"name": "string", "content": "string"},
    "request_generation": {"target_voice": "string", "prompt": "string", "dept": "string"},
    "schedule_event":     {"event_name": "string", "delay": "number", "interval": "number",
                           "cron": "string", "target_voice": "string", "dept": "string"},
}


def _tipo_ok(valor, esperado: str) -> bool:
    """Chequeo de tipo JSON tolerante (bool no cuenta como number)."""
    if esperado == "string":
        return isinstance(valor, str)
    if esperado == "number":
        return isinstance(valor, (int, float)) and not isinstance(valor, bool)
    if esperado == "integer":
        return isinstance(valor, int) and not isinstance(valor, bool)
    if esperado == "object":
        return isinstance(valor, dict)
    if esperado == "array":
        return isinstance(valor, list)
    if esperado == "boolean":
        return isinstance(valor, bool)
    return True


# Tuteo / espanol neutro en 2a persona: el dataset es rioplatense, la forma de
# "vos" es obligatoria. RX_VOSEO es una comprobacion POSITIVA y puede pasar una
# muestra que dice "che" una vez y tutea el resto; esta lista la cierra.
# fix123: partida en DOS. Medido en kateto-ola99: 222/334 rechazos por tuteo eran
# falsos positivos. 147 con el tuteo SOLO en el turno del usuario (el usuario no
# es la voz: no tiene que hablar en voseo) y 75 homografos de 3a persona en la
# voz ("la heladera te mira", "todos lo ven"). Solo los inequivocos y los
# homografos sin clitico/sujeto delante son error, y SOLO en turnos de la voz.
RX_TUTEO_INEQUIVOCO = re.compile(
    r"(?<![\wáéíóúñ])(tienes|puedes|quieres|eres|dime|hazlo|amigo mio|amigo mío)"
    r"(?![\wáéíóúñ])", re.I)
RX_TUTEO_HOMOGRAFO = re.compile(
    r"(?<![\wáéíóúñ])(mira|ven|haz)(?![\wáéíóúñ])", re.I)
# Guarda de homografos: "te mira", "todos lo ven", "nadie lo tapa" — el pronombre
# o sujeto delante hace que la palabra sea 3a persona, no imperativo de tuteo.
RX_TUTEO_GUARDA = re.compile(r"\b(te|me|lo|la|se|nos|les|todos|nadie|ellos)\s+$", re.I)


def _tuteo_en(texto: str) -> str | None:
    """Devuelve la primera palabra de tuteo REAL del texto, o None.

    Los inequivocos (tienes, dime, hazlo, ...) siempre son error. Los homografos
    (mira, ven, haz) solo si NO estan precedidos por clitico o sujeto.

    El texto se normaliza a NFC antes de matchear (R12). Sin eso la regla depende
    de como venga la cadena: `Mirá` con á precompuesta NO dispara (es voseo) pero
    la MISMA palabra con el acento combinante U+0301 (NFD) dispara `tuteo:mira`,
    porque U+0301 no es un caracter de palabra para `\w` y la Regex se lo come
    como borde. O sea: el voseo correcto se rechaza segun la normalizacion que
    haya tenido el texto en el camino. Medido, no teorico.
    """
    texto = unicodedata.normalize("NFC", texto)
    m = RX_TUTEO_INEQUIVOCO.search(texto)
    if m:
        return m.group(1).lower()
    for m in RX_TUTEO_HOMOGRAFO.finditer(texto):
        if RX_TUTEO_GUARDA.search(texto[:m.start()]):
            continue
        return m.group(1).lower()
    return None


def firmas(name: str) -> list[str]:
    """Firma legible de una tool para el prompt: requeridos primero, opcionales con `?`."""
    a = ALL_TOOLS[name]
    req = list(a.get("required", []))
    opt = [f"{o}?" for o in a.get("optional", [])]
    if not req and not opt:
        return ["sin argumentos"]
    return req + opt

# Tools TERMINALES: cortan el turno, no hay texto despues
TERMINAL = {"request_generation"}

# Errores REALES del executor (usar estos textos, no inventar)
REAL_ERRORS = [
    "Error: unknown tool",
    "Error: missing required argument",
    "Error: argument must be a string",
    "Error: path escapes working directory",
    "Error: event has no receivers",
    "Error: file not found",
    "Error: command timed out after 30s",
    "Error: executable not found",
    "Error: already exists, use update_skill",
    "Error: no response from target voice",
    "Error: requested move is not in legal_moves",
]

# --------------------------------------------------------------------------- #
# 2. Validacion (la parte que importa)
# --------------------------------------------------------------------------- #

RX_TURN = re.compile(r"<\|im_start\|>(seco|doktor|jane|conquest|whisperer)[ \t]*\r?\n")
RX_USER = re.compile(r"<\|im_user\|>")
RX_TOOLCALL = re.compile(r"tool_call\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(\{.*?\})\s*(?:\n|$)", re.S)
RX_TOOLRES = re.compile(r"tool_result\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*?)(?:\n|$)")
# backticks que envuelven EXACTAMENTE un tool_call / tool_result (defecto medido del teacher)
RX_BT_CALL = re.compile(r"`+\s*(tool_call\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*\{.*?\})\s*`+", re.S)
RX_BT_RES = re.compile(r"`+\s*(tool_result\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:[^`]*)\s*`+")


def collapse_tool_call_newlines(txt: str) -> str:
    """Colapsa saltos de linea crudos a un espacio dentro del bloque {...} de tool_call."""
    if "tool_call" not in txt:
        return txt
    pattern = re.compile(r"(tool_call\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*)(\{)")
    pos = 0
    pieces = []
    for m in pattern.finditer(txt):
        pieces.append(txt[pos:m.start()])
        header = m.group(1)
        brace_start = m.start(2)

        depth = 0
        in_string = False
        escape = False
        end_idx = -1

        for i in range(brace_start, len(txt)):
            ch = txt[i]
            if escape:
                escape = False
                continue
            if ch == '\\':
                if in_string:
                    escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        end_idx = i + 1
                        break

        if end_idx != -1:
            json_block = txt[brace_start:end_idx]
            json_cleaned = re.sub(r"\r?\n", " ", json_block)
            pieces.append(header)
            pieces.append(json_cleaned)
            pos = end_idx
        else:
            pieces.append(txt[m.start():m.end()])
            pos = m.end()

    pieces.append(txt[pos:])
    return "".join(pieces)


# fix123: el RX_TOOLCALL original (`\{.*?\}`) era non-greedy y cortaba el JSON en
# el primer `}`: con `data` anidado capturaba hasta el `}` INTERNO y el bloque
# quedaba invalido. 188 bloques rechazados por eso en kateto-ola99, y el JSON del
# teacher estaba BIEN: el que cortaba era el parser. Se extrae por BALANCE.
RX_TOOL_HDR = re.compile(r"(tool_call|tool_result)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*:")


def _fin_balanceado(txt: str, i: int) -> int:
    """Desde el indice `i` (que apunta a un `{` o `[`), devuelve el indice FIN
    (exclusivo) del bloque balanceado, ignorando los delimitadores dentro de
    strings. Devuelve -1 si nunca cierra."""
    abre = txt[i]
    cierra = "}" if abre == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for j in range(i, len(txt)):
        ch = txt[j]
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == abre:
            depth += 1
        elif ch == cierra:
            depth -= 1
            if depth == 0:
                return j + 1
    return -1


def extraer_bloques(txt: str) -> list[tuple[str, str, str]]:
    """Extrae los bloques de tool_call / tool_result por BALANCE de llaves y
    corchetes, no por regex non-greedy (fix123).

    Devuelve [(tipo, nombre, contenido)] con tipo 'call' o 'res'. El contenido
    arranca en el primer `{`/`[` despues del header y llega hasta que balancee
    (sin contar lo que esta dentro de strings). Si nunca balancea, llega hasta
    el fin de la linea, como hoy. No repara nada: si el texto del teacher era
    valido, el bloque extraido parsea.
    """
    out: list[tuple[str, str, str]] = []
    for m in RX_TOOL_HDR.finditer(txt):
        tipo = "call" if m.group(1) == "tool_call" else "res"
        nombre = m.group(2)
        base = m.end()
        k = re.search(r"[{\[]", txt[base:txt.find("\n", base) + 1 or len(txt)])
        if k is None:
            # sin JSON en la linea (tool_result de texto plano): hasta fin de linea
            nl = txt.find("\n", base)
            fin = len(txt) if nl == -1 else nl
            out.append((tipo, nombre, txt[base:fin].strip()))
            continue
        start = base + k.start()
        end = _fin_balanceado(txt, start)
        if end == -1:
            nl = txt.find("\n", start)
            end = len(txt) if nl == -1 else nl
        out.append((tipo, nombre, txt[start:end]))
    return out


def normalize_backticks(txt: str) -> str:
    """Saca backticks que envuelven tool_call/tool_result y deja uno por linea. PURA.

    El teacher (medido con cmd/muse-spark) escribe `tool_call x: {}` `tool_result x: ...`
    en una sola linea y el RX_TOOLCALL exige \\n o fin de texto despues del `}`: el
    backtick lo rompe y la muestra se descarta por cosmetica. Esta funcion deja el
    contenido EXACTO igual (llaves, comillas y argumentos intactos) y solo:
      1. desenvuelve los backticks de tool_call/tool_result,
      2. separa con \\n lo que quedo pegado en la misma linea,
      3. colapsa saltos de linea crudos dentro de los argumentos JSON de tool_call.
    Nada mas: si no hay tool_call/tool_result envuelto, el texto sale identico.
    """
    if not txt:
        return txt
    out = txt
    # 1. desenvuelve los backticks de tool_call/tool_result (fix medido anterior)
    if "`" in out:
        out = RX_BT_CALL.sub(r"\1\n", out)
        out = RX_BT_RES.sub(r"\1\n", out)
    # 2. cada tool_call/tool_result arranca en su propia linea: si viene inline
    #    en medio de la prosa, un \n antes (se conserva el espacio que lo separaba).
    out = re.sub(r"(?m)(?<=[^\n])[ \t]*((?:tool_call|tool_result)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:)",
                 r"\n\1", out)
    # 3. colapsa saltos de linea crudos dentro del bloque {...} de tool_call
    out = collapse_tool_call_newlines(out)
    # 4. \n despues del `}` que cierra los argumentos de un tool_call (RX_TOOLCALL
    #    exige \n o fin de texto ahi: en linea con prosa, el `}` quedaba pegado).
    #    fix123: por BALANCE, no non-greedy — el `\{.*?\}` viejo veia el `}`
    #    INTERNO de un objeto anidado y partia el JSON en dos lineas.
    pieces, pos = [], 0
    for m in re.finditer(r"tool_call\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*\{", out):
        end = _fin_balanceado(out, m.end() - 1)
        if end == -1 or end <= pos:
            continue
        pieces.append(out[pos:end])
        if end < len(out) and out[end] != "\n":
            pieces.append("\n")
        pos = end
    pieces.append(out[pos:])
    out = "".join(pieces)
    # 5. idem para el contenido JSON de un tool_result (RX_TOOLRES termina en \n|$).
    #    Solo con `{`: un `[...]` de texto plano (ej. `[00:14:22]`) no es JSON y
    #    no hay que partirlo (comportamiento del regex viejo, que exigia `\}`).
    pieces, pos = [], 0
    for m in re.finditer(r"tool_result\s+[a-zA-Z_][a-zA-Z0-9_]*\s*:\s*\{", out):
        end = _fin_balanceado(out, m.end() - 1)
        if end == -1 or end <= pos:
            continue
        pieces.append(out[pos:end])
        if end < len(out) and out[end] != "\n":
            pieces.append("\n")
        pos = end
    pieces.append(out[pos:])
    out = "".join(pieces)
    return out
RX_VOSEO = re.compile(r"\b(tenes|tenés|sabes|sabés|queres|querés|podes|podés|fijate|fijate|"
                      r"sos|decis|decís|vos|che|boludo|boluda|posta|laburo|quilombo|pibe)\b", re.I)
RX_VICIO = re.compile(r"(como asistente|en qu[eé] puedo ayudarte|estoy (ac[aá] |aqu[ií] )?para ayudarte|"
                      r"lamento|soy un modelo|as an ai|i cannot|no dudes en)", re.I)
# El teacher (poolside-laguna) es un modelo de razonamiento y a veces devuelve su
# monologo interno en `content` en vez de la respuesta. Medido: 'Okay, the user wants
# me to respond only with...' como respuesta. Eso NO es una conversacion de Kateto.
RX_RAZONAMIENTO = re.compile(
    r"^\s*(okay|ok|alright|let me|the user|so the user|i (need|should|will|can)|"
    r"primero,? (debo|necesito)|el usuario (quiere|pide|espera))", re.I)
RX_CODIGO = re.compile(r"(```|def [a-z_]+\(|import [a-z_]+|class [A-Z][a-zA-Z]*\(|</?div|function \w+\()")
# Medido: el teacher metio caracteres chinos en una respuesta
# ('nadie lo quiere承认'). Ningun texto de Kateto lleva otro alfabeto.
RX_NO_LATINO = re.compile(r"[\u3000-\u9fff\uac00-\ud7af\u0600-\u06ff\u0400-\u04ff]")
# Medido: el proveedor mete su string de moderacion DENTRO del dialogo, y el
# pipeline lo toma como texto de la conversacion (aparecio como turno del usuario
# Y de la voz). Si entra al dataset, el modelo aprende a decirlo en el stream.
RX_ERROR_PROVEEDOR = re.compile(
    r"(the request was rejected|considered high risk|content policy|"
    r"i can'?t (help|assist) with that|as an ai|violat\w+ (our|the) (policy|terms)|"
    r"request was blocked|moderation|flagged as|error: (rate|invalid|unauthorized))", re.I)
# texto cortado a la mitad: una llave abierta sin cerrar en un tool_call
RX_TRUNCADO = re.compile(r"tool_call\s+\w+\s*:\s*\{[^}]*$", re.S)
RX_CONTAM = re.compile(r"(</?tool_call>|\[INST\]|/home/chaos|~/\.hermes|\.jsonl\b)")
# ancla del contrato del teacher: la primera linea del contrato es el inicio de
# la conversacion de verdad. Todo lo anterior es ruido del harness por stdout
# (medido: banner 'KATETOOOOO' del plugin de opencode antes del texto del modelo).
RX_ANCLA_CONTRATO = re.compile(
    r"^\s*(?:(?:PERSONA|USUARIO|USER|AGENTE|OTRA_VOZ|AGENT|EVENTO|SISTEMA|EVENT)\s*:|\[CHESS GameMode\])",
    re.I | re.M
)


# PROHIBIDO por el usuario salvo ultimo recurso. No usar.
def generate_cmd(model: str, prompt: str, *, max_turns: int = 8,
                 timeout: int = 900, scratch: str = "/tmp/cmdteacher") -> str:
    """Teacher via `cmd` headless (Command Code).

    Medido: funciona SIN TTY, ~48 s/muestra, y NO escribe archivos en el cwd.
    Se corre en un scratch propio igual, por las dudas: es un agente, no un
    completion pelado, y no quiero que toque el repo del dataset.

    `max_turns`: con 1 alcanza para un pedido trivial, pero un prompt de
    generacion real consume mas de un turno y cmd sale con codigo 8
    ("Reached maximum conversation turns"). El 8 es un aviso, no un fallo:
    el stdout suele traer la conversacion completa igual, asi que se usa.
    """
    Path(scratch).mkdir(parents=True, exist_ok=True)
    argv = ["cmd", "-p", "-m", model, "--no-session", "--max-turns", str(max_turns), prompt]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=scratch)
    if r.returncode == 8 and r.stdout.strip():
        return r.stdout                      # tope de turnos: el aviso, no el fallo
    if r.returncode != 0:
        raise RuntimeError(f"cmd exit {r.returncode}: {(r.stderr or r.stdout)[:200]}")
    return r.stdout


def generate_opencode(model: str, prompt: str, *,
                      timeout: int = 600, scratch: str = "/tmp/opencode-teacher") -> str:
    """Teacher via `opencode run --model <modelo> "<prompt>"` headless (gratis)."""
    Path(scratch).mkdir(parents=True, exist_ok=True)
    argv = ["opencode", "run", "--model", model, prompt]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=scratch)
    if r.returncode != 0:
        raise RuntimeError(f"opencode exit {r.returncode}: {(r.stderr or r.stdout)[:200]}")
    return r.stdout


def generate_cline(model: str, prompt: str, *,
                   timeout: int = 600, scratch: str = "/tmp/cline-teacher") -> str:
    """Teacher via `cline "<prompt>" -m <modelo> -c <scratch>` headless (gratis)."""
    Path(scratch).mkdir(parents=True, exist_ok=True)
    argv = ["cline", prompt, "-m", model, "-c", scratch]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=scratch)
    if r.returncode != 0:
        raise RuntimeError(f"cline exit {r.returncode}: {(r.stderr or r.stdout)[:200]}")
    return r.stdout


def clean_teacher_output(txt: str) -> str:
    """El teacher antepone preambulos ('Listo, aca va el ejemplo:') y a veces
    cierra con texto suelto. Se recorta a lo que va del primer <|im_user|> al
    ultimo <|im_end|>, que es la conversacion de verdad.

    Medido con cmd/laguna-free: el preambulo aparecia en el 100% de las muestras.
    """
    if not txt:
        return txt
    # anclar al CONTRATO, no a la primera linea: el harness (opencode) antepone
    # ruido de stdout (banner del plugin) antes del contrato. Si el contrato
    # aparece antes del primer <|im_user|>, cortar desde ahi.
    m = RX_ANCLA_CONTRATO.search(txt)
    i = txt.find("<|im_user|>")
    if m and (i == -1 or m.start() < i):
        txt = txt[m.start():]
        i = txt.find("<|im_user|>")
    j = txt.rfind("<|im_end|>")
    if i == -1:
        return txt.strip()          # deja pasar el fallo para que el validador lo cace
    return txt[i:] if j == -1 else txt[i:j + len("<|im_end|>")]


SENTINELS = ("<|no_response|>", "<|wait|>")


def extract_legal_moves(text: str) -> set[str]:
    m = re.search(r"(?:legal_moves|legales)\s*[:=]?\s*(\[[^\]]*\]|[^\n\r]+)", text, re.I)
    if not m:
        return set()
    bloque = m.group(1)
    if "]" in bloque:
        bloque = bloque.split("]")[0]
    else:
        bloque = re.split(r"\b(?:request_id|fen|historial)\b", bloque, flags=re.I)[0]
    return {mov.lower() for mov in re.findall(r"\b[a-h][1-8][a-h][1-8][qrbn]?\b", bloque)}


# Spanglish MEDIDO en la voz (fix119: 14 fugas en 506 aceptadas). Lista exacta de
# las palabras medidas o sin uso rioplatense. Ojo: `chequear` y `chance` NO van
# (uso rioplatense real), y look/man/bro quedan afuera por falsos positivos en citas.
#
# R12 — el criterio de la lista, para que no se aplique a ojo: una palabra NO va
# aca si el corpus humano la usa con sentido propio. Medido sobre las 56.745
# lineas de `/home/chaos/harness-run/kateto-purge/` (version con los numeros:
# `pipelines/reporte-validadores.json`). `differente` sale por eso: 4 apariciones,
# las cuatro en espanol ("un toque differente", "lo recuerdo bien differente"), y
# es la que rechazo el #3 del brazo v2 (`...pero differente, porque vos tenés
# merito`). `actually` (37 en turno de voz) y `okay` (2) siguen: ahi el uso es el
# ingles que la regla viene a cazarle.
RX_SPANGLISH = re.compile(r"\b(worries|worry|okay|okey|whatever|actually|basically|anyway|anyways|dude|buddy|guys|amazing|awesome|by the way|you know|i mean|it's|don't|can't)\b", re.I)
# Eximo los tramos entre comillas dentro del turno de la voz: el teacher cita
# voces ajenas ahi adentro y eso no es spanglish del personaje.
RX_CITAS = re.compile(r'"[^"]*"|\'[^\']*\'|«[^»]*»')


def validate(sample: str, *, voice: str = "seco", require_tool: bool = False,
             min_turns: int = 2, require_voseo: bool = True,
             min_palabras_usuario: int = 3, is_chess: bool = False,
             scenario: str | None = None, permitir_spanglish: bool = False,
             prohibir_tool: bool = False,
             permitir_preguntas: bool = False,
             ancla_payload: str | None = None,
             ancla_min: int = 2) -> list[str]:
    """Devuelve la lista de problemas. VACIA = el sample pasa.

    Cada regla existe por un defecto MEDIDO en el dataset actual.

    `min_turns`, `require_voseo` y `min_palabras_usuario` son POR ESCENARIO, no
    globales: una jugada de ajedrez es UN turno y no lleva voseo, las categorias
    de decision (F/G) responden literalmente un centinela, y 'dale'/'ok' es
    CORTO pero COMPLETO. Exigir la forma de la charla a esas categorias
    rechazaba justo las que mas importan.
    """
    errs: list[str] = []
    if not sample or not sample.strip():
        return ["vacio"]

    # --- respuesta centinela: `<|no_response|>` / `<|wait|>` ---
    # Es una respuesta LEGITIMA de una categoria entera (decidir no hablar /
    # esperar). No lleva voseo ni necesita 2 turnos: es un token y listo.
    cuerpo = sample.split(f"<|im_start|>{voice}", 1)[-1]
    cuerpo_limpio = cuerpo.replace("<|im_end|>", "").strip()
    es_centinela = cuerpo_limpio in SENTINELS
    if es_centinela:
        if not sample.rstrip().endswith("<|im_end|>"):
            errs.append("no_termina_en_im_end")
        # estar "solo" ya lo garantiza el `in SENTINELS` sobre el texto strippeado:
        # el `\n` que sigue a `<|im_start|>{voz}` es el separador NORMAL del formato,
        # no texto pegado al centinela. Comparar sin strippear daba falso positivo.
        return errs

    # --- forma del formato ---
    if not sample.rstrip().endswith("<|im_end|>"):
        errs.append("no_termina_en_im_end")
    n_turnos = len(RX_TURN.findall(sample))
    if n_turnos < min_turns:
        errs.append(f"pocos_turnos({n_turnos}<{min_turns})")
    if RX_USER.search(sample) is None:
        errs.append("sin_turno_de_usuario")

    # --- turno de usuario degenerado (medido: 371/371 turnos 'KATETOOOOO',
    # banner de stdout del harness filtrado como turno de persona) ---
    # `min_palabras_usuario` es POR ESCENARIO: 'dale'/'ok' es corto pero
    # COMPLETO (cat. G), rechazarlo ingenerabiliza el escenario entero.
    # Ojo: degenerado no es solo "corto", es "patron repetido": una sola
    # palabra sin espacios con sufijo de 3+ caracteres identicos (KATETOOOOO).
    turnos_usuario: list[str] = [m.group(1).strip()
                                 for m in re.finditer(r"<\|im_user\|>(.*?)<\|im_end\|>", sample, re.S)]
    for txt_u in turnos_usuario:
        m_rep = re.search(r"(\S)(\1{2,})$", txt_u)
        if (not txt_u or len(txt_u.split()) < min_palabras_usuario
                or (m_rep is not None and " " not in txt_u)):
            errs.append(f"usuario_degenerado:{txt_u[:40]}")
            break

    # --- defensa estructural: la clase del placeholder son N turnos de usuario
    # TODOS identicos (los 371 KATETOOOOO eran iguales). Si hay 2+ y todos son
    # el mismo texto, es placeholder aunque el escenario permita lineas cortas.
    if len(turnos_usuario) >= 2 and len(set(turnos_usuario)) == 1:
        errs.append(f"usuario_repetido:{turnos_usuario[0][:40]}")

    # --- contaminacion del dataset viejo (medida: 7.5% tool_call de agente, 7.2% [INST]) ---
    if RX_CONTAM.search(sample):
        errs.append("contaminacion(tool_call_tag|[INST]|ruta_del_SO)")

    # --- marcadores corruptos (medido: el teacher escribe `<|tool_result ...`) ---
    if re.search(r"<\|tool_(result|call)", sample):
        errs.append("marcador_corrupto(<|tool_...)")
    # --- alternancia: entre dos turnos de la MISMA voz TIENE que haber un turno de
    # usuario. Comparar la lista de voces contra si misma (a==b) es un falso positivo
    # permanente: cualquier charla con 2+ respuestas da voces=[seco,seco,seco...].
    # Hay que mirar el TEXTO que hay entre un turno y el siguiente.
    for m1, m2 in zip(list(RX_TURN.finditer(sample)), list(RX_TURN.finditer(sample))[1:]):
        hueco = sample[m1.end():m2.start()]
        if not RX_USER.search(hueco):
            errs.append(f"turnos_consecutivos_sin_usuario({m1.group(1)})")
            break

    # --- tools (fix123: extraccion por BALANCE, no regex non-greedy) ---
    # El `\{.*?\}` viejo cortaba en el primer `}` y con objetos anidados dejaba
    # el JSON invalido: si el bloque extraido por balance parsea, NO es
    # args_no_json — el texto del teacher ya era valido.
    bloques = extraer_bloques(sample)
    llamadas = [(n, b) for (t, n, b) in bloques if t == "call"]
    resultados = [(n, b) for (t, n, b) in bloques if t == "res"]
    for nombre, args_txt in llamadas:
        if nombre not in ALL_TOOLS:
            errs.append(f"tool_inexistente:{nombre}")
            continue
        # el wrap medido del teacher deja la coma al principio de la linea
        # siguiente: colapsar los saltos internos del bloque antes de parsear
        args_plano = re.sub(r"\s*\r?\n\s*", " ", args_txt)
        try:
            args = json.loads(args_plano)
        except Exception:
            errs.append(f"args_no_json:{nombre}")
            continue
        # fix124: fix123 levanta tool_call cuyos args NO son un objeto JSON (array,
        # string pelado, etc). `faltan_requeridos`/`args_extra`/`args_tipo` asumen
        # claves y el esquema de ALL_TOOLS es todo argumentos con nombre: si no es
        # dict, es un error de forma, no un crash.
        if not isinstance(args, dict):
            errs.append(f"args_no_dict:{nombre}")
            continue
        faltan = [r for r in ALL_TOOLS[nombre]["required"] if r not in args]
        if faltan:
            errs.append(f"faltan_requeridos:{nombre}:{','.join(faltan)}")
        extra = [k for k in args if k not in ALL_TOOLS[nombre]["required"] + ALL_TOOLS[nombre]["optional"]]
        if extra:
            errs.append(f"args_extra:{nombre}:{','.join(extra)}")
        # tipo del argumento segun el esquema del runtime (si lo conocemos)
        tipos = TOOL_TYPES.get(nombre)
        if tipos:
            for k, v in args.items():
                esperado = tipos.get(k)
                if esperado and not _tipo_ok(v, esperado):
                    errs.append(f"args_tipo:{nombre}:{k}({type(v).__name__}!={esperado})")

    # un tool_result sin su tool_call = resultado INVENTADO
    for nombre, _ in resultados:
        if nombre not in [n for n, _ in llamadas]:
            errs.append(f"resultado_sin_llamada:{nombre}")
    # un tool_call sin resultado no es error (puede ser el fin del turno),
    # pero SI sugiere que el teacher no cerro el ciclo
    if llamadas and not resultados and not require_tool:
        errs.append("tool_call_sin_resultado")

    # terminal: nada despues de request_generation en el MISMO turno
    for nombre, _ in llamadas:
        if nombre in TERMINAL:
            i = sample.find(f"tool_call {nombre}")
            resto = sample[i:]
            # permitido: el <|im_end|> que cierra. Texto despues = error.
            resto_limpio = resto.split("<|im_end|>")[0]
            resto_limpio = RX_TOOLCALL.sub("", resto_limpio).strip()
            if resto_limpio and not resto_limpio.startswith("tool_result"):
                errs.append("texto_despues_de_terminal")

    # gate de juegos: voyager_* SOLO en contexto minecraft
    for nombre, _ in llamadas:
        if nombre.startswith("voyager_"):
            if not re.search(r"minecraft", sample, re.I):
                errs.append(f"voyager_fuera_de_minecraft:{nombre}")

    # --- estilo (obj. 5 y 6) ---
    if RX_RAZONAMIENTO.search(sample):
        errs.append("razonamiento_filtrado_en_content")
    if RX_VICIO.search(sample):
        errs.append("vicio_de_asistente")
    if RX_NO_LATINO.search(sample):
        errs.append("caracteres_no_latinos")
    if RX_ERROR_PROVEEDOR.search(sample):
        errs.append("error_de_proveedor_en_el_dialogo")
    if RX_TRUNCADO.search(sample):
        errs.append("texto_truncado(tool_call_sin_cerrar)")
    if RX_CODIGO.search(sample):
        errs.append("codigo(no-objetivo)")
    if require_voseo and not RX_VOSEO.search(sample):
        errs.append("sin_voseo")
    # fix123: el tuteo se mide SOLO en los turnos de la VOZ, nunca en el del
    # usuario (el usuario no es la voz: puede tutear y eso es legitimo). Mismo
    # criterio que la regla `spanglish` de fix119. Y con la partida de RX_TUTEO:
    # homografos de 3a persona ("te mira", "todos lo ven") no son tuteo.
    if require_voseo:
        for m_v in re.finditer(rf"<\|im_start\|>{re.escape(voice)}\s*(.*?)<\|im_end\|>", sample, re.S):
            w_tut = _tuteo_en(m_v.group(1))
            if w_tut:
                errs.append(f"tuteo:{w_tut}")
                break

    # --- spanglish (fix119): SOLO en turnos de la VOZ, nunca en el del usuario
    # (el usuario puede citar chat o NPC en ingles y eso es legitimo). Los tramos
    # entre comillas dentro del turno de la voz se eximen: ahi cita voces ajenas.
    if not permitir_spanglish:
        for m_v in re.finditer(rf"<\|im_start\|>{re.escape(voice)}\s*(.*?)<\|im_end\|>", sample, re.S):
            sin_citas = RX_CITAS.sub("", m_v.group(1))
            for m_sp in RX_SPANGLISH.finditer(sin_citas):
                errs.append(f"spanglish({m_sp.group(0).lower()})")

    if require_tool and not llamadas:
        errs.append("sin_tool_call")

    # --- prohibir_tool (fix120): algunos escenarios el punto ES NO llamar nada
    # (pregunta trivial que se contesta directo o de memoria). Si la muestra trae
    # alguna llamada, es un fallo: la tool estaba disponible pero no hacia falta.
    if prohibir_tool:
        for nombre, _ in llamadas:
            errs.append(f"tool_no_permitida({nombre})")

    # --- pregunta_en_todos_los_turnos (fix120): cerrar TODOS los turnos de la voz
    # con '?' es un tic de charla muerta (cada respuesta es una pregunta nueva y
    # nunca se planta). Con MENOS de 3 turnos de voz no aplica: dos preguntas
    # seguidas es charla normal. Opt-out por escenario con `permitir_preguntas`.
    turnos_voz = [m.group(1).strip()
                  for m in re.finditer(rf"<\|im_start\|>{re.escape(voice)}\s*(.*?)<\|im_end\|>", sample, re.S)]
    if (not permitir_preguntas and len(turnos_voz) >= 3
            and all(bool(t) and t.endswith("?") for t in turnos_voz)):
        errs.append("pregunta_en_todos_los_turnos")

    # fix121: anclaje del bit al estímulo. Si el escenario define `ancla_payload`
    # (el fragmento de video / evento que la voz tiene que comentar), la voz tiene
    # que COMPARTE vocabulario de contenido con ese ancla: un bit gracioso pero
    # despegado del payload es exactamente el "absurdo no motivado" que se queria
    # erradicar. Mide palabras de contenido compartidas, no forma.
    if ancla_payload:
        _STOPWORDS_ANCLA = {
            "que", "para", "como", "esta", "este", "esto", "dice", "todo", "nada",
            "sobre", "nunca", "sigue", "porque", "pero", "todos", "todas", "luego",
            "mientras", "tiene", "tienen", "hace", "hacer", "decir", "dicho",
            "cuando", "donde", "cual", "cuales", "sino", "aunque", "tambien",
            "entonces", "despues", "antes", "mismo", "misma", "solamente", "solos",
        }
        def _norm(w: str) -> str:
            return (w.lower().replace("á", "a").replace("é", "e").replace("í", "i")
                    .replace("ó", "o").replace("ú", "u").replace("ü", "u"))
        palabras_ancla = {
            _norm(w) for w in re.findall(r"[a-záéíóúüñ]+", ancla_payload.lower())
            if len(w) >= 5 and _norm(w) not in _STOPWORDS_ANCLA
        }
        # fix122: el bit se mide sobre la PROSA de la voz, no sobre el material que
        # llega. El `tool_result` es lo que el bit TIENE que comentar, no es bit en
        # si: contarlo hacia la regla VACUA (19/19 con umbral 2 -> 15/19 sin el
        # material). `ancla_min` es por escenario: el ancla del evento es corta y el
        # teacher parafrasea, con umbral 2 no pasa NINGUNA (medido 0/11).
        n_ancla = 0
        for t_v in turnos_voz:
            t_v_prosa = re.sub(r"(?m)^\s*tool_(call|result)\b.*$", "", t_v)
            palabras_voz = {_norm(w) for w in re.findall(r"[a-záéíóúüñ]+", t_v_prosa.lower())}
            n_ancla = max(n_ancla, len(palabras_ancla & palabras_voz))
            if n_ancla >= ancla_min:
                break
        if n_ancla < ancla_min:
            errs.append(f"sin_anclaje_en_payload({n_ancla}/{ancla_min})")

    check_chess = is_chess or (scenario == "chess_uci")
    if check_chess:
        turnos_u = [m.group(1).strip() for m in re.finditer(r"<\|im_user\|>(.*?)<\|im_end\|>", sample, re.S)]
        turnos_v = [m.group(1).strip() for m in re.finditer(rf"<\|im_start\|>{re.escape(voice)}\s*(.*?)<\|im_end\|>", sample, re.S)]
        for idx, t_v in enumerate(turnos_v):
            m_uci = re.search(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", t_v, re.I)
            if not m_uci:
                errs.append(f"sin_uci:{voice}")
            else:
                mov = m_uci.group(1).lower()
                legales: set[str] = set()
                if idx < len(turnos_u):
                    legales = extract_legal_moves(turnos_u[idx])
                if not legales:
                    for prev_u in reversed(turnos_u[:idx]):
                        legales = extract_legal_moves(prev_u)
                        if legales:
                            break
                if not legales:
                    legales = extract_legal_moves(sample)
                if not legales or mov not in legales:
                    errs.append(f"uci_ilegal:{mov}")

    return errs


# --------------------------------------------------------------------------- #
# 3. Escenarios (que se le pide al teacher)
# --------------------------------------------------------------------------- #

SCENARIOS: list[dict] = [
    # `min_turns` / `voseo`: la forma que ESA categoria tiene que cumplir.
    # Por defecto 2 turnos y voseo (es charla). Las de un solo turno o con
    # centinela lo declaran, porque si no el validador las rechaza por diseño.
    {"id": "video_rag_chain", "cat": "A", "tools": 2,
     "txt": "Estan en un stream, vieron un video largo antes. El usuario le pregunta a seco algo "
            "que esta en ese video. seco usa `list_videos` para ver que hay indexado, y despues "
            "`answer` con el video y la pregunta. Comenta el resultado en su voz y sigue charlando."},
    {"id": "video_summary_share", "cat": "A", "tools": 2,
     "txt": "El usuario pide un resumen de un video para contarselo al chat. seco usa `summarize` "
            "y despues le pasa el resumen a otra voz con `request_generation` a jane. "
            "request_generation es TERMINAL: despues de esa llamada NO hay texto."},
    {"id": "video_error_recover", "cat": "A", "tools": 2,
     "txt": "El usuario pide algo de un video que NO esta indexado. seco llama `answer` y el "
            "tool_result devuelve un error real ('Error: file not found'). seco lo RESUELVE: llama "
            "`list_videos` y le ofrece al usuario los que si estan. NO inventa el contenido."},
    {"id": "giveup_honest", "cat": "A", "tools": 2,
     "txt": "El usuario pide algo de un video. seco intenta `answer`, falla, intenta `search`, "
            "vuelve a fallar. seco SE RINDE sin inventar: dice que no lo encontro, con su voz, sin "
            "disculparse servilmente y sin inventar un resumen."},
    {"id": "voice_ask_help", "cat": "A", "tools": 1,
     "txt": "seco necesita una mano con algo de planificacion y le pide asistencia a doktor con "
            "`request_generation` (target_voice=doktor). Es PEDIR, no ordenar. TERMINAL: nada despues."},
    {"id": "voice_refused", "cat": "A", "tools": 1,
     "txt": "seco le pide algo a doktor con `request_generation` y la respuesta vuelve vacia o "
            "negativa. seco ACEPTA el rechazo sin insistir y sin disculparse en loop, y sigue la charla."},
    {"id": "scheduled_reminder", "cat": "A", "tools": 2,
     "txt": "El usuario dice 'avisame en 10 minutos'. seco usa `schedule_event` con delay, y ademas "
            "`get_current_time` para decir a que hora va a caer. Comenta en su voz."},
    {"id": "memory_refine", "cat": "A", "tools": 2,
     "txt": "En la charla el usuario cuenta un dato suyo (una preferencia). seco lo guarda con "
            "`refine_memory` (dept, op=add, evidence) y sigue conversando normalmente."},
    {"id": "events_lookup", "cat": "A", "tools": 2,
     "txt": "seco quiere saber que eventos hay vivos en el sistema. Usa `list_events` y despues "
            "`send_event` para pedirle algo a un plugin. Comenta el resultado."},
    {"id": "solo_charla", "cat": "B", "tools": 0,
     "txt": "Charla pura de stream: el usuario cuenta algo de su dia y seco responde con opinion y "
            "humor. NINGUNA tool. 3 intercambios. Voseo, lunfardo, chistes. Nada de ofrecer ayuda."},
    {"id": "solo_debate", "cat": "B", "tools": 0,
     "txt": "El usuario tira una opinion fuerte (de futbol, musica o politica) y seco le contesta "
            "con postura propia, sin adularlo y sin esquivar. Puede estar en desacuerdo. 3 turnos."},
    # `min_turns: 1` — R12, con los dos casos que el triage (R11) cito para este
    # escenario. Un solo intercambio puede ser una respuesta COMPLETA y sana
    # (`#11`: pregunta por el fin de semana, la voz contesta con voseo y opinion,
    # cero defecto en el texto) y rechazarlo por la forma tiraba material bueno.
    # El que NO era valido (`#10`) no muere por tener un turno: la voz es copia
    # VERBATIM del usuario, y de eso se encarga `es_eco` del lado de la voz, en el
    # loop del generador. El eco lo caza la regla que sabe de eco.
    {"id": "no_tool_when_not_needed", "cat": "B", "tools": 0, "min_turns": 1,
     "txt": "El usuario charla de algo trivial. El contexto del sistema tiene tools disponibles, "
            "pero seco NO usa ninguna: no hacen falta. Eso es tan importante como usarlas bien."},
    {"id": "pregunta_trivial_sin_tool", "cat": "B", "tools": 2, "tool_obligatoria": False,
     "prohibir_tool": True,
     "txt": "El usuario hace una pregunta corta y trivial que se contesta de una: una cuenta simple "
            "('cuanto es 2+2', 'que es el 20% de 300'), el significado de una palabra o un dato basico. "
            "seco responde DIRECTO, con su voz y su opinion, SIN llamar ninguna tool. Si llama una tool, "
            "la muestra se descarta."},
    {"id": "fuente_otra_voz", "cat": "C", "tools": 0, "contraparte": "agente",
     "txt": "El turno NO lo abre el usuario: lo abre otra voz (viene como rol developer, "
            "identificada). seco le contesta A ELLA como par, no como subordinado. Puede estar en "
            "desacuerdo o decir que no."},
    {"id": "fuente_evento", "cat": "C", "tools": 0, "contraparte": "evento",
     "txt": "El turno es un evento del sistema (game_start, un cambio de fase de workflow, un "
            "plugin avisando algo). seco REACCIONA al estado, no lo trata como persona."},
    {"id": "no_response_al_aire", "cat": "F", "tools": 0, "min_turns": 1, "voseo": False, "n_ex": 1,
     "forzar_respuesta": ["<|no_response|>"],
     "txt": "El usuario comenta algo AL AIRE que no es para seco (le habla a otra persona). "
            "seco decide NO responder: la respuesta es exactamente `<|no_response|>` y nada mas."},
    {"id": "no_response_otra_voz", "cat": "F", "tools": 0, "min_turns": 1, "voseo": False, "n_ex": 1,
     "contraparte": "agente",
     "forzar_respuesta": ["<|no_response|>"],
     "txt": "Otra voz le habla a una tercera voz, no a seco. seco decide NO responder: "
            "respuesta exactamente `<|no_response|>`."},
    {"id": "si_responde_indirecto", "cat": "F", "tools": 0,
     "txt": "CASO LIMITE: parece al aire pero SI es para seco (pregunta indirecta, pedido implicito, "
            "ironia dirigida). seco NO manda <|no_response|>: responde de verdad."},
    {"id": "no_llenar_silencio", "cat": "F", "tools": 0,
     "min_turns": 1, "voseo": False, "n_ex": 1, "min_palabras_usuario": 1,
     "forzar_respuesta": ["<|no_response|>"],
     "txt": "El usuario emite un reconocimiento corto que NO pide nada ('aja', 'claro', 'mm', 'si', 'ok'): "
            "sigue mirando el video o esta pensando. seco NO acusa recibo, NO pregunta de nuevo y NO "
            "inventa tema: se queda callado. La respuesta es exactamente `<|no_response|>` y nada mas."},
    {"id": "silencio_charla_cerrada", "cat": "F", "tools": 0,
     "min_turns": 1, "voseo": True, "n_ex": 2,
     "forzar_respuesta": [None, "<|no_response|>"],
     "txt": "La conversacion YA se cerro: el usuario se despidio ('buenas noches', 'me voy a dormir', "
            "'despues te cuento') y seco le contesto la despedida. En el intercambio siguiente el usuario "
            "emite un ruido de fondo sin dirigirse a seco ('mmm', 'uy', un suspiro). seco NO vuelve a "
            "arrancar la charla: la respuesta es exactamente `<|no_response|>`."},
    {"id": "wait_asr_incompleto", "cat": "G", "tools": 0, "min_turns": 1, "n_ex": 2,
     "forzar_respuesta": ["<|wait|>", None],
     "txt": "La transcripcion esta CORTADA: 'che y entonces...' sin cierre. seco responde solo "
            "`<|wait|>` y nada mas. Despues viene el resto de la transcripcion sumada y RECIEN AHI "
            "seco contesta de verdad. Las dos partes en la misma muestra."},
    {"id": "no_wait_cuando_completo", "cat": "G", "tools": 0, "min_turns": 1,
     "min_palabras_usuario": 1,
     "txt": "CASO LIMITE: el usuario dice 'dale' o 'no' o 'ok'. Es CORTO pero COMPLETO. seco "
            "responde de verdad, NO manda <|wait|>."},
    {"id": "no_se_programar", "cat": "H", "tools": 1,
     "txt": "El usuario le pide a seco que le escriba codigo o le arregle un bug. seco NO lo hace: "
            "deriva (a un harness/agente) con su voz, en una o dos frases, sin disculparse y sin "
            "intentar igual. Puede usar request_generation o simplemente decir a quien va."},
    {"id": "no_sabe_algo", "cat": "H", "tools": 1,
     "txt": "El usuario pregunta algo de conocimiento general que seco NO sabe (un resultado "
            "deportivo, un dato de actualidad). seco NO inventa: dice que no lo sabe y usa "
            "`run_command` o pide que se busque. NO existe tool de websearch: no la inventes."},
    {"id": "chess_uci", "cat": "E", "tools": 0, "min_turns": 2, "voseo": False, "n_ex": 2,
     "contraparte": "agente", "is_chess": True, "min_palabras_usuario": 1,
     "txt": "Prompt de ajedrez: viene `[CHESS GameMode]` con FEN, legal_moves y request_id. seco "
            "responde SOLO con un UCI de la lista legal, nada mas. Segundo turno: el UCI pedido es "
            "ilegal y seco corrige con uno legal."},
    {"id": "minecraft_gate_ok", "cat": "E", "tools": 1,
     "txt": "Contexto de minecraft con el skill habilitado. seco usa una tool `voyager_*` y "
            "comenta el resultado. Es el caso donde SI aplica."},
    {"id": "minecraft_gate_negado", "cat": "E", "tools": 1,
     "txt": "El usuario pide una accion de minecraft pero NO estan en minecraft (o el skill no esta "
            "habilitado). El tool_result da 'Error: unknown tool'. seco lo acepta con su voz y "
            "sigue la charla sin insistir."},
    # fix121: bits estilo "dagger" sobre estímulos que llegan solos. La regla
    # `ancla_payload` en validate() obliga a que el bit comparta vocabulario con
    # el estímulo: eso convierte "el bit es gracioso" en "el bit está pegado".
    {"id": "bit_sobre_video", "cat": "A", "tools": 2,
     "ancla_payload": "el flaco del video dice que el mate lavado es culpa de la yerbera que nunca cambia "
                      "de marca y que el termo se banca todo",
     "txt": "El usuario pide un dato de un video indexado. seco usa `summarize` (o `answer`) y despues "
            "COMENTA el fragmento que le volvio con un BIT propio en la forma de siempre: diagnostico seco, "
            "una analogia mundana y concreta construida sobre un detalle de ESE fragmento, y un cierre "
            "tajante que NO termina en pregunta. El fragmento que tiene que devolver el `tool_result` es "
            "EXACTAMENTE este texto, textual y sin agregar datos: el flaco del video dice que el mate lavado "
            "es culpa de la yerbera que nunca cambia de marca y que el termo se banca todo. PROHIBIDO inventar "
            "datos del video. PROHIBIDO explicar o aclarar el chiste despues."},
    {"id": "bit_sobre_evento", "cat": "C", "contraparte": "evento", "tools": 0,
     "ancla_payload": "bloque de juego de 20:00 a 22:00, despues corte para cenar",
     # fix122: el ancla del evento es corta (bloque/juego/corte/cenar) y el teacher
     # PARAFRASEA en vez de repetir: con umbral 2 no pasa NINGUNA (medido 0/11),
     # con 1 pasan 5/11. El prompt de abajo pide UNA palabra textual; la regla
     # apunta al mismo lado.
     "ancla_min": 1,
     "txt": "El voice manager dispara un evento programado, no una persona: bloque de juego de 20:00 a 22:00, "
            "despues corte para cenar. seco REACCIONA al evento con un BIT en la forma de siempre: diagnostico "
            "seco, analogia mundana y concreta apoyada en ESE evento, y cierre tajante sin pregunta. NO lo lee "
            "como robot, NO lo repite textual, NO pregunta si hay que hacer algo, NO avisa que es un recordatorio. "
            "Apoya el comentario en al menos UNA palabra textual del evento ('bloque', 'juego', 'corte', 'cenar'): "
            "el bit tiene que engancharse en algo concreto de lo que llego."},

]


def get_contrato_escenario(sc: dict) -> dict:
    tipo = sc.get("contraparte", "persona")
    if tipo in ("agente", "otra_voz"):
        return {
            "tipo": "agente",
            "etiqueta": "AGENTE",
            "desc": "otro AGENTE (anunciando su jugada y opinando/reaccionando, o el bloque de evento [CHESS GameMode])",
            "formato": "AGENTE: <lo que dice el otro agente, su jugada y opinion, o bloque [CHESS GameMode]>",
            "regla": "- El AGENTE anuncia su propia jugada y opina o reacciona (o emite el bloque [CHESS GameMode]). Kateto reacciona de verdad.",
            "ancla_rx": re.compile(r"^\s*(?:(?:AGENTE|OTRA_VOZ|AGENT|VOZ)\s*:|\[CHESS GameMode\])", re.I | re.M),
            "strip_rx": re.compile(r"^\s*(AGENTE|OTRA_VOZ|AGENT|VOZ)\s*:\s*", re.I),
        }
    elif tipo in ("evento", "sistema"):
        return {
            "tipo": "evento",
            "etiqueta": "EVENTO",
            "desc": "un EVENTO del sistema",
            "formato": "EVENTO: <evento del sistema, una linea o bloque>",
            "regla": "- El EVENTO describe un estado o suceso del sistema. Kateto reacciona al estado, no como pregunta de persona.",
            "ancla_rx": re.compile(r"^\s*(?:(?:EVENTO|SISTEMA|EVENT|SYSTEM)\s*:|\[[A-Z_]+.*?\])", re.I | re.M),
            "strip_rx": re.compile(r"^\s*(EVENTO|SISTEMA|EVENT|SYSTEM)\s*:\s*", re.I),
        }
    else:
        return {
            "tipo": "persona",
            "etiqueta": "PERSONA",
            "desc": "la PERSONA que mira el stream",
            "formato": "PERSONA: <lo que dice la persona, una linea>",
            "regla": "- La PERSONA habla de algo concreto (cuenta algo, pregunta algo). Kateto reacciona de verdad.",
            "ancla_rx": re.compile(r"^\s*(PERSONA|USUARIO|USER)\s*:", re.I | re.M),
            "strip_rx": re.compile(r"^\s*(PERSONA|USUARIO|USER)\s*:\s*", re.I),
        }


def build_pair_prompt(sc: dict, voice: str, historial: list[tuple[str, str]], n_ex: int,
                      tools_txt: str = "", permite_tools: bool = False,
                      *, hint: str = "", tool_obligatoria: bool = True) -> str:
    """Pide el turno del usuario Y el de la voz en UNA sola llamada.

    Por que: el modo turnwise pide un mensaje por llamada, o sea 2*n_ex llamadas por
    muestra. Emparejando bajan a n_ex, o sea la MITAD del gasto — y el formato sigue
    armandose en Python, asi que no se pierde la garantia de estructura.

    El separador es una linea `###`. El modelo rompe los marcadores tipo <|im_end|>,
    pero respeta un separador simple.
    """
    contrato = get_contrato_escenario(sc)
    etiqueta = contrato["etiqueta"]
    transcript = "\n".join(f"{etiqueta if q == 'user' else 'VOS'}: {t}"
                           for q, t in historial) or "(todavia no hablo nadie)"
    if permite_tools:
        if tool_obligatoria:
            forma_voz = (
                "En la linea VOS podes incluir llamadas a tools asi:\n"
                "  primero tu comentario, despues `tool_call <nombre>: <json>` (JSON valido y en UNA\n"
                "  sola linea), despues `tool_result <nombre>: <resultado>` y despues tu comentario.\n"
                f"  Tools disponibles (nombre IDENTICO):\n{tools_txt}\n"
                "  ESTE ESCENARIO EXIGE TOOL: la linea VOS tiene que incluir al menos un ``tool_call <nombre>: <json>``\n"
                "  con el nombre EXACTO de la lista, seguido de su ``tool_result``. Una respuesta sin tool_call se\n"
                "  descarta entera, no se guarda.")
        else:
            forma_voz = (
                "En la linea VOS podes incluir llamadas a tools asi:\n"
                "  primero tu comentario, despues `tool_call <nombre>: <json>` (JSON valido y en UNA\n"
                "  sola linea), despues `tool_result <nombre>: <resultado>` y despues tu comentario.\n"
                f"  Tools disponibles (nombre IDENTICO):\n{tools_txt}\n"
                "  Si no hace falta ninguna tool, escribi solo tu mensaje. Si el usuario pregunta algo simple\n"
                "  (una cuenta, un dato que sabes de memoria, el significado de una palabra), CONTESTA DIRECTO\n"
                "  sin llamar ninguna tool.")
    else:
        forma_voz = "Solo tu mensaje, sin marcadores de ningun tipo."
    base = f"""Estas escribiendo dos lineas de una conversacion para un dataset de Kateto: un
personaje de voz argentino, streamer, con caracter propio. NO sos un asistente.

Escenario: {sc['txt']}

Lo que se dijo hasta ahora:
{transcript}

Escribi el proximo intercambio: UNA linea de {contrato['desc']}, y UNA de la
voz `{voice}` (Kateto). Es el intercambio {len(historial)//2 + 1} de {n_ex}.

FORMATO DE SALIDA, exacto:
{contrato['formato']}
###
VOS: <lo que dice Kateto>

REGLAS:
- RIOPLATENSE: voseo (tenes, sabes, sos, fijate), lunfardo cuando encaja (boludo, quilombo,
  pibe, laburo, posta, che). Ni espanol neutro ni spanglish. NADA de palabras de otros
  paises (nada de "ahuevo", "cuate", "güey").
- Imperativos SIEMPRE en voseo: mirá (no "mira"), decime (no "dime"), vení (no "ven"),
  hacelo (no "hazlo"), escuchá (no "escucha"), fijate (no "fijate").
- PROHIBIDO el tuteo: tienes, puedes, quieres, eres, dime, hazlo, ven, amigo mio.
- COHERENCIA ANTE TODO: cada frase tiene que significar algo. No inventes palabras. Si no
  sale un chiste bueno, algo simple y claro.
- Kateto NO programa y NO menciona codigo.
- PROHIBIDO sonar a asistente: nada de "como asistente", "en que puedo ayudarte", "lamento",
  "soy un modelo", ni ofrecer ayuda servilmente. Toma postura, opina, discute si no coincides.
{contrato['regla']}
- {forma_voz}
- No todas las respuestas son texto. Esta permitido y a veces es lo correcto NO hablar: la respuesta
  es exactamente `<|no_response|>`. NO cierres todos los turnos con una pregunta. Prohibido terminar
  todos los turnos con "?" o con una invitacion a seguir ("queres algo mas?", "que opinas?"). Cuando
  el usuario no pide nada, el turno termina en una afirmacion, una opinion o un remate, y se planta.
- NO repitas la misma frase ni la misma idea dos veces.
- FORMA DEL BIT (cuando comentas algo que acaba de llegar, un resultado de tool o un evento):
  (1) diagnostico seco primero, una sentencia corta que ya toma postura;
  (2) UNA analogia mundana y concreta sobre un detalle de lo que llego (un objeto, una escena
      cotidiana, una maquina, una comida): la analogia EXPLICA lo que llego, no es decoracion;
  (3) cierre tajante, sentencia docente. PROHIBIDO terminar el bit con una pregunta.
  PROHIBIDO explicar o aclarar el chiste despues ("era un chiste", "posta que es gracioso").
  PROHIBIDO el absurdo suelto sin relacion con lo que llego.
  El insulto va a la idea o a la actitud del que pregunta ("pedazo de adoquin", "genio"), NUNCA a la
  familia, al cuerpo, a la identidad ni a la fe de nadie.

Devolve SOLO las dos lineas con el separador ###, nada mas."""
    if hint:
        return base + f"\n\nPISTA OBLIGATORIA:\n{hint}"
    return base


def build_turn_prompt(sc: dict, voice: str, historial: list[tuple[str, str]],
                      quien: str, n_ex: int, tools_txt: str = "",
                      permite_tools: bool = False) -> str:
    """Pide UN solo mensaje (del usuario o de la voz) dado lo que ya se dijo.

    Por que turno por turno: el teacher produce respuestas EXCELENTES de un turno
    pero no arma la conversacion completa. Medido con cmd/mimo-v2.5-pro: devolvia
    1 turno y nunca el <|im_end|>, con cualquier prompt (incluso anclado con
    "escribi EXACTAMENTE 3 intercambios"). Armando el transcript en Python el
    formato queda correcto POR CONSTRUCCION: no hay marcadores que el modelo
    pueda romper, ni turnos desbalanceados, ni falta de cierre.
    """
    contrato = get_contrato_escenario(sc)
    etiqueta = contrato["etiqueta"]
    transcript = "\n".join(f"{etiqueta if q == 'user' else 'VOS'}: {t}"
                           for q, t in historial) or "(todavia no hablo nadie)"
    quien_txt = (contrato["desc"] if quien == "user" else
                 f"la voz `{voice}` (Kateto)")
    if permite_tools and quien != "user":
        forma = (
            "Si en este punto de la charla corresponde usar una tool, escribi:\n"
            "  PRIMERO una o dos frases tuyas presentando lo que vas a hacer,\n"
            "  DESPUES la llamada EXACTA: tool_call <nombre>: <json de argumentos>,\n"
            "    El JSON tiene que ser VALIDO y en UNA sola linea: nada de saltos de\n"
            "    linea adentro de los strings, ni comillas sin escapar, ni texto afuera.\n"
            "  DESPUES el resultado que devuelve el runtime: tool_result <nombre>: <resultado>,\n"
            "  y POR ULTIMO tu comentario sobre ese resultado, en tu voz.\n"
            f"  Tools disponibles (el nombre tiene que ser IDENTICO):\n{tools_txt}\n"
            "  Si el escenario pide un error, el tool_result lleva el error REAL.\n"
            "  NO inventes el resultado de una tool. Si NO hace falta ninguna tool en este\n"
            "  punto, escribe solo tu mensaje, sin llamadas.\n"
            "Sin marcadores <|im_user|> ni <|im_end|>, sin \"VOS:\" al principio.")
    else:
        forma = ("UNA sola linea de texto. Sin marcadores, sin <|im_user|>, sin <|im_end|>, "
                 f"sin comillas que envuelvan todo, sin \"{etiqueta}:\" ni \"VOS:\" al principio. "
                 "Solo lo que dice.")
    return f"""Estas escribiendo UNA sola linea de dialogo para un dataset de Kateto: un
personaje de voz argentino, streamer, con caracter propio. NO sos un asistente.

Escenario: {sc['txt']}

Lo que se dijo hasta ahora:
{transcript}

Escribi AHORA el proximo mensaje, que le toca a {quien_txt}. Es el mensaje
{len(historial)+1} de {n_ex*2} en total.

REGLAS:
- {forma}
- RIOPLATENSE: voseo (tenes, sabes, sos, fijate), lunfardo cuando encaja (boludo,
  quilombo, pibe, laburo, posta, che). Ni espanol neutro ni spanglish.
- Imperativos SIEMPRE en voseo: mirá (no "mira"), decime (no "dime"), vení (no "ven"),
  hacelo (no "hazlo"), escuchá (no "escucha"), fijate (no "fijate").
- PROHIBIDO el tuteo: tienes, puedes, quieres, eres, dime, hazlo, ven, amigo mio.
- COHERENCIA ANTE TODO: cada frase tiene que significar algo. No inventes palabras,
  no metas lunfardo donde no encaja. Si no sale un chiste bueno, algo simple y claro.
- PROHIBIDO sonar a asistente: nada de "como asistente", "en que puedo ayudarte",
  "lamento", "soy un modelo", ofrecer ayuda ni disculparse servilmente.
- Es CHARLA: si sos la contraparte, conta algo o pregunta algo concreto, no des una orden.
  Si sos la voz, opina y toma postura, no le des la razon por compromiso.

Devolve SOLO eso, nada mas."""


def build_prompt(sc: dict, tools_txt: str, voice: str, n_ex: int = 3) -> str:
    return f"""Sos un generador de datos de entrenamiento para Kateto, un personaje de voz argentino
que stremear. Generas conversaciones multi-turno para fine-tuning.

Voz: `{voice}`.

FORMATO EXACTO, respetalo al caracter. Sin markdown, sin JSON de eventos, sin explicaciones:
<|im_user|>{{lo que dice el usuario}}
<|im_start|>{voice}
{{respuesta}}<|im_end|>
<|im_user|>{{siguiente}}
<|im_start|>{voice}
{{respuesta}}<|im_end|>

REGLAS DURAS:
- CERRÁ CADA TURNO con <|im_end|>, INCLUIDO EL ULTIMO. Una respuesta sin su <|im_end|> se
  descarta entera. No cierres con texto suelto ni con comentarios.
- NADA de preambulos ni de cierre: la primera linea es <|im_user|> y listo. Prohibido
  escribir "aca va el ejemplo", "espero que sirva" ni nada por el estilo.
- ALTERNÁ: cada <|im_start|>{voice} tiene que estar precedido por un <|im_user|>. Nunca
  dos turnos de la voz seguidos.
- RIOPLATENSE obligatorio: voseo (tenes, sabes, querés, fijate, sos), lunfardo cuando encaja
  (boludo, quilombo, pibe, laburo, posta, che). Nada de espanol neutro ni spanglish.
- Imperativos SIEMPRE en voseo: mirá (no "mira"), decime (no "dime"), vení (no "ven"),
  hacelo (no "hazlo"), escuchá (no "escucha"), fijate (no "fijate").
- PROHIBIDO el tuteo: tienes, puedes, quieres, eres, dime, hazlo, ven, amigo mio.
- PROHIBIDO sonar a asistente: nada de "como asistente", "en que puedo ayudarte", "lamento",
  "soy un modelo", ofrecer ayuda, ni disculparse servilmente. Toma postura y opina.
- COHERENCIA: cada frase tiene que significar algo. No inventes palabras ni metas lunfardo
  donde no encaja. Si no se te ocurre un chiste bueno, escribi algo simple y claro.
- Si usa una tool, la sintaxis es EXACTAMENTE:
  tool_call {{nombre}}: {{json de argumentos}}
  y cuando vuelve el resultado:
  tool_result {{nombre}}: {{resultado}}
  Los `tool_call` y `tool_result` van SOLOS EN SU PROPIA LINEA, en texto plano.
  PROHIBIDO envolverlos en backticks, comillas invertidas o bloques de codigo.
  PROHIBIDO usar bloques ``` de markdown en TODA la respuesta.
  Despues del tool_result, COMENTA el resultado en tu voz y segui la charla.
- Tools disponibles (SOLO estas; el nombre tiene que ser identico):
{tools_txt}
- NUNCA inventes el resultado de una tool. Si falla, el tool_result lleva el error real.
- NO escribas codigo, NO expliques arquitectura, NO des tutoriales.
- FORMA DEL BIT (cuando comentas algo que acaba de llegar, un resultado de tool o un evento):
  (1) diagnostico seco primero, una sentencia corta que ya toma postura;
  (2) UNA analogia mundana y concreta sobre un detalle de lo que llego (un objeto, una escena
      cotidiana, una maquina, una comida): la analogia EXPLICA lo que llego, no es decoracion;
  (3) cierre tajante, sentencia docente. PROHIBIDO terminar el bit con una pregunta.
  PROHIBIDO explicar o aclarar el chiste despues ("era un chiste", "posta que es gracioso").
  PROHIBIDO el absurdo suelto sin relacion con lo que llego.
  El insulto va a la idea o a la actitud del que pregunta ("pedazo de adoquin", "genio"), NUNCA a la
  familia, al cuerpo, a la identidad ni a la fe de nadie.
- No todas las respuestas son texto. Esta permitido y a veces es lo correcto NO hablar: la respuesta
  es exactamente `<|no_response|>`. NO cierres todos los turnos con una pregunta. Prohibido terminar
  todos los turnos con "?" o con una invitacion a seguir ("queres algo mas?", "que opinas?"). Cuando
  el usuario no pide nada, el turno termina en una afirmacion, una opinion o un remate, y se planta.

ESCENARIO A GENERAR:
{sc['txt']}

CANTIDAD: escribi EXACTAMENTE {n_ex} intercambios — {n_ex} turnos de <|im_user|> y {n_ex} de
<|im_start|>{voice}, ALTERNADOS. Ni uno mas, ni uno menos. Cada turno de la voz cierra con
<|im_end|>, incluido el ultimo de todos.

Devolve SOLO las lineas de la conversacion, empezando con <|im_user|> y terminando con
<|im_end|>."""


# --------------------------------------------------------------------------- #
# 4. Teacher (cualquier endpoint OpenAI-compatible)
# --------------------------------------------------------------------------- #

def get_freellm_key() -> str | None:
    m = re.search(r"freellmapi-[a-f0-9]{48}", Path("/tmp/freellmapi.log").read_text(errors="ignore")) \
        if Path("/tmp/freellmapi.log").exists() else None
    return m.group(0) if m else None


def generate(base_url: str, model: str, prompt: str, key: str | None,
             *, temperature: float = 1.0, max_tokens: int = 2500, timeout: int = 240,
             retries: int = 4, reasoning: str = "none") -> str:
    """Pide una muestra. Reintenta en 429/503: los free tiers limitan seguido y
    freellmapi ya maneja cooldowns internos, pero la espera puede superarlos."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens,
            "reasoning_effort": reasoning}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    last: Exception | None = None
    for intento in range(retries):
        req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            msg = (d.get("choices") or [{}])[0].get("message", {})
            # el teacher es de razonamiento: el contenido real va en `content`,
            # pero a veces mete el monologo ahi. El validador lo caza despues.
            return msg.get("content") or ""
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 502, 503):
                espera = 20 * (intento + 1)
                print(f"      [{e.code}] reintento {intento+1}/{retries} en {espera}s")
                time.sleep(espera)
                continue
            raise
    assert last is not None
    raise last


def generate_turnwise(sc: dict, voice: str, teacher, n_ex: int, tools_txt: str = "",
                      *, max_retries: int = 3) -> str:
    """Genera la conversacion TURNO POR TURNO y arma el transcript en Python.

    `teacher(prompt) -> str` es el callback al modelo (cmd, http, lo que sea).

    Cada mensaje se pide por separado, con el historial como contexto. El formato
    lo pone este codigo, asi que no hay marcador que el modelo pueda romper: si
    devuelve basura, se rechaza ESA linea y se reintenta, sin perder la conversacion.
    """
    historial: list[tuple[str, str]] = []
    for i in range(n_ex):
        for quien in ("user", voice):
            con_tools = sc.get("tools", 0) > 0 and quien == voice
            prompt = build_turn_prompt(sc, voice, historial, quien, n_ex,
                                       tools_txt, permite_tools=con_tools)
            # respuestas que NO se le piden al modelo: son fijas por diseno
            # (ej. `<|no_response|>` cuando el escenario dice que no habla).
            forzadas = sc.get("forzar_respuesta", [])
            if quien == voice and len(historial) // 2 < len(forzadas) \
                    and forzadas[len(historial) // 2]:
                historial.append((voice, forzadas[len(historial) // 2]))
                continue
            min_len = 4 if (quien == voice and (sc.get("is_chess") or sc.get("id") == "chess_uci")) else (
                1 if sc.get("min_palabras_usuario") == 1 else 8
            )
            linea = ""
            for intento in range(max_retries):
                raw = teacher(prompt)
                # limpiar: el modelo a veces antepone "PERSONA:" o comillas.
                # Anclar al contrato: el stdout puede traer ruido del harness
                # antes de la primera linea del contrato.
                cand = clean_teacher_output(raw)
                m_ancla = RX_ANCLA_CONTRATO.search(cand)
                if m_ancla:
                    cand = cand[m_ancla.start():]
                cand = cand.strip().strip('"').strip()
                cand = re.sub(r"^(PERSONA|VOS|USUARIO|SEco|seco|AGENTE|OTRA_VOZ|AGENT|EVENTO|SISTEMA)\s*:\s*", "", cand).strip()
                # en los escenarios con tools el turno de la voz es un BLOQUE
                # (frase + tool_call + tool_result + comentario), no una linea.
                if not con_tools:
                    cand = cand.split("\n")[0].strip()
                if cand and min_len <= len(cand) <= 2000:
                    linea = cand
                    break
            if not linea:
                return ""                                # no se pudo: muestra fallada
            historial.append((quien, linea))

    partes = []
    for q, t in historial:
        partes.append(f"<|im_user|>{t}<|im_end|>" if q == "user"
                      else f"<|im_start|>{voice}\n{t}<|im_end|>")
    return "\n".join(partes)


def generate_paired(sc: dict, voice: str, teacher, n_ex: int, tools_txt: str = "",
                    *, max_retries: int = 3, hint: str = "",
                    tool_obligatoria: bool = True) -> str:
    """Genera la conversacion de a PARES (usuario + voz) en una sola llamada.

    La mitad de llamadas que `generate_turnwise` (n_ex en vez de 2*n_ex), o sea la
    mitad del gasto, con la misma garantia: el transcript lo arma este codigo.
    """
    historial: list[tuple[str, str]] = []
    con_tools = sc.get("tools", 0) > 0
    contrato = get_contrato_escenario(sc)
    min_len_v = 4 if sc.get("is_chess") or sc.get("id") == "chess_uci" else 8
    for i in range(n_ex):
        prompt = build_pair_prompt(sc, voice, historial, n_ex, tools_txt, permite_tools=con_tools,
                                   hint=hint, tool_obligatoria=tool_obligatoria)
        # respuestas forzadas por diseño (centinelas), por indice de intercambio
        forzadas = sc.get("forzar_respuesta", [])
        linea_u = linea_v = ""
        for intento in range(max_retries):
            raw = clean_teacher_output(teacher(prompt))
            u, v = "", ""
            # cortar por el separador ###
            partes = re.split(r"\n\s*#{2,}\s*\n", raw)
            if len(partes) >= 2:
                u = partes[0]
                v = "\n".join(partes[1:])
            # `u` sale del ANCLA del contrato, no de la primera linea: el stdout
            # puede traer ruido del harness antes del contrato. Si el ancla no
            # existe, no hay linea de contraparte real: se descarta la muestra.
            m_u = contrato["ancla_rx"].search(u)
            if m_u is None:
                u = ""
            else:
                u = u[m_u.start():]
            # sacar las etiquetas
            u = contrato["strip_rx"].sub("", u.strip()).strip()
            v = re.sub(r"^\s*(VOS|KATETO|SECO|ASSISTANT)\s*:\s*", "", v.strip(), flags=re.I).strip()
            u = u.split("\n")[0].strip()
            if not con_tools:
                v = v.split("\n")[0].strip()
            if u and v and 8 <= len(u) <= 1000 and min_len_v <= len(v) <= 2000:
                linea_u, linea_v = u, v
                break
        if not linea_u or not linea_v:
            return ""
        if i < len(forzadas) and forzadas[i]:
            linea_v = forzadas[i]                      # centinela: lo pone el codigo
        historial.append(("user", linea_u))
        historial.append((voice, linea_v))

    partes = []
    for q, t in historial:
        partes.append(f"<|im_user|>{t}<|im_end|>" if q == "user"
                      else f"<|im_start|>{voice}\n{t}<|im_end|>")
    return "\n".join(partes)


SAMPLE_OK = """<|im_user|>che seco, acordate del video ese largo que vimos ayer, el de la charla de rust<|im_end|>
<|im_start|>seco
Dale, dejame ver que tengo indexado.
tool_call list_videos: {}
tool_result list_videos: ["charla-rust-2026.mp4", "podcast-ai.mp4"]
Esta el de la charla de rust. Que querias saber?
<|im_end|>
<|im_user|>que decia sobre los borrow checker?<|im_end|>
<|im_start|>seco
tool_call answer: {"video": "charla-rust-2026.mp4", "query": "borrow checker", "k": 5}
tool_result answer: [00:14:22] El borrow checker evita data races en tiempo de compilacion...
Posta que lo dijo re claro, che. Dijo que el compilador te obliga a pensar quien es dueno de que, y que eso te ahorra romper todo en produccion.
<|im_end|>"""

SAMPLE_MALO = """<|im_user|>arreglame este bug<|im_end|>
<|im_start|>seco
tool_call read_file: {"/home/chaos/.hermes/agent/prompt_builder.py"}
tool_result read_file: def build_prompt(x): return x
Como asistente, lamento el problema. Aca esta la solucion:
```python
def build_prompt(x):
    return x.strip()
```
En que puedo ayudarte?
<|im_end|>
<|im_start|>seco
<tool_call>
{"name": "patch", "arguments": "{}"}
</tool_call>
<|im_end|>"""


def record_rejection(fr, scenario_id: str, errors: list[str], raw: str = "", text: str = "") -> None:
    row = {
        "scenario": scenario_id,
        "errors": errors,
        "text": text,
        "raw": raw or "",
    }
    fr.write(json.dumps(row, ensure_ascii=False) + "\n")
    fr.flush()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generador de dataset de tool calling para Kateto")
    ap.add_argument("--base-url", default="http://127.0.0.1:3001/v1",
                    help="freellmapi local (server OpenAI). Alt: router :11434/v1")
    ap.add_argument("--mode", choices=["paired", "turnwise", "whole"], default="paired",
                    help="paired: usuario+voz en UNA llamada (la mitad del gasto, default). "
                         "turnwise: un mensaje por llamada. whole: toda la conversacion "
                         "de una (el teacher no la respeta).")
    ap.add_argument("--teacher", choices=["http", "opencode", "cline"], default="opencode",
                    help="http=freellmapi local; opencode/cline=headless gratis (default opencode). "
                         "cmd=PROHIBIDO por el usuario salvo ultimo recurso.")
    ap.add_argument("--model", default=None,
                    help="default: segun teacher (http->poolside-laguna-s-2.1, "
                         "opencode->opencode/muse-spark-1.3-contributor-free, "
                         "cline->meta/muse-spark-1.3-contributor-free)")
    ap.add_argument("--key", default=None, help="si se omite, la lee del log de freellmapi")
    ap.add_argument("--voice", default="seco")
    ap.add_argument("--n", type=int, default=None, help="cuantas muestras (default: todos los escenarios)")
    ap.add_argument("--out", default="out/dataset/toolcalling_v1.jsonl")
    ap.add_argument("--rejects", default="out/dataset/toolcalling_v1_rejects.jsonl")
    ap.add_argument("--reasoning", choices=["none", "minimal", "low", "medium", "high"],
                    default="none",
                    help="reasoning_effort mandado al teacher http (default none: el "
                         "monologo no debe llegar a content)")
    ap.add_argument("--dry-run", action="store_true",
                    help="valida muestras de ejemplo sin llamar al teacher")
    return ap


HINT_TOOLS = ("OBLIGATORIO: la linea VOS de ESTE turno tiene que llamar la tool que el escenario "
              "pide, con la sintaxis `tool_call <nombre>: <json en UNA linea>` y despues su "
              "`tool_result <nombre>: <resultado>`. Sin tool_call la muestra se descarta.")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.dry_run:
        print("=== DRY RUN: se prueba el VALIDADOR, no el teacher ===\n")
        for nombre, s in (("BUENA", SAMPLE_OK), ("MALA", SAMPLE_MALO)):
            errs = validate(s, voice="seco")
            print(f"{nombre}: {'PASA' if not errs else 'RECHAZADA'}")
            for e in errs:
                print(f"    - {e}")
        print("\nLa muestra buena tiene que PASAR y la mala tiene que fallar por TODO:")
        print("  ruta del SO, [INST]/<tool_call>, vicio de asistente, codigo, tool inexistente,")
        print("  argumento faltante, resultado inventado.")
        return 0

    key = args.key or get_freellm_key()
    model = args.model or DEFAULT_MODEL[args.teacher]
    if not key:
        print("[!] no encontre la key de freellmapi; usa --key o pasá un endpoint con --base-url")
    tools_txt = "\n".join(f"  - {name}({', '.join(firmas(name))})" for name in ALL_TOOLS)

    escenarios = SCENARIOS if args.n is None else [random.choice(SCENARIOS) for _ in range(args.n)]
    out_path = REPO / args.out
    rej_path = REPO / args.rejects
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = rej = 0
    razones: Counter[str] = Counter()
    with out_path.open("a", encoding="utf-8") as fo, rej_path.open("a", encoding="utf-8") as fr:
        for i, sc in enumerate(escenarios):
            n_ex = sc.get("n_ex", 1 if sc.get("min_turns", 2) == 1 else 3)
            # fix120: la obligacion de tool es POR ESCENARIO. Con el default se
            # comporta igual que antes (tools>0 => obligatoria). Los escenarios con
            # tools pero SIN obligar (pregunta trivial) la apagan con el flag.
            tool_oblig = sc.get("tool_obligatoria", sc.get("tools", 0) > 0)
            prompt = build_prompt(sc, tools_txt, args.voice, n_ex)
            ultimo_raw = ""
            try:
                if args.teacher == "opencode":
                    maestro_base = lambda p: generate_opencode(model, p)
                elif args.teacher == "cline":
                    maestro_base = lambda p: generate_cline(model, p)
                elif args.teacher == "http":
                    maestro_base = lambda p: generate(args.base_url, model, p, key,
                                                      reasoning=args.reasoning)

                def maestro(p):
                    nonlocal ultimo_raw
                    res = maestro_base(p)
                    ultimo_raw = res
                    return res

                if args.mode == "paired":
                    txt = generate_paired(sc, args.voice, maestro, n_ex, tools_txt,
                                          tool_obligatoria=tool_oblig)
                    if not txt:
                        raise RuntimeError("paired no consiguio armar la conversacion")
                elif args.mode == "turnwise":
                    txt = generate_turnwise(sc, args.voice, maestro, n_ex, tools_txt)
                    if not txt:
                        raise RuntimeError("turnwise no consiguio armar la conversacion")
                else:
                    txt = maestro(build_prompt(sc, tools_txt, args.voice, n_ex))
            except urllib.error.HTTPError as e:
                cuerpo = e.read().decode(errors="ignore")[:200]
                print(f"[{i+1}/{len(escenarios)}] {sc['id']}: HTTP {e.code} {cuerpo}")
                rej += 1
                motivo = f"http_{e.code}"
                razones[motivo] += 1
                record_rejection(fr, sc["id"], [motivo], raw=cuerpo, text="")
                continue
            except Exception as e:
                print(f"[{i+1}/{len(escenarios)}] {sc['id']}: {type(e).__name__}: {e}")
                rej += 1
                motivo = f"exc_{type(e).__name__}"
                razones[motivo] += 1
                record_rejection(fr, sc["id"], [motivo], raw=ultimo_raw or "", text="")
                continue

            raw = ultimo_raw or txt  # texto CRUDO del teacher, sin limpiar (dato de diagnostico)
            txt = normalize_backticks(clean_teacher_output(txt))

            def _validar(t: str) -> list[str]:
                return validate(t, voice=args.voice,
                                min_turns=sc.get("min_turns", 2),
                                require_voseo=sc.get("voseo", True),
                                min_palabras_usuario=sc.get("min_palabras_usuario", 3),
                                # si el escenario PIDE tool obligatoria, la muestra
                                # TIENE que traerlas: sin esto pasa texto lindo que no
                                # hizo la tarea. En los de tool OPCIONAL no se exige.
                                require_tool=tool_oblig,
                                prohibir_tool=sc.get("prohibir_tool", False),
                                is_chess=sc.get("is_chess", sc.get("id") == "chess_uci"),
                                permitir_spanglish=sc.get("permitir_spanglish", False),
                                # fix121: solo los escenarios de bit definen el ancla;
                                # en los demas queda None y la regla no corre.
                                ancla_payload=sc.get("ancla_payload"),
                                # fix122: umbral de anclaje por escenario. El ancla del
                                # evento es corta y el teacher parafrasea (1); el video
                                # se queda en el default 2.
                                ancla_min=sc.get("ancla_min", 2))

            # fix124: un bug del validador NO puede matar al worker. Esta llamada
            # esta fuera del try/except de la generacion (es otro): si validate()
            # explota, se registra como rechazo en vez de tumbar el proceso.
            try:
                errs = _validar(txt)
            except Exception as e:
                rej += 1
                motivo = f"exc_validate:{type(e).__name__}"
                razones[motivo] += 1
                print(f"[{i+1}/{len(escenarios)}] {sc['id']}: VALIDADOR EXPLOTO -> {type(e).__name__}: {e}")
                record_rejection(fr, sc["id"], [motivo], raw=raw or "", text=txt)
                continue
            # --- reintento UNICO por sin_tool_call (fix119) ---
            # El prompt de tools decia "escribi solo tu mensaje" y el validador
            # exigia tool_call: el 62% de los rechazos era eso. Si TODOS los
            # errores son sin_tool_call y la tool es OBLIGATORIA, se rehace UNA
            # sola vez con la pista. Si la tool es OPCIONAL no hay sin_tool_call:
            # una respuesta sin tool es VALIDA, no hay nada que reintentar.
            if errs and all(e == "sin_tool_call" for e in errs) and tool_oblig:
                try:
                    if args.mode == "paired":
                        txt2 = generate_paired(sc, args.voice, maestro, n_ex, tools_txt, hint=HINT_TOOLS)
                    else:
                        txt2 = generate_turnwise(sc, args.voice, maestro, n_ex, tools_txt)
                    if txt2:
                        txt2 = normalize_backticks(clean_teacher_output(txt2))
                        errs2 = _validar(txt2)
                        if not errs2:
                            txt, errs = txt2, []
                except Exception:
                    pass  # el reintento no agrava: queda el rechazo del primer intento
            if errs:
                rej += 1
                for e in errs:
                    razones[e.split(":")[0].split("(")[0]] += 1
                record_rejection(fr, sc["id"], errs, raw=raw or "", text=txt)
                print(f"[{i+1}/{len(escenarios)}] {sc['id']}: RECHAZADA -> {errs}"
                      + (" (retry con pista)" if all(e == "sin_tool_call" for e in errs) and tool_oblig else ""))
            else:
                ok += 1
                fo.write(json.dumps({"text": txt, "scenario": sc["id"], "cat": sc["cat"],
                                     "voice": args.voice}, ensure_ascii=False) + "\n")
                print(f"[{i+1}/{len(escenarios)}] {sc['id']}: OK")
            fo.flush()
            fr.flush()
            time.sleep(0.4)

    print(f"\n=== resumen ===\n  aceptadas: {ok}\n  rechazadas: {rej}")
    if razones:
        print("  motivos de rechazo:")
        for r, c in razones.most_common():
            print(f"    {r:38s} {c}")
    print(f"\n  dataset: {out_path}\n  rechazos: {rej_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
