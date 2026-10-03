#!/usr/bin/env python3
"""R2 — Generacion turnwise del dataset de Kateto, con estado autoritativo.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python pipelines/kateto_gen.py \
      --n 2 --model auto --out pipelines/smoke/tc_r2.jsonl

Que cambia respecto de `scripts/gen_toolcalling_dataset.py` (la referencia, que NO
se toca): los `tool_result` los pone el PIPELINE, no el teacher.

La referencia le pide al teacher las dos mitades del turno de la voz — el
`tool_call` Y el `tool_result` — y el teacher inventa el resultado. Un
`tool_result` sin estado detras es una alucinacion con formato valido: pasa
`args_no_json`, pasa `resultado_sin_llamada` y ensena al modelo a fabricated
resultados de tools. Aca el `tool_result` sale de un estado real en memoria
(`ToolState`: minecraft, video-rag, eventos, archivos, memoria) y el teacher
nunca escribe uno.

Los cuatro roles del loop:

  1. usuario   — teacher (simulador de usuario)
  2. voz       — teacher (Kateto). Escribe prosa y `tool_call`; NUNCA el resultado
  3. tools     — el pipeline. Ejecuta cada `tool_call` contra `ToolState` y emite el
                 `tool_result` (o un error real de `ref.REAL_ERRORS`)
  4. juez      — el pipeline. En el loop decide si el turno se acepta (largo,
                 vacio); al final es el Step `AnotarValidacion`

Se reusa de la referencia, sin duplicar: `SCENARIOS`, `ALL_TOOLS`, `firmas`,
`validate`, `extraer_bloques`, `normalize_backticks`, `clean_teacher_output`,
`build_turn_prompt`, `generate`, `get_freellm_key`, `REAL_ERRORS`, `SENTINELS`.
Y de R1, sin duplicar: las reglas de purga de `kateto_purge.REGLAS`.

El formato del dato NO se toca:
    <|im_user|>texto<|im_end|>
    <|im_start|>seco
    respuesta<|im_end|>

Distilabel 1.5.3 no expresa el loop con estado (cada `TextGeneration` es una
llamada por fila y es stateless respecto de la conversacion), asi que el loop es
codigo propio y el framework aporta los Steps. Ver `main()`.
"""

# Sin `from __future__ import annotations`: distilabel detecta el parametro de
# entrada de un Step por identidad sobre el Annotated real, y con PEP 563 la
# anotacion llega como string y el Step se rechaza. Mismo motivo que en R1.
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

PIPELINES = Path(__file__).resolve().parent
REPO = PIPELINES.parent
for _p in (PIPELINES, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_toolcalling_dataset as ref  # noqa: E402  la referencia, sin tocar
import kateto_purge as purge  # noqa: E402  reglas de R1

if TYPE_CHECKING:  # pragma: no cover
    from distilabel.typing import StepColumns

# --------------------------------------------------------------------------- #
# Tokens y formato. Marcadores tal cual vienen en el dato: 4 literales, no un
# parser de chat. Si el formato cambia, este es el unico lugar a tocar.
# --------------------------------------------------------------------------- #
IM_USER = "<|im_user|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
MARCADOR_VOZ = f"{IM_START}{{voz}}\n"          # el marcador real de un turno de voz
# grupo 1 = el marcador SOLO, grupo 2 = el cuerpo: re-armar es `g1+g2+<|im_end|>`.
RX_BLOQUE = re.compile(
    f"({re.escape(IM_USER)}|{re.escape(IM_START)}[^\n]*\n)(.*?){re.escape(IM_END)}", re.S
)
RX_TOOL_RESULT = re.compile(r"(?m)^\s*tool_result\b.*$")
RX_TOOL_BLOCK = re.compile(r"(?m)^\s*tool_(?:call|result)\b.*$")

MAX_LINEA = 2000
MIN_LINEA_VOZ = 8
MAX_INTENTOS = 3
REINTENTOS_401 = 4

# El teacher del brief es `auto` de freellmapi y esta prohibido pinear otro.
DEFAULT_MODEL = {"http": "auto", "opencode": "auto", "cline": "auto"}


# --------------------------------------------------------------------------- #
# 2. Estado autoritativo de las tools
# --------------------------------------------------------------------------- #
class ToolState:
    """Estado real en memoria. El `tool_result` sale de ACA, no del teacher.

    Escenario vacio = la tool falla con un error REAL de `ref.REAL_ERRORS`. Un
    `tool_result` que no se puede resolver es un error del runtime, no texto
    inventado: el modelo tiene que aprender a((*reaccionar al error*).

    ponytail: un dict por dominio y un dispatch. Si el catalogo de tools crece,
    el `else` del dispatch lo cubre con `unknown tool` en vez de agregar metodos.
    """

    def __init__(self, *, reloj: str = "22:41", voces_vivas: tuple[str, ...] = ()) -> None:
        # video-rag: id -> fragmento de transcripcion. Vacio por defecto: la
        #Escena lo puebla segun el escenario.
        self.videos: dict[str, str] = {}
        self.eventos: list[dict[str, Any]] = []
        self.archivos: dict[str, str] = {}
        self.memoria: dict[str, str] = {}
        self.plugins: dict[str, bool] = {}
        self.minecraft = False
        self.skill_voyager = False
        self.reloj = reloj
        self.voces_vivas = set(voces_vivas)

    # -- material por escenario ------------------------------------------- #
    def sembrar(self, sc: dict) -> None:
        """Carga el estado con el material del escenario.

        Solo lo que el escenario declara de forma verificable. Lo demas arranca
        vacio: es preferible un `Error: file not found` honesto a un indice
        inventado.
        """
        vid = sc.get("video_id")
        frag = sc.get("video_fragmento")
        if not vid and sc.get("ancla_payload"):
            vid, frag = "video-indexado.mp4", sc["ancla_payload"]
        if vid and frag:
            self.videos[vid] = frag
        if sc.get("id") == "minecraft_gate_ok":
            self.minecraft = True
            self.skill_voyager = True
        if sc.get("id") == "bit_sobre_evento":
            self.eventos.append(
                {"event_name": "bloque_de_juego", "data": sc.get("ancla_payload", "")}
            )
        if sc.get("id") == "fuente_otra_voz":
            self.voces_vivas.add("doktor")

    # -- dispatch ---------------------------------------------------------- #
    def call(self, nombre: str, args: Any) -> str:
        """Ejecuta una tool y devuelve el `tool_result` como texto plano."""
        if nombre not in ref.ALL_TOOLS:
            return "Error: unknown tool"
        if not isinstance(args, dict):
            return "Error: missing required argument"
        faltan = [r for r in ref.ALL_TOOLS[nombre]["required"] if r not in args]
        if faltan:
            return f"Error: missing required argument"
        fn = getattr(self, f"_{nombre}", None)
        if fn is None:
            return "Error: unknown tool"
        try:
            return fn(args)
        except Exception as e:  # un handler roto es un error, no una muerte
            return f"Error: {type(e).__name__}"

    # -- video-rag --------------------------------------------------------- #
    def _list_videos(self, a: dict) -> str:
        return json.dumps(sorted(self.videos), ensure_ascii=False)

    def _describe(self, a: dict) -> str:
        return self._fragmento(a.get("video", ""))

    def _summarize(self, a: dict) -> str:
        return self._fragmento(a.get("video", ""))

    def _answer(self, a: dict) -> str:
        return self._fragmento(a.get("video", ""))

    def _search(self, a: dict) -> str:
        q = str(a.get("query", "")).lower()
        hits = [v for v, f in self.videos.items() if q in f.lower() or q in v.lower()]
        return json.dumps(hits, ensure_ascii=False)

    def _fragmento(self, video: str) -> str:
        if video not in self.videos:
            return "Error: file not found"
        return f"[00:14:22] {self.videos[video]}"

    # -- eventos ----------------------------------------------------------- #
    def _list_events(self, a: dict) -> str:
        return json.dumps(self.eventos, ensure_ascii=False)

    def _send_event(self, a: dict) -> str:
        data = a.get("data")
        if not isinstance(data, dict):
            return "Error: argument must be a string"
        if not self.voces_vivas:
            return "Error: event has no receivers"
        self.eventos.append({"event_name": a.get("event_name", ""), "data": data})
        return "OK"

    def _schedule_event(self, a: dict) -> str:
        ev = {"event_name": a.get("event_name", "")}
        for k in ("delay", "interval", "cron", "target_voice", "dept"):
            if k in a:
                ev[k] = a[k]
        self.eventos.append(ev)
        return f"OK scheduled {ev['event_name']}"

    def _get_current_time(self, a: dict) -> str:
        return self.reloj

    # -- archivos ---------------------------------------------------------- #
    def _read_file(self, a: dict) -> str:
        path = str(a["path"])
        return self.archivos.get(path) or "Error: file not found"

    def _write_file(self, a: dict) -> str:
        self.archivos[str(a["path"])] = str(a["content"])
        return "OK"

    def _delete_file(self, a: dict) -> str:
        path = str(a["path"])
        if path not in self.archivos:
            return "Error: file not found"
        del self.archivos[path]
        return "OK"

    def _run_command(self, a: dict) -> str:
        return "Error: executable not found"

    # -- voz a voz --------------------------------------------------------- #
    def _request_generation(self, a: dict) -> str:
        if a.get("target_voice") not in self.voces_vivas:
            return "Error: no response from target voice"
        return "OK"

    # -- plugins / memoria -------------------------------------------------- #
    def _list_plugins(self, a: dict) -> str:
        return json.dumps(sorted(self.plugins), ensure_ascii=False)

    def _enable_plugin(self, a: dict) -> str:
        self.plugins[str(a["name"])] = True
        return "OK"

    def _disable_plugin(self, a: dict) -> str:
        self.plugins[str(a["name"])] = False
        return "OK"

    def _refine_memory(self, a: dict) -> str:
        self.memoria[str(a["dept"])] = str(a.get("fact") or a.get("evidence", ""))
        return "OK"

    def _create_skill(self, a: dict) -> str:
        return "Error: already exists, use update_skill"

    # -- juegos: GATE DOBLE (minecraft + skill habilitado) ------------------ #
    def _voyager(self, a: dict) -> str:
        if not (self.minecraft and self.skill_voyager):
            return "Error: unknown tool"
        return "OK mined diamond at 12, 64, -31"

    _voyager_craft_stick = _voyager
    _voyager_mine_diamond = _voyager


# --------------------------------------------------------------------------- #
# 3. Prompts. Los de la referencia se reusan; solo el turno de la voz CON tools
# cambia de protocolo (no escribe el resultado), y ese es el nucleo del port.
# --------------------------------------------------------------------------- #
REGISTRO_PATH = REPO / "prompts" / "prompt-compartido.md"
# La receta de tres partes (`Diagnóstico:` -> analogia -> `Cierre:`) vivia en el
# `txt` de los escenarios `bit_*`. R4: la comedia es REGISTRO, no estructura — el
# andamiaje se filtraba al dialogo y por eso existe `purga.andamiaje_en_dialogo`.
# La referencia (`scripts/gen_toolcalling_dataset.py`) es de otro owner y no se
# toca, asi que la clausula se saca aca, en el prompt, y no en el archivo.
RX_RECETA = re.compile(r"con un BIT (?:propio )?en la forma de siempre:[^.]*\.")


@lru_cache(maxsize=1)
def registro() -> str:
    """`prompts/prompt-compartido.md` VERBATIM. El archivo es la especificacion:
    copiarlo al codigo lo desalinearia en silencio en la primera edicion."""
    return REGISTRO_PATH.read_text(encoding="utf-8")


def sin_receta(txt: str) -> str:
    """El `txt` del escenario sin las tres partes etiquetadas.

    Un solo regex para los dos `bit_*`: los dos textos arrancan la clausula en
    `en la forma de siempre:` y la terminan en `pregunta.`, sin puntos adentro.
    """
    return RX_RECETA.sub("en la voz de Kateto.", txt)


def tools_txt() -> str:
    return "\n".join(f"  - {n}({', '.join(ref.firmas(n))})" for n in ref.ALL_TOOLS)


def transcript(sc: dict, historial: list[tuple[str, str]]) -> str:
    """Lo dicho hasta ahora, con la etiqueta de contrato del escenario.

    La misma linea que usa `build_pair_prompt` de la referencia, para que este
    prompt lea igual que los dems.
    """
    etiqueta = ref.get_contrato_escenario(sc)["etiqueta"]
    return "\n".join(
        f"{etiqueta if q == 'user' else 'VOS'}: {t}" for q, t in historial
    ) or "(todavia no hablo nadie)"


def build_voice_tool_prompt(sc: dict, voice: str, historial: list[tuple[str, str]],
                            n_ex: int, estado: ToolState) -> str:
    """Pide el turno de la voz CON tools: prosa + `tool_call`, nunca el resultado.

    Es `build_turn_prompt` de la referencia con otro protocolo de tool: el
    teacher no escribe el `tool_result`.
    """
    estado_txt = json.dumps(
        {"videos": sorted(estado.videos), "eventos": [e["event_name"] for e in estado.eventos],
         "minecraft": estado.minecraft, "skill_voyager": estado.skill_voyager},
        ensure_ascii=False,
    )
    return f"""Estas escribiendo UNA sola linea de dialogo para un dataset de Kateto: un
personaje de voz argentino, streamer, con caracter propio. NO sos un asistente.

Escenario: {sc['txt']}

Lo que se dijo hasta ahora:
{transcript(sc, historial)}

Escribi AHORA el proximo mensaje, que le toca a la voz `{voice}` (Kateto). Es el
mensaje {len(historial)+1} de {n_ex*2} en total.

PROTOCOLO DE TOOLS (distinto al del resto de los escenarios):
- PRIMERO una o dos frases tuyas anunciando lo que vas a hacer.
- DESPUES la llamada, sola en su linea, JSON VALIDO en UNA sola linea:
  tool_call <nombre>: <json de argumentos>
- El NOMBRE tiene que ser IDENTICO al de la lista. El JSON tiene que tener
  todos los argumentos requeridos y ningun otro.
- NUNCA escribas `tool_result`. El resultado lo produce el runtime, no vos. Si
  escribis un `tool_result`, el turno entero se descarta.
- ULTIMO: nada. Tu turno termina en el `tool_call`. El comentario sobre el
  resultado te lo van a pedir despues, cuando te lo muestren.

Tools disponibles (el nombre tiene que ser IDENTICO):
{tools_txt()}

ESTADO REAL DEL SISTEMA (podes citarlo, no podés inventar otra cosa):
{estado_txt}

Si NO hace falta ninguna tool en este punto, escribe solo tu mensaje, sin
llamadas.

REGLAS:
- RIOPLATENSE: voseo (tenes, sabes, sos, fijate), lunfardo cuando encaja (boludo,
  quilombo, pibe, laburo, posta, che). Ni espanol neutro ni spanglish.
- Imperativos SIEMPRE en voseo: mira (no "mira"), decime (no "decime").
- PROHIBIDO el tuteo: tienes, puedes, quieres, eres, dime, hazlo, ven, amigo mio.
- COHERENCIA ANTE TODO: cada frase tiene que significar algo. No inventes palabras.
- Kateto NO programa y NO menciona codigo.
- PROHIBIDO sonar a asistente: nada de "como asistente", "en que puedo ayudarte",
  "lamento", "soy un modelo", ofrecer ayuda ni disculparte servilmente.
- No cierres con una pregunta. No expliques el chiste despues.
- Sin marcadores {IM_USER} ni {IM_END}, sin "VOS:" al principio, sin markdown.

Devolve SOLO eso, nada mas."""


# --------------------------------------------------------------------------- #
# 4. El loop de cuatro roles
# --------------------------------------------------------------------------- #
def _limpiar_linea(raw: str, *, con_tools: bool) -> str:
    """Limpia una respuesta del teacher: ruido de stdout, etiquetas, comillas.

    Reusa `clean_teacher_output` y el anclaje al contrato de la referencia: el
    stdout del harness mete ruido antes de la primera linea real.
    """
    cand = ref.clean_teacher_output(raw)
    m = ref.RX_ANCLA_CONTRATO.search(cand)
    if m:
        cand = cand[m.start():]
    cand = cand.strip().strip('"').strip()
    cand = re.sub(
        r"^(PERSONA|VOS|USUARIO|seco|SEco|AGENTE|OTRA_VOZ|AGENT|EVENTO|SISTEMA)\s*:\s*",
        "", cand,
    ).strip()
    return cand if con_tools else cand.split("\n")[0].strip()


def _solo_tool_calls(texto: str) -> str:
    """Deja la prosa y los `tool_call`; borra los `tool_result` del teacher.

    El teacher los escribe por costumbre (la referencia se lo pide asi). Acá el
    resultado lo pone el pipeline: si se deja el del teacher, el `tool_result` es
    una alucinacion con formato valido.
    """
    return RX_TOOL_RESULT.sub("", texto).strip()


def _solo_prosa(texto: str) -> str:
    """Saca los bloques `tool_call`/`tool_result` y deja solo la prosa.

    El loop reconstruye las llamadas canonicales desde el estado, asi que la
    linea cruda del teacher no puede sobrevivir: si queda, el turno lleva la
    llamada DUPLICADA (una del teacher, una del pipeline).
    """
    return RX_TOOL_BLOCK.sub("", texto).strip()


def _bloques_tools(texto: str) -> list[tuple[str, str]]:
    """[(nombre, args_json)] de los `tool_call` del turno de la voz."""
    out = []
    for tipo, nombre, cuerpo in ref.extraer_bloques(texto):
        if tipo == "call":
            out.append((nombre, cuerpo))
    return out


def _turno_valido(texto: str, min_len: int) -> bool:
    """ROL 4, en el loop: el juez de corte decide si el turno entra.

    Vacio, o fuera de rango de largo, y el turno se reintenta. El corte final
    (el sample entero) es el Step `AnotarValidacion`.
    """
    return bool(texto) and min_len <= len(texto) <= MAX_LINEA


def generar_muestra(sc: dict, voice: str, teacher: Callable[[str], str],
                    n_ex: int, *, max_retries: int = MAX_INTENTOS,
                    usar_registro: bool = True) -> tuple[str, ToolState]:
    """Arma UNA conversacion turno por turno. Devuelve `(texto, estado)`.

    El transcript lo arma este codigo: no hay marcador que el modelo pueda
    romper, ni turnos desbalanceados, ni falta de cierre. Los centinelas los
    pone el codigo por indice de intercambio, nunca se los pide al modelo.

    `usar_registro=False` es el brazo OFF del A/B: mismo loop, mismo teacher,
    mismo escenario, sin el registro inyectado.
    """
    sc = {**sc, "txt": sin_receta(sc["txt"])}
    estado = ToolState(voces_vivas=(voice,))
    estado.sembrar(sc)
    con_tools = sc.get("tools", 0) > 0
    min_len_v = 4 if sc.get("is_chess") or sc.get("id") == "chess_uci" else MIN_LINEA_VOZ
    min_len_u = 1 if sc.get("min_palabras_usuario") == 1 else 8
    forzadas = sc.get("forzar_respuesta", [])

    def de_kateto(prompt: str) -> str:
        """El registro va en los prompts donde HABLA Kateto.

        No en el del simulador de usuario: `prompt-compartido.md` es la ficha del
        personaje ("Sos Kateto", "Cerrás sin preguntar"), y metersela a quien
        tiene que PREGUNTAR leense la propia voz y deja de hacer su trabajo.
        """
        return f"{registro()}\n\n{prompt}" if usar_registro else prompt

    historial: list[tuple[str, str]] = []
    for i in range(n_ex):
        # --- ROL 1: simulador de usuario ---------------------------------- #
        linea_u = _un_turno(
            teacher, ref.build_turn_prompt(sc, voice, historial, "user", n_ex),
            min_len=min_len_u, con_tools=False, max_retries=max_retries,
        )
        if not linea_u:
            return "", estado
        historial.append(("user", linea_u))

        # --- centinela: lo pone el codigo, por indice de intercambio ------ #
        if i < len(forzadas) and forzadas[i]:
            historial.append((voice, forzadas[i]))
            continue

        # --- ROL 2: la voz (Kateto) --------------------------------------- #
        if not con_tools:
            linea_v = _un_turno(
                teacher, de_kateto(ref.build_turn_prompt(sc, voice, historial, voice, n_ex)),
                min_len=min_len_v, con_tools=False, max_retries=max_retries,
            )
            if not linea_v:
                return "", estado
            historial.append((voice, linea_v))
            continue

        # --- ROL 2 + 3: prosa y tool_call del teacher, resultado del pipeline #
        crudo = ""
        for _ in range(max_retries):
            crudo = ref.normalize_backticks(
                _limpiar_linea(
                    teacher(de_kateto(build_voice_tool_prompt(sc, voice, historial,
                                                             n_ex, estado))),
                    con_tools=True,
                )
            )
            if _turno_valido(crudo, min_len_v):
                break
            crudo = ""
        if not crudo:
            return "", estado

        llamadas = _bloques_tools(_solo_tool_calls(crudo))
        if not llamadas:
            # El escenario pedia tools y no hubo llamada: el turno va igual, y
            # `validate` lo rechaza con `sin_tool_call` (o `prohibir_tool`).
            historial.append((voice, _solo_tool_calls(crudo)))
            continue

        # el pipeline reconstruye las llamadas desde el estado y ejecuta
        lineas = [_solo_prosa(crudo)]
        for nombre, args_txt in llamadas:
            try:
                args = json.loads(re.sub(r"\s*\r?\n\s*", " ", args_txt))
            except Exception:
                lineas.append(f"tool_call {nombre}: {args_txt}")
                lineas.append("Error: missing required argument")
                continue
            args_txt = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
            lineas.append(f"tool_call {nombre}: {args_txt}")
            lineas.append(f"tool_result {nombre}: {estado.call(nombre, args)}")

        # --- ROL 2 bis: la voz comenta el resultado que acaba de llegar ---- #
        bloque = "\n".join(lineas)
        historial.append((voice, bloque))
        comentario = _un_turno(
            teacher, de_kateto(ref.build_turn_prompt(sc, voice, historial, voice, n_ex)),
            min_len=min_len_v, con_tools=False, max_retries=max_retries,
        )
        if not comentario:
            return "", estado
        historial[-1] = (voice, f"{bloque}\n{comentario}")

    return ensamblar(historial, voice), estado


def _un_turno(teacher: Callable[[str], str], prompt: str, *, min_len: int,
              con_tools: bool, max_retries: int) -> str:
    """Un mensaje del teacher, con reintentos. Vacio = no se pudo."""
    for _ in range(max_retries):
        cand = _limpiar_linea(teacher(prompt), con_tools=con_tools)
        if _turno_valido(cand, min_len):
            return cand
    return ""


def ensamblar(historial: list[tuple[str, str]], voice: str) -> str:
    """El texto final, con el formato exacto del dato. Sin modelo adentro.

    El cierre del ultimo bloque lleva salto: `<|im_end|>\\n` es el canonico (R4).
    Ojo con el otro lado de la moneda: MEDIDO, ningun texto del corpus termina en
    `\\n` — 0 de 55.137 del train set, 0 de los shards, 0 de los turnos reales.
    El `\\n` va ENTRE bloques (`<|im_end|>\\n<|im_start|>`), que es lo que este
    `join` ya hacia. El salto final es inerte para el entrenamiento
    (`build_labels` revierte a -100 todo lo que sigue al `<|im_end|>`), asi que
    no filtra gradiente: es consistencia de forma, noaprendizaje.

    ponytail: 4 concatenaciones. Si el formato del dato cambia, se toca aca y en
    las 4 constantes de arriba; no hay otro lugar que arme bloques.
    """
    partes = [
        f"{IM_USER}{t}{IM_END}" if q == "user" else f"{IM_START}{voice}\n{t}{IM_END}"
        for q, t in historial
    ]
    return "\n".join(partes) + "\n"


# Escape hatch de mutacion para los tests: GEN_MUTAR=armado saca el cierre del
# ultimo bloque, y la suite tiene que CAER.
if os.environ.get("GEN_MUTAR") == "armado":
    _orig_ensamblar = ensamblar

    def ensamblar(historial: list[tuple[str, str]], voice: str) -> str:  # noqa: F811
        txt = _orig_ensamblar(historial, voice)
        return txt[: -len(IM_END + "\n")] if txt.endswith(IM_END + "\n") else txt


def desarmar(txt: str) -> list[tuple[str, str]]:
    """Re-armado por regex: `[(cabecera, cuerpo)]`. Inverso de `ensamblar`.

    La cabecera es el marcador SOLO (`<|im_user|>` o `<|im_start|>seco\n`), asi
    que re-armar es `cabecera + cuerpo + IM_END`: si eso no devuelve el texto
    EXACTO, el formato tiene una fuga.
    """
    return [(m.group(1), m.group(2)) for m in RX_BLOQUE.finditer(txt)]


def rearmar(txt: str) -> str:
    """`desarmar` y volver a `ensamblar`. Da el mismo texto si el formato cierra.

    El separador `\\n` entre bloques no lo captura `RX_BLOQUE` (cae entre el cierre
    de uno y la cabecera del siguiente), asi que se pone al re-armar. El del
    cierre final tambien: es parte del formato.
    """
    return "\n".join(f"{cab}{cuerpo}{IM_END}" for cab, cuerpo in desarmar(txt)) + "\n"


# --------------------------------------------------------------------------- #
# 5. Steps de distilabel
# --------------------------------------------------------------------------- #
from distilabel.pipeline import Pipeline  # noqa: E402
from distilabel.steps import LoadDataFromDicts, Step, StepInput  # noqa: E402
from pydantic import PrivateAttr  # noqa: E402


class GenerarMuestras(Step):
    """ROL 1-4: corre el loop por escenario y devuelve la muestra armada.

    El estado de las tools vive adentro del Step, no en el modelo: es lo que
    hace que un `tool_result` tenga estado detras.

    El teacher se arma DENTRO de `process` y no se guarda como campo: distilabel
    corre cada Step en otro proceso y al serializar los campos un callable
    llega como dict.
    """

    voice: str = "seco"
    base_url: str = "http://localhost:3001/v1"
    model: str = "auto"
    teacher_key: str | None = None
    reasoning: str = "none"
    n_ex: int | None = None
    sin_registro: bool = False

    @property
    def inputs(self) -> "StepColumns":
        return ["escenario"]

    @property
    def outputs(self) -> "StepColumns":
        return ["text", "scenario", "cat", "voice", "fallo"]

    def process(self, *inputs: StepInput) -> "StepOutput":
        teacher = TeacherHTTP(self.base_url, self.model, self.teacher_key, self.reasoning)
        for rows in inputs:
            for row in rows:
                sc = row["escenario"]
                n_ex = self.n_ex or sc.get(
                    "n_ex", 1 if sc.get("min_turns", 2) == 1 else 3
                )
                txt, _ = generar_muestra(sc, self.voice, teacher, n_ex,
                                         usar_registro=not self.sin_registro)
                row["text"] = txt
                row["scenario"] = sc["id"]
                row["cat"] = sc["cat"]
                row["voice"] = self.voice
                row["fallo"] = "" if txt else "no_se_pudo_armar"
            yield rows


class AnotarValidacion(Step):
    """Gate: el validador de la referencia + las reglas de purga de R1.

    Anota `_errores` y no descarta: distilabel conserva el numero de filas entre
    Steps, asi que el filtrado real ocurre en el sink.
    """

    @property
    def inputs(self) -> "StepColumns":
        return ["text", "escenario"]

    @property
    def outputs(self) -> "StepColumns":
        return ["_errores"]

    def process(self, *inputs: StepInput) -> "StepOutput":
        for rows in inputs:
            for row in rows:
                row["_errores"] = (
                    ["muestra_vacia"]
                    if not row["text"]
                    else validar(row["text"], row["scenario"], row["voice"])
                )
            yield rows


def validar(txt: str, escenario_id: str, voice: str) -> list[str]:
    """`ref.validate` con los knobs POR ESCENARIO + las reglas de R1.

    El mapeo de knobs es el de `main()` de la referencia, para que el gate sea
    el mismo y no una segunda version de las reglas.
    """
    sc = next((s for s in ref.SCENARIOS if s["id"] == escenario_id), None)
    if sc is None:
        return [f"escenario_desconocido:{escenario_id}"]
    tool_oblig = sc.get("tool_obligatoria", sc.get("tools", 0) > 0)
    errs = ref.validate(
        txt,
        voice=voice,
        min_turns=sc.get("min_turns", 2),
        require_voseo=sc.get("voseo", True),
        min_palabras_usuario=sc.get("min_palabras_usuario", 3),
        require_tool=tool_oblig,
        prohibir_tool=sc.get("prohibir_tool", False),
        is_chess=sc.get("is_chess", sc.get("id") == "chess_uci"),
        permitir_spanglish=sc.get("permitir_spanglish", False),
        ancla_payload=sc.get("ancla_payload"),
        ancla_min=sc.get("ancla_min", 2),
    )
    return errs + [f"purga:{r}" for r in purge.evaluar(txt)]


class EscribirDataset(Step):
    """Sink: escribe el dataset aceptado y los rechazos con su motivo."""

    out_path: str
    rejects_path: str

    _out: Any = PrivateAttr(default=None)
    _rej: Any = PrivateAttr(default=None)
    _ok: int = PrivateAttr(default=0)
    _rech: int = PrivateAttr(default=0)
    _motivos: Counter = PrivateAttr(default_factory=Counter)

    @property
    def inputs(self) -> "StepColumns":
        return ["text", "scenario", "cat", "voice", "_errores"]

    @property
    def outputs(self) -> "StepColumns":
        return []

    def _abrir(self) -> None:
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.rejects_path).parent.mkdir(parents=True, exist_ok=True)
        self._out = Path(self.out_path).open("w", encoding="utf-8")
        self._rej = Path(self.rejects_path).open("w", encoding="utf-8")

    def process(self, inputs: StepInput) -> "StepOutput":
        if self._out is None:
            self._abrir()
        for row in inputs:
            errs = row["_errores"]
            if errs:
                self._rech += 1
                for e in errs:
                    self._motivos[e.split(":")[0].split("(")[0]] += 1
                self._rej.write(
                    json.dumps({"scenario": row["scenario"], "errors": errs,
                                "text": row["text"]}, ensure_ascii=False) + "\n"
                )
                continue
            self._ok += 1
            self._out.write(
                json.dumps({k: row[k] for k in ("text", "scenario", "cat", "voice")},
                           ensure_ascii=False) + "\n"
            )
        yield inputs

    def unload(self) -> None:
        for fh in (self._out, self._rej):
            if fh is not None:
                fh.flush()
                fh.close()
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        stats = Path(self.out_path).with_suffix(".stats.json")
        stats.write_text(
            json.dumps(
                {"aceptadas": self._ok, "rechazadas": self._rech,
                 "motivos": dict(self._motivos)},
                ensure_ascii=False, indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        self._out = self._rej = None
        super().unload()


# --------------------------------------------------------------------------- #
# 6. main
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TeacherHTTP:
    """El teacher de freellmapi. Pickleable a proposito: distilabel corre cada
    Step en su propio proceso y una closure de `main()` no se puede mandar.

    `auto` rota proveedor upstream, asi que un 401 es TRANSITORIO: el mismo
    pedido al mismo instante responde 200. `ref.generate` re-lanza el 401 (solo
    reintenta 429/502/503), asi que el reintento va aca.
    """

    base_url: str
    model: str
    key: str | None
    reasoning: str = "none"

    def __call__(self, prompt: str) -> str:
        for intento in range(REINTENTOS_401):
            try:
                return ref.generate(self.base_url, self.model, prompt, self.key,
                                    reasoning=self.reasoning)
            except urllib.error.HTTPError as e:
                if e.code != 401 or intento == REINTENTOS_401 - 1:
                    raise
                time.sleep(3 * (intento + 1))
        raise RuntimeError("inalcanzable")


def elegir_escenarios(n: int, seed: int | None, solo: list[str] | None) -> list[dict]:
    if solo:
        por_id = {sc["id"]: sc for sc in ref.SCENARIOS}
        faltan = [i for i in solo if i not in por_id]
        if faltan:
            raise SystemExit(f"escenario inexistente: {faltan}")
        return [por_id[i] for i in solo]
    if n <= 0:
        return list(ref.SCENARIOS)
    rng = random.Random(seed)
    return [rng.choice(ref.SCENARIOS) for _ in range(n)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="R2: generacion turnwise con estado autoritativo")
    ap.add_argument("--n", type=int, default=2, help="cuantas muestras")
    ap.add_argument("--voice", default="seco")
    ap.add_argument("--model", default="auto", help="modelo del teacher (auto de freellmapi)")
    ap.add_argument("--base-url", default="http://localhost:3001/v1")
    ap.add_argument("--key", default=None, help="default: la lee del log de freellmapi")
    ap.add_argument("--out", default="pipelines/smoke/tc_r2.jsonl")
    ap.add_argument("--rejects", default=None, help="default: <out>_rejects.jsonl")
    ap.add_argument("--reasoning", default="none", choices=["none", "minimal", "low", "medium", "high"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sin-registro", action="store_true",
                    help="brazo OFF del A/B: no inyecta prompts/prompt-compartido.md")
    ap.add_argument("--scenario", action="append", default=None,
                    help="id de escenario, repetible. Ignora --n (smoke reproducible).")
    args = ap.parse_args(argv)

    rejects = args.rejects or str(Path(args.out).with_suffix("")) + "_rejects.jsonl"
    key = args.key or ref.get_freellm_key()
    if not key:
        print("[!] no encontre la key de freellmapi; pasala con --key", flush=True)

    escenarios = elegir_escenarios(args.n, args.seed, args.scenario)
    cache = Path(os.environ.get("GEN_CACHE_DIR", "/run/media/chaos/secundario/gen-cache"))
    print(f"generando {len(escenarios)} muestras (teacher={args.model})", flush=True)

    with Pipeline(name="kateto-gen", cache_dir=cache) as pipeline:
        load = LoadDataFromDicts(
            name="load_escenarios", data=[{"escenario": sc} for sc in escenarios], batch_size=4
        )
        generar = GenerarMuestras(
            name="generar_turnwise", input_batch_size=4, voice=args.voice,
            base_url=args.base_url, model=args.model, teacher_key=key, reasoning=args.reasoning,
            sin_registro=args.sin_registro,
        )
        gate = AnotarValidacion(name="anotar_validacion", input_batch_size=4)
        sink = EscribirDataset(
            name="escribir_dataset", input_batch_size=4,
            out_path=args.out, rejects_path=rejects,
        )
        load >> generar >> gate >> sink
    pipeline.run(use_cache=False)

    stats = Path(args.out).with_suffix(".stats.json")
    print(f"\n=== resumen ===\n  dataset: {args.out}\n  rechazos: {rejects}\n  {stats.read_text()}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())