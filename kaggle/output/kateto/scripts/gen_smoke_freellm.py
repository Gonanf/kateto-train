#!/usr/bin/env python
"""Smoke test: freellmapi produce conversaciones multi-turno con tool calling
en registro argentino y en el formato exacto del runtime?

Antes de construir el pipeline entero del dataset, se verifica lo mas riesgoso.
Criterio de exito: tool names REALES, formato `tool_call {n}: {json}` correcto,
voseo, y NINGUN resultado inventado por la tool.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

API = "http://127.0.0.1:3001/v1/chat/completions"
KEYFILE = Path("/tmp/freellmapi.log")


def get_key() -> str:
    m = re.search(r"freellmapi-[a-f0-9]{48}", KEYFILE.read_text(errors="ignore"))
    if not m:
        raise SystemExit("no encontre la key en el log")
    return m.group(0)


TOOLS = [
    ("request_generation", "target_voice", "prompt", "dept?"),
    ("read_file", "path"),
    ("write_file", "path", "content"),
    ("list_events", "-"),
    ("send_event", "event_name", "data?"),
    ("schedule_event", "event_name", "delay|interval|cron", "target_voice?"),
    ("refine_memory", "dept", "op(add|replace|remove)", "evidence", "fact?"),
    ("video_search", "query", "k?"),
    ("video_answer", "video", "query"),
    ("voyager_craft_stick", "- (solo minecraft + skill habilitado)"),
]

SYS = """Sos un generador de datos de entrenamiento para Kateto, un personaje de voz argentino.

Generá UNA conversacion multi-turno (2-3 intercambios) entre el usuario y la voz `seco`.

FORMATO EXACTO (respetalo al caracter, sin markdown, sin JSON de eventos):
<|im_user|>{lo que dice el usuario}
<|im_start|>seco
{respuesta de seco}<|im_end|>
<|im_user|>{siguiente}
<|im_start|>seco
{respuesta}<|im_end|>

REGLAS DURAS:
- seco habla RIOPLATENSE: voseo (tenes, sabes, querés, fijate), lunfardo (boludo, quilombo,
  pibe, laburo, posta). NUNCA "como asistente", nunca ofrecer ayuda, nunca disculparse servilmente.
- Si usa una tool, la sintaxis es EXACTAMENTE:
  tool_call {nombre}: {json con los argumentos}
  y despues, cuando vuelve el resultado:
  tool_result {nombre}: {resultado}
  Despues del tool_result, seco COMENTA el resultado en su voz y sigue la charla.
- Las tools disponibles son SOLO estas: %s
- NUNCA inventes el resultado de una tool: si no tenes el resultado, no lo escribas.
- Si una tool falla, el tool_result dice el error REAL (ej: "Error: file not found") y seco lo
  resuelve corrigiendo o se rinde sin inventar nada.
- NO escribas codigo, NO expliques arquitectura."""


def chat(model: str, prompt: str, key: str, temp: float = 1.0) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "max_tokens": 900,
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())


def main() -> int:
    key = get_key()
    model = sys.argv[1] if len(sys.argv) > 1 else "auto"
    tools_txt = "\n".join(f"  - {t[0]}({', '.join(x for x in t[1:] if x != '-')})" for t in TOOLS)
    prompt = SYS % tools_txt + """

ESCENARIO: El usuario esta en un stream y le pide a seco que averigue algo de un video
que vieron antes, y despues que le avise a otra voz (doktor) que revise una cosa.
Hace que seco use 2 tools reales, una encadenada con la otra."""

    print(f"modelo: {model}")
    r = chat(model, prompt, key)
    ch = r.get("choices", [{}])[0].get("message", {})
    txt = ch.get("content") or ""
    print("=" * 70)
    print(txt[:2000])
    print("=" * 70)

    # validaciones
    reales = {t[0] for t in TOOLS} | {"video_search", "video_list_videos", "video_describe",
                                     "video_describe_images", "video_answer", "video_summarize"}
    usadas = set(re.findall(r"tool_call\s+([a-zA-Z_]+)\s*:", txt)) | \
             set(re.findall(r"tool_result\s+([a-zA-Z_]+)\s*:", txt))
    VOSEO = re.compile(r"\b(tenes|sabes|queres|fijate|vos|boludo|posta|che)\b", re.I)
    VICIO = re.compile(r"(como asistente|en que puedo ayudarte|estoy para ayudarte|lamento)", re.I)
    print(f"tools usadas: {sorted(usadas)}")
    print(f"  todas reales? {'SI' if usadas <= reales else 'NO -> ' + str(usadas - reales)}")
    print(f"  tiene voseo?  {'SI' if VOSEO.search(txt) else 'NO'}")
    print(f"  tiene vicio de asistente? {'SI (MAL)' if VICIO.search(txt) else 'NO'}")
    print(f"  tiene tool_result inventado sin tool_call? {'SI (MAL)' if ('tool_result' in txt and 'tool_call' not in txt) else 'NO'}")
    print(f"  cantidad de turnos seco: {txt.count('<|im_start|>seco')}")
    print(f"  termina bien: {'SI' if txt.rstrip().endswith('<|im_end|>') else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
