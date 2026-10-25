"""
generate_v2_with_agy.py
Usa agy con gemini-3.8-flash-low para generar ejemplos adicionales del dataset v2
siguiendo estrictamente las directivas de README.orig.md, toolcalling_expansion_brief.md,
patrones_habla.md y voice_turn_taking_no_response_spec.md.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VDIR = ROOT / "data" / "toolcalling_v2"

FORBIDDEN_TEC = [
    "deleg", "orquest", "workflow", "backlog", "planning", "planific",
    "kubernetes", "microservicio", "como IA", "como modelo", "mi equipo",
    "yo lidero", "yo delego", "cron", "pipeline", "endpoint"
]
FORBIDDEN_SYC = [
    "claro que si", "con gusto", "ya mismo", "por supuesto",
    "lo que necesites", "encantado", "seria un placer", "a tu disposicion"
]
SUBMISSION = [
    "te parece", "lo probamos", "probamos mas tranqui",
    "que querias hacer posta", "te va?", "lo vemos?", "dale?"
]
REAL_TOOLS = {
    "read_file", "write_file", "send_event", "request_generation",
    "list_events", "schedule_event", "get_current_time", "delete_file",
    "enable_plugin", "disable_plugin", "list_plugins", "game_action"
}
SYS = "Sos Kateto, un asistente de equipo con voz argentina, canchero y humano. Haces chistes, referencias cultura y musica argentina, cerras tajante sin pedir permiso."

def validate_item(q: str, final: str, tool: str | None = None) -> str | None:
    low = (q + " " + final).lower()
    for w in FORBIDDEN_TEC:
        if w in low:
            return f"tecnicismo: {w}"
    for w in FORBIDDEN_SYC:
        if w in low:
            return f"sumision: {w}"
    if final.strip().endswith("?"):
        return "cierre con pregunta"
    for s in SUBMISSION:
        if s in low:
            return f"pregunta sumisa: {s}"
    if tool and tool not in REAL_TOOLS:
        return f"tool invalida: {tool}"
    return None

def build_record(q: str, think: str, tool: str | None, args: dict, resp: dict, final: str, label: str):
    if tool:
        tc = json.dumps({"name": tool, "arguments": args}, ensure_ascii=False)
        tr = json.dumps(resp, ensure_ascii=False)
        answer = (f"<think>\n{think}\n</think>\n<tool_call>\n{tc}\n</tool_call>\n"
                  f"<tool_response>\n{tr}\n</tool_response>\n{final}")
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": q},
            {"role": "assistant", "content": tc,
             "tool_calls": [{"id": "call_0", "type": "function",
                             "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)}}]},
            {"role": "tool", "tool_call_id": "call_0", "content": tr},
            {"role": "assistant", "content": final}
        ]
        rwkv = {"think": think, "tool_call": {"name": tool, "arguments": args},
                "tool_response": resp, "final": final}
    else:
        answer = f"<think>\n{think}\n</think>\n{final}" if think else final
        msgs = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": q},
            {"role": "assistant", "content": final}
        ]
        rwkv = {"think": think, "final": final}
    return {
        "question": q, "answer": answer, "query": q,
        "response": final, "source_label": label,
        "tool": tool, "arguments": args or {},
        "openai": {"messages": msgs}, "rwkv": rwkv
    }

def call_agy(prompt: str) -> list[dict]:
    cmd = [
        "agy", "-p", prompt,
        "--model", "gemini-3.8-flash-low",
        "--disable-slash-commands"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res.returncode != 0:
        print(f"Error agy: {res.stderr}")
        return []
    raw = res.stdout.strip()
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    except Exception as e:
        print(f"JSON parse error: {e}\nRaw output:\n{raw[:300]}")
        return []

def run_generation():
    # 1. Turn taking batch
    prompt_tt = """
Generá exactamente 25 ejemplos en formato JSON array para entrenar turn-taking y addressability en un asistente de voz argentino (Kateto).
Reglas:
- 10 ejemplos de 'side_speech': El usuario le habla a un familiar, compañero o mascota en la misma habitación, o ruidos/quejas solitarias ("Che má pasame la sal", "Apagá la luz", "Vení firulais", "¿Dónde dejé la SUBE?").
  Para estos: response DEBE ser estrictamente "<NO_RESPONSE><|endoftext|>", tool null.
- 10 ejemplos de 'incomplete': El usuario titubea, duda o corta la frase a la mitad ("Che Kateto, me parece que mañana...", "Ehhh pará que abro el...", "Fijate si en el cofre... aguantá").
  Para estos: response DEBE ser estrictamente "<WAIT><|endoftext|>", tool null.
- 5 ejemplos de 'positive': El usuario le habla directamente a Kateto con una pregunta completa y cotidiana rioplatense ("Kateto, ¿qué hora es?", "Che Kateto, ¿estás ahí?").
  Para estos: response DEBE ser una respuesta rioplatense canchera, seca, sin cerrar con pregunta ("Son las cinco menos cuarto, che.", "Acá estoy firme, decime.").

Devolvé ÚNICAMENTE un JSON array con objetos que tengan:
{
  "question": "...",
  "think": "breve explicacion del contexto",
  "response": "...",
  "label": "turn-taking-side-speech" | "turn-taking-incomplete" | "turn-taking-positive"
}
"""
    print("--- Generando batch Turn-Taking con agy ---")
    items_tt = call_agy(prompt_tt)
    added_tt = 0
    tt_file = VDIR / "voice_turn_taking.jsonl"
    existing_q = set()
    if tt_file.exists():
        for l in open(tt_file, encoding="utf-8"):
            if l.strip():
                existing_q.add(json.loads(l)["question"])
    
    with tt_file.open("a", encoding="utf-8") as f:
        for it in items_tt:
            q = it.get("question", "").strip()
            resp = it.get("response", "").strip()
            lbl = it.get("label", "turn-taking-v2")
            think = it.get("think", "")
            if not q or not resp or q in existing_q:
                continue
            err = validate_item(q, resp, None)
            if err:
                print(f"Descartado turn-taking: {err} -> {q}")
                continue
            rec = build_record(q, think, None, {}, {}, resp, lbl)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing_q.add(q)
            added_tt += 1
    print(f"Turn-Taking agregados: {added_tt}")

    # 2. Core Tool-calling batch
    prompt_tc = """
Generá exactamente 15 ejemplos en formato JSON array de tool-calling para Kateto (asistente de equipo con voz argentina y seco).
Reglas duras:
- Las tools permitidas son ÚNICAMENTE: "read_file", "write_file", "send_event", "request_generation", "list_events", "schedule_event", "get_current_time".
- NUNCA inventar tools como "websearch", "doktor_tool", "minecraft_craft".
- Para pedirle algo a otra voz (Doktor, Jane, Conquest) se usa "request_generation" con {"target_voice": "doktor"|"jane"|"conquest", "prompt": "..."}.
- El usuario habla en rioplatense directo pidiendo acciones, NUNCA planes ni código ("Doktor, metete al server", "Anotame la lista en notas.txt").
- La respuesta final del asistente debe ser rioplatense tajante, sin sumisión (prohibido "claro que sí", "con gusto"), sin cerrar con pregunta ("te parece?").

Devolvé ÚNICAMENTE un JSON array con objetos:
{
  "question": "...",
  "think": "...",
  "tool": "read_file" | "write_file" | "send_event" | "request_generation" | "list_events" | "schedule_event" | "get_current_time",
  "arguments": {...},
  "tool_response": {...},
  "final": "...",
  "label": "core-toolcalling"
}
"""
    print("--- Generando batch Core Tool-calling con agy ---")
    items_tc = call_agy(prompt_tc)
    added_tc = 0
    core_file = VDIR / "core_v2.jsonl"
    if core_file.exists():
        for l in open(core_file, encoding="utf-8"):
            if l.strip():
                existing_q.add(json.loads(l)["question"])
    with core_file.open("a", encoding="utf-8") as f:
        for it in items_tc:
            q = it.get("question", "").strip()
            t = it.get("tool")
            args = it.get("arguments", {})
            resp = it.get("tool_response", {"ok": True})
            final = it.get("final", "").strip()
            think = it.get("think", "")
            lbl = it.get("label", "core-toolcalling")
            if not q or not final or q in existing_q:
                continue
            err = validate_item(q, final, t)
            if err:
                print(f"Descartado tool-calling: {err} -> {q}")
                continue
            rec = build_record(q, think, t, args, resp, final, lbl)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing_q.add(q)
            added_tc += 1
    print(f"Core Tool-calling agregados: {added_tc}")

    # 3. Comedians / Triggers / Dagger batch
    prompt_trig = """
Generá exactamente 15 ejemplos en formato JSON array de diálogos de Kateto con comedia y triggers culturales según el README:
- Dagger: Lógica formal callejera rioplatense, premisas vs validez, analogías con pitos o animales, docente seco sin permiso.
- Terry Davis: Código que anda de milagro -> no toques ring-0, sistema a 640x480 con 16 colores, intelecto divino del puntero.
- Ultrakill: Alguien soberbio en un juego -> megalómano en inglés o español violento ("A mere object defying the will of God?!").
- BoJack / Mr Peanutbutter: Discusión moral que entra en loop -> corte absurdo tipo "¡Erica! ¿Qué hacés con dos cabezas y un zapato en la mano?".
- NTVG / Rock rioplatense: Honestidad brutal y seca ("Mirá, te diría que le des una chance, pero no te va a gustar; chau.").
Reglas:
- Cero sumisión ("claro que sí" prohibido).
- NUNCA cerrar con pregunta.
- Sin tecnicismos corporativos ("workflow", "pipeline").

Devolvé ÚNICAMENTE un JSON array con objetos:
{
  "question": "...",
  "think": "...",
  "final": "...",
  "label": "voice-dagger" | "voice-triggers" | "voice-gamer" | "voice-streamer"
}
"""
    print("--- Generando batch Comedians / Triggers con agy ---")
    items_trig = call_agy(prompt_trig)
    added_trig = 0
    trig_file = VDIR / "voice_varied.jsonl"
    with trig_file.open("a", encoding="utf-8") as f:
        for it in items_trig:
            q = it.get("question", "").strip()
            final = it.get("final", "").strip()
            think = it.get("think", "")
            lbl = it.get("label", "voice-varied")
            if not q or not final or q in existing_q:
                continue
            err = validate_item(q, final, None)
            if err:
                print(f"Descartado trigger: {err} -> {q}")
                continue
            rec = build_record(q, think, None, {}, {}, final, lbl)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            existing_q.add(q)
            added_trig += 1
    print(f"Comedians/Triggers agregados: {added_trig}")

if __name__ == "__main__":
    run_generation()
