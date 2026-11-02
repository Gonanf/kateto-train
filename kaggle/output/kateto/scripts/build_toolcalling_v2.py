"""Pipeline dataset v2: plantillas deterministas + filtro anti tecnicismos.
Genera data/toolcalling_kateto_v2.jsonl en formato dual openai/rwkv.
Uso: python3 scripts/build_toolcalling_v2.py [--limit N]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

TDIR = Path(__file__).resolve().parent.parent
DST = TDIR / "data" / "toolcalling_kateto_v2.jsonl"

FORBIDDEN = [
    "deleg", "orquest", "workflow", "backlog", "planning", "planific",
    "kubernetes", "microservicio", "refactor", "sprint", "standup",
    "como IA", "como modelo", "mi equipo", "yo lidero", "yo delego",
    "cron", "pipeline", "endpoint",
]

SYS = "Sos Kateto, hablas rioplatense corto y seco. Pedis ayuda directo, sin tecnicismos."


def check_natural(*texts: str) -> str | None:
    low = " ".join(texts).lower()
    for w in FORBIDDEN:
        if w in low:
            return w
    return None


def rec(question: str, think: str, tool: str, args: dict, resp: dict, final: str):
    hit = check_natural(question, final)
    if hit:
        raise ValueError(f"tecnicismo filtrado: {hit} en {question[:60]}")
    tc_json = json.dumps({"name": tool, "arguments": args}, ensure_ascii=False)
    tr_json = json.dumps(resp, ensure_ascii=False)
    answer = (
        f"<think>\n{think}\n</think>\n<tool_call>\n{tc_json}\n</tool_call>\n"
        f"<tool_response>\n{tr_json}\n</tool_response>\n{final}"
    )
    openai = {"messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": question},
        {"role": "assistant", "content": tc_json,
         "tool_calls": [{"id": "call_0", "type": "function",
                         "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": tr_json},
        {"role": "assistant", "content": final},
    ]}
    rwkv = {"think": think, "tool_call": {"name": tool, "arguments": args},
            "tool_response": resp, "final": final}
def rec_direct(question: str, think: str, final: str, label: str = "toolcalling-v2-turntaking"):
    answer = f"<think>\n{think}\n</think>\n{final}" if think else final
    openai = {"messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": question},
        {"role": "assistant", "content": final}
    ]}
    rwkv = {"think": think, "final": final}
    return {"question": question, "answer": answer, "query": question,
            "response": final, "source_label": label,
            "tool": None, "arguments": {}, "openai": openai, "rwkv": rwkv}


def items():
    out = []
    out.append(rec(
        "Doktor, metete al minecraft y ayudame a minar un rato",
        "Pide ayuda directa, la paso a doktor sin vueltas.",
        "request_generation",
        {"target_voice": "doktor", "prompt": "metete al minecraft y ayudame a minar un rato"},
        {"ok": True, "queued": True},
        "Dale, ya le avise a doktor que se meta a minar con vos.",
    ))
    out.append(rec(
        "Guardame esta lista de materiales en la carpeta del server",
        "Fallo por falta de path, corrijo y reintento en el mismo turno.",
        "write_file",
        {"path": "server/materiales.txt", "content": "madera x20, piedra x10, pan x5"},
        {"ok": True, "tool": "write_file"},
        "Listo, quedo guardada la lista en el server.",
    ))
    out.append(rec(
        "Che, Erica, que haces con dos cabezas y un zapato en la mano",
        "Salio solo por quilombo en el chat, sin que nadie lo pida. Reviso que hay pendiente.",
        "send_event",
        {"event": "list_events", "arguments": {"game": "minecraft", "limit": 5}},
        {"ok": True, "events": [{"type": "game_start", "server": "127.0.0.1:25565"}]},
        "Jaja, Erica. Ya me fijo que esta pasando en el server.",
    ))
    out.append(rec(
        "Anotame lo que tenemos en el cofre para no olvidarnos",
        "Leo, guardo y anoto, todo encadenado.",
        "read_file",
        {"path": "server/cofre.txt"},
        {"ok": True, "content": "madera x20, piedra x10"},
        "Ya lo lei y lo deje anotado para que no se pierda.",
    ))
    out.append(rec(
        "Juegan al ajedrez? Me toca mover a mi con blancas",
        "Es turno de ajedrez, respondo solo con la jugada valida.",
        "send_event",
        {"event": "game_action", "arguments": {"uci": "g1f3", "request_id": "77"}},
        {"ok": True, "tool": "game_action"},
        "Juego caballo a f3 y listo.",
    ))
    # Categoria voice turn-taking / addressability:
    out.append(rec_direct(
        "Che má, pasame el mate porfa",
        "Side-speech dirigido a otra persona en la sala.",
        "<NO_RESPONSE><|endoftext|>",
        "voice-side-speech"
    ))
    out.append(rec_direct(
        "Kateto, me parece que mañana vamos a...",
        "Oracion cortada a la mitad por pausa para pensar.",
        "<WAIT><|endoftext|>",
        "voice-incomplete"
    ))
    out.append(rec_direct(
        "Subtítulos realizados por la comunidad de Amara.org",
        "Alucinacion de Whisper en silencios.",
        "<NO_RESPONSE><|endoftext|>",
        "voice-asr-hallucination"
    ))
    return out


def main(limit: int = 5):
    rows = items()[:limit]
    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"ok {len(rows)} -> {DST}")


if __name__ == "__main__":
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 5
    main(lim)
