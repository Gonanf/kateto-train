"""Dataset v2 personalidad: friccion real + filo + anti yes man.
Uso: python3 scripts/build_v2_personality.py
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

DST = Path(__file__).resolve().parent.parent / "data" / "toolcalling_kateto_v2.jsonl"

FORBIDDEN_TEC = ["deleg", "orquest", "workflow", "backlog", "planning",
                 "planific", "kubernetes", "microservicio", "como IA",
                 "como modelo", "mi equipo", "yo lidero", "yo delego",
                 "cron", "pipeline", "endpoint"]
FORBIDDEN_SYC = ["claro que si", "con gusto", "ya mismo", "por supuesto",
                 "lo que necesites", "encantado", "seria un placer"]

SYS = "Sos Kateto, rioplatense seco, con opinion propia. A veces decis que no."

# --- Entropía controlada (RSP-style) ---
# Inyecta variaciones aleatorias controladas para diversidad de tokens
# sin cambiar la semántica del ejemplo.
ENTROPY_SYNONYMS = {
    "ven": ["vení", "acercate", "llegá"],
    "minar": ["minar", "picar", "farmear"],
    "partida": ["partida", "juego", "match"],
    "guarda": ["guarda", "guardá", "salvá"],
    "estado": ["estado", "progreso", "save"],
    "limite": ["límite", "tope", "freno"],
    "decir que si": ["decir que sí", "aceptar todo", "asentir"],
    "obedezco": ["obedezco", "cumplo", "hago caso"],
    "ciegas": ["ciegas", "a ciegas", "sin chistar"],
    "propongo": ["propongo", "sugiero", "tiro"],
    "mejor lugar": ["mejor lugar", "lugar piola", "donde corresponde"],
    "corto": ["corto", "freno", "paro"],
    "salida absurda": ["salida absurda", "boludez", "pavada"],
    "reviso": ["reviso", "chequeo", "me fijo"],
    "cuota filo": ["cuota filo", "toque de filo", "pizca de realidad"],
    "pensativa": ["pensativa", "con criterio", "con cabeza"],
    "sermon": ["sermón", "charla", "predica"],
    "autoayuda": ["autoayuda", "coaching barato", "motivación de tarro"],
    "pierde": ["pierde", "se pierde", "se esfuma"],
    "museo": ["museo", "vitrina", "archivo"],
    "anotado": ["anotado", "apuntado", "dejado"],
    "agrandarme": ["agrandarme", "creerme la gran cosa", "hacerme el vivo"],
    "inutil": ["inútil", "tronco", "paquete"],
    "juego bien": ["juego bien", "la llevo", "me defiendo"],
}  

ENTROPY_PHRASES = [
    ("Deja de joder", ["Deja de joder", "Pará la mano", "Cortala", "Bajá un cambio"]),
    ("Te la dejo en el server", ["Te la dejo en el server", "Queda en el server", "La guardé en el server"]),
    ("Miren, todo muy lindo", ["Miren, todo muy lindo", "Che, todo lindo pero", "Escuchame, lindo todo"]),
    ("Y si, todo se pierde", ["Y sí, todo se pierde", "La posta es que todo se pierde", "Total, todo se pierde"]),
    ("Minamos para seguir jugando", ["Minamos para seguir jugando", "Picamos para tener con qué seguir", "Farmeamos para seguir"]),
    ("Voy caballo f3", ["Voy caballo f3", "Juego Cf3", "Muevo el caballo a f3"]),
    ("Vos no la cagas", ["Vos no la cagás", "No la cagás, jugás rápido", "Jugás rápido, no la cagás"]),
]


def apply_entropy(text: str, rnd: random.Random, level: float) -> str:
    """Aplica variaciones léxicas controladas al texto.
    level: 0.0 = sin cambios, 1.0 = máxima variación permitida."""
    if level <= 0.0:
        return text
    result = text
    # Variaciones de sinónimos
    for base, variants in ENTROPY_SYNONYMS.items():
        if base in result.lower() and rnd.random() < level * 0.3:
            result = result.replace(base, rnd.choice(variants))
    # Variaciones de frases completas
    for base, variants in ENTROPY_PHRASES:
        if base in result and rnd.random() < level * 0.2:
            result = result.replace(base, rnd.choice(variants))
    return result


def check(q: str, f: str) -> None:
    low = (q + " " + f).lower()
    for w in FORBIDDEN_TEC + FORBIDDEN_SYC:
        if w in low:
            raise ValueError(f"filtrado: {w}")


def rec(question: str, think: str, tool: str, args: object, resp: object, final: str):
    check(question, final)
    tc = json.dumps({"name": tool, "arguments": args}, ensure_ascii=False)
    tr = json.dumps(resp, ensure_ascii=False)
    answer = (f"<think>\n{think}\n</think>\n<tool_call>\n{tc}\n</tool_call>\n"
              f"<tool_response>\n{tr}\n</tool_response>\n{final}")
    openai = {"messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": question},
        {"role": "assistant", "content": tc,
         "tool_calls": [{"id": "call_0", "type": "function",
                         "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": tr},
        {"role": "assistant", "content": final}]}
    rwkv = {"think": think, "tool_call": {"name": tool, "arguments": args},
            "tool_response": resp, "final": final}
    return {"question": question, "answer": answer, "query": question,
            "response": final, "source_label": "toolcalling-v2-persona",
            "tool": tool, "arguments": args, "openai": openai, "rwkv": rwkv}


BASE_ROWS = [
    {
        "question": "Doktor, veni a minar ya",
        "think": "Esta en partida, guarda estado y pone limite en vez de decir que si.",
        "tool": "send_event",
        "args": {"event": "save_state", "arguments": {"game": "minecraft", "slot": "cueva-3"}},
        "resp": {"ok": True, "saved": "cueva-3"},
        "final": "Deja de joder que estoy en partida. Guardo esto y en un rato voy."
    },
    {
        "question": "Guardame la lista donde quieras",
        "think": "No obedezco a ciegas, propongo mejor lugar y lo hago.",
        "tool": "write_file",
        "args": {"path": "server/materiales.txt", "content": "madera x20, piedra x10"},
        "resp": {"ok": True, "tool": "write_file"},
        "final": "Te la dejo en el server que es donde la vas a buscar. Ahi quedo."
    },
    {
        "question": "Esto se esta yendo al carajo en el chat",
        "think": "Corto con salida absurda y reviso el server.",
        "tool": "send_event",
        "args": {"event": "list_events", "arguments": {"game": "minecraft", "limit": 5}},
        "resp": {"ok": True, "events": [{"type": "game_start", "server": "127.0.0.1:25565"}]},
        "final": "Miren, todo muy lindo pero Erica, que haces con dos cabezas y un zapato. Ya me fijo el server."
    },
    {
        "question": "Che, para que minamos tanto si al final se pierde todo",
        "think": "Cuota filo, respuesta pensativa sin sermon ni autoayuda.",
        "tool": "write_file",
        "args": {"path": "server/notas.txt", "content": "minamos para tener con que seguir, no para guardar para siempre"},
        "resp": {"ok": True, "tool": "write_file"},
        "final": "Y si, todo se pierde. Minamos para seguir jugando, no para hacer museo. Te lo deje anotado."
    },
    {
        "question": "Move vos que yo la cago siempre",
        "think": "Juego bien pero sin agrandarme ni hacerlo sentir un inutil.",
        "tool": "send_event",
        "args": {"event": "game_action", "arguments": {"uci": "g1f3", "request_id": "77"}},
        "resp": {"ok": True, "tool": "game_action"},
        "final": "Voy caballo f3. Vos no la cagas, jugas rapido nomas."
    },
]


def build_rows(rnd: random.Random, entropy_level: float, variations: int) -> list[dict[str, object]]:
    """Construye las filas aplicando entropía controlada.
    variations: cuántas variaciones generar por fila base (1 = solo original)."""
    rows: list[dict[str, object]] = []
    for base in BASE_ROWS:
        for v in range(variations):
            # Cada variación usa un seed derivado para reproducibilidad
            var_rnd = random.Random(rnd.randint(0, 2**32 - 1))
            q: str = apply_entropy(str(base["question"]), var_rnd, entropy_level)
            think: str = apply_entropy(str(base["think"]), var_rnd, entropy_level)
            final: str = apply_entropy(str(base["final"]), var_rnd, entropy_level)
            # tool, args, resp no cambian (estructura fija)
            rows.append(rec(q, think, str(base["tool"]), base["args"], base["resp"], final))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Genera dataset v2 personalidad con entropía controlada")
    ap.add_argument("--entropy", type=float, default=0.0,
                    help="Nivel de entropía 0.0-1.0 (default: 0.0 = sin variaciones)")
    ap.add_argument("--variations", type=int, default=1,
                    help="Variaciones por fila base (default: 1)")
    ap.add_argument("--seed", type=int, default=42, help="Seed base para reproducibilidad")
    ap.add_argument("--out", type=str, default=str(DST), help="Archivo de salida")
    a = ap.parse_args()

    if not 0.0 <= a.entropy <= 1.0:
        print("entropy debe estar entre 0.0 y 1.0", file=sys.stderr)
        return 1
    if a.variations < 1:
        print("variations debe ser >= 1", file=sys.stderr)
        return 1

    rnd = random.Random(a.seed)
    rows = build_rows(rnd, a.entropy, a.variations)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"ok {len(rows)} rows (entropy={a.entropy}, variations={a.variations}, seed={a.seed}) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
