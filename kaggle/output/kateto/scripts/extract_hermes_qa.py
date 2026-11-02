#!/usr/bin/env python3
"""
extract_hermes_qa.py — Extrae ~7k pares de conversación Q&A y tool-calling desde el clon de state.db de Hermes.
Aplica filtros de seguridad (secrets, credenciales, API keys), dedup, filtrado de logs/código excesivo y limpieza.
Salida: data/hermes_qa_7k.jsonl
"""
import os, sys, re, json, shutil, sqlite3
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parents[1]
LIVE_DB = Path.home() / ".hermes" / "state.db"
CLONE_DB = PROJECT / "data" / "state.clone.db"
OUT_JSONL = PROJECT / "data" / "hermes_qa_7k.jsonl"

# Agregar Kateto tools para reusar filtros
KATETO_TOOLS = Path.home() / "proyectos" / "OpenaiBuildWeek" / "Kateto"
if str(KATETO_TOOLS) not in sys.path:
    sys.path.insert(0, str(KATETO_TOOLS))

try:
    from kateto.tools.dataset_filter.secrets_filter import contains_secret
    from kateto.tools.dataset_filter.ai_filter import is_ai_like
    print("Filtros de Kateto cargados exitosamente (secrets + ai).")
except Exception as e:
    print(f"Aviso: no se pudo cargar kateto tools ({e}), usando filtros regex locales.")
    def contains_secret(text):
        class R: found = False; reasons = []
        if re.search(r"(sk-[A-Za-z0-9_\-]{20,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z\-_]{35}|password\s*[:=]|client_secret)", text, re.IGNORECASE):
            r = R(); r.found = True; r.reasons = ["api_key"]; return r
        return R()
    def is_ai_like(text):
        return False

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Quitar prefijos de usuario comunes de hermes como [Soloto], [User], etc.
    text = re.sub(r"^\[(Soloto|User|Admin|Chaos)\]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    return text

HARNESS_PATTERNS = [
    "you just executed tool calls",
    "[system note",
    "[important:",
    "[cron:",
    "[context",
    "[reminder",
]

def is_garbage_or_dump(text: str) -> bool:
    low = text.lower()
    if any(h in low for h in HARNESS_PATTERNS):
        return True
    # Descartar salidas de diff gigantes, tracebacks kilométricos o dumps binarios
    if text.count("\n") > 70:
        return True
    if text.count("--- a/") > 1 and text.count("+++ b/") > 1:
        return True
    if "Traceback (most recent call last):" in text and len(text) > 1000:
        return True
    return False

def main():
    print(f"=== 1/3 Verificando / Actualizando clon de state.db ===")
    if LIVE_DB.exists():
        print(f"Copiando {LIVE_DB} ({LIVE_DB.stat().st_size / 1024**2:.1f} MB) -> {CLONE_DB}...")
        shutil.copyfile(LIVE_DB, CLONE_DB)
        print("Clon actualizado.")
    else:
        if not CLONE_DB.exists():
            print(f"Error: no existe {LIVE_DB} ni {CLONE_DB}")
            sys.exit(1)
        print(f"Usando clon existente en {CLONE_DB}")

    print(f"\n=== 2/3 Extrayendo mensajes ordenados por sesión ===")
    conn = sqlite3.connect(str(CLONE_DB))
    cur = conn.cursor()
    cur.execute("SELECT id, session_id, role, content, tool_calls FROM messages ORDER BY session_id, id ASC")
    rows = cur.fetchall()
    print(f"Total mensajes leídos de DB: {len(rows)}")

    sessions = defaultdict(list)
    for r in rows:
        sessions[r[1]].append(r)

    print(f"Total sesiones agrupadas: {len(sessions)}")

    print(f"\n=== 3/3 Filtrando, extrayendo QA y tool calls ===")
    qa_items = []
    tool_items = []
    seen_qa = set()
    seen_tc = set()
    dropped_secrets = 0
    dropped_garbage = 0

    for sid, s_msgs in sessions.items():
        last_user = None
        # Recorremos la conversación
        for mid, _, role, content, tool_calls in s_msgs:
            c_text = clean_text(content or "")

            if role == "user":
                if 5 <= len(c_text) <= 3000 and not is_garbage_or_dump(c_text):
                    last_user = c_text
                else:
                    last_user = None

            elif role == "assistant":
                if not last_user:
                    continue

                # 1. Caso Tool-Call
                if tool_calls:
                    try:
                        tdata = json.loads(tool_calls)
                        if isinstance(tdata, list) and len(tdata) > 0:
                            first = tdata[0]
                            name = first.get("function", {}).get("name") or first.get("name") or "tool"
                            args = first.get("function", {}).get("arguments") or first.get("arguments") or {}
                            ans = f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(args, ensure_ascii=False)}}}\n</tool_call>'
                            
                            key = (last_user[:100].lower(), name)
                            if key not in seen_tc:
                                seen_tc.add(key)
                                if contains_secret(last_user).found or contains_secret(ans).found:
                                    dropped_secrets += 1
                                else:
                                    tool_items.append({
                                        "question": last_user,
                                        "answer": ans,
                                        "query": last_user,
                                        "response": ans,
                                        "source_label": "hermes-tool-call",
                                        "session_id": sid,
                                        "openai": {
                                            "messages": [
                                                {"role": "system", "content": "Sos Kateto, rioplatense seco, con opinion propia."},
                                                {"role": "user", "content": last_user},
                                                {"role": "assistant", "content": ans}
                                            ]
                                        }
                                    })
                    except Exception:
                        pass

                # 2. Caso Respuesta Conversacional
                if c_text and 8 <= len(c_text) <= 3500:
                    if is_garbage_or_dump(c_text):
                        dropped_garbage += 1
                        continue

                    key = (last_user[:100].lower(), c_text[:100].lower())
                    if key not in seen_qa:
                        seen_qa.add(key)
                        if contains_secret(last_user).found or contains_secret(c_text).found:
                            dropped_secrets += 1
                        else:
                            qa_items.append({
                                "question": last_user,
                                "answer": c_text,
                                "query": last_user,
                                "response": c_text,
                                "source_label": "hermes-chat-qa",
                                "session_id": sid,
                                "openai": {
                                    "messages": [
                                        {"role": "system", "content": "Sos Kateto, rioplatense seco, con opinion propia."},
                                        {"role": "user", "content": last_user},
                                        {"role": "assistant", "content": c_text}
                                    ]
                                }
                            })

    print(f"Extracción preliminar:")
    print(f"  - QA conversacionales únicos limpios: {len(qa_items)}")
    print(f"  - Tool calls únicos limpios: {len(tool_items)}")
    print(f"  - Descartados por secretos: {dropped_secrets}")
    print(f"  - Descartados por logs / dumps: {dropped_garbage}")

    # Seleccionar ~7000 items balanceados (ej. ~5000 QA + ~2000 tool calls)
    import random
    random.seed(42)
    random.shuffle(qa_items)
    random.shuffle(tool_items)

    target_qa = min(len(qa_items), 5000)
    target_tc = min(len(tool_items), 2000)
    final_items = qa_items[:target_qa] + tool_items[:target_tc]
    random.shuffle(final_items)

    print(f"\nSeleccionados para dataset final: {len(final_items)} items ({target_qa} QA + {target_tc} tool calls)")

    # Guardar en jsonl
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for it in final_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"Guardado exitoso en {OUT_JSONL} ({OUT_JSONL.stat().st_size / 1024**2:.2f} MB)")

if __name__ == "__main__":
    main()
