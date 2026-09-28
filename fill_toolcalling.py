"""Rellena responses vacíos de toolcalling_sft.jsonl usando Orion-26B local.

Uso: python3 fill_toolcalling.py [--limit N]
Lee data/toolcalling_sft.jsonl, escribe data/toolcalling_filled.jsonl
(resume: salta queries ya presentes en el output).
Endpoint directo al backend llama-server para evitar overhead del gateway.
Secuencial: el server tiene parallel=1, la concurrencia no sirve.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

TDIR = Path(__file__).resolve().parent
SRC = TDIR / "data" / "toolcalling_sft.jsonl"
DST = TDIR / "data" / "toolcalling_filled.jsonl"
URL = "http://127.0.0.1:41007/completion"

TOOLS = (
    "send_event, list_events, list_plugins, enable_plugin, disable_plugin, "
    "create_workflow, update_workflow, create_skill, update_skill, read_file, "
    "write_file, run_command, get_weather, mcp_call"
)

def make_prompt(query: str) -> str:
    return (
        "You generate tool-calling training data. Available tools: " + TOOLS + ".\n"
        "User message (rioplatense Spanish): " + query + "\n"
        "Reply with JSON ONLY, exactly this shape:\n"
        '{"tool": "<tool name>", "arguments": {...}, '
        '"answer": "<short reply in rioplatense Spanish, dry, no fluff>"}'
    )


def call_orion(prompt: str, timeout: float = 300.0) -> str | None:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": 350,
        "temperature": 0.6,
        "top_k": 40,
        "top_p": 0.9,
        "repeat_penalty": 1.15,
        "cache_prompt": True,
    }).encode()
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                URL, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            return (d.get("content") or "").strip() or None
        except Exception as e:
            print(f"  intento {attempt + 1} falló: {e}", flush=True)
            time.sleep(5)
    return None


def parse_json(text: str) -> dict | None:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[: -3]
    start = t.find("{")
    end = t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(t[start:end + 1])
    except Exception:
        return None
    if not isinstance(obj, dict) or "tool" not in obj:
        return None
    return obj


def main(limit: int | None = None) -> None:
    rows = [json.loads(l) for l in SRC.open(encoding="utf-8") if l.strip()]
    done: dict[str, dict] = {}
    if DST.exists():
        for l in DST.open(encoding="utf-8"):
            if l.strip():
                try:
                    o = json.loads(l)
                    done[o["query"]] = o
                except Exception:
                    pass
    print(f"total={len(rows)} ya_hechos={len(done)}", flush=True)
    n_new = 0
    with DST.open("a", encoding="utf-8") as f:
        for r in rows:
            q = r.get("query", "")
            if not q or q in done:
                continue
            if limit is not None and n_new >= limit:
                break
            print(f"[{n_new + 1}] {q[:70]}", flush=True)
            t0 = time.time()
            txt = call_orion(make_prompt(q))
            if not txt:
                print("  sin respuesta, salto", flush=True)
                continue
            obj = parse_json(txt)
            if not obj:
                print(f"  JSON inválido: {txt[:120]}", flush=True)
                continue
            ans = str(obj.get("answer", "")).strip()
            out = {
                "query": q,
                "question": q,
                "response": ans,
                "answer": ans,
                "tool": obj.get("tool"),
                "arguments": obj.get("arguments", {}),
                "source_label": r.get("source_label", "toolcalling-orion"),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            done[q] = out
            n_new += 1
            print(f"  ok ({time.time() - t0:.0f}s) tool={out['tool']}", flush=True)
    print(f"hechos_nuevos={n_new}", flush=True)


if __name__ == "__main__":
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    main(lim)
