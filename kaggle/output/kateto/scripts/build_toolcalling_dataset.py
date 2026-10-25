#!/usr/bin/env python3
"""
build_toolcalling_dataset.py — Dataset toolcalling Kateto con descubrimiento real

Cubre: plugins (listar/activar/desactivar), eventos entre voces, skills, MCPs, workflows (crear/avanzar/terminar)
Descubre tools reales del repo Kateto (kateto/voices/tools.py BUILTIN_TOOLS + events) y skills en ~/.hermes/skills
Genera formato dual: OpenAI chat con tools y RWKV G1 con <think> + <tool_call> + <tool_response>
Salida compatible con to_sft.py: {question, answer} (answer = RWKV G1 serializado; incluye metadata openai/rwkv)

Uso:
  python scripts/build_toolcalling_dataset.py --output data/toolcalling_sft.jsonl --dry-run --limit 200
  python scripts/build_toolcalling_dataset.py --output data/toolcalling_sft.jsonl --limit 200  # real Orion
  OPENAI_BASE_URL=http://127.0.0.1:11434/v1 MODEL_NAME=Orion-26B python scripts/build_toolcalling_dataset.py --output data/toolcalling_sft.jsonl --limit 200

Si Orion ocupado, deja --dry-run verificado y avisa.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]
os.chdir(TRAIN_ROOT)
KATETO_ROOT = Path.home() / "proyectos" / "OpenaiBuildWeek" / "Kateto"


def discover_skills() -> list[str]:
    """Skills reales de Kateto en config/defaults/skills (directorio = skill id).""" 
    skills_dir = KATETO_ROOT / "config" / "defaults" / "skills"
    if not skills_dir.exists():
        return []
    out = []
    for p in skills_dir.iterdir():
        if p.name.startswith("."):
            continue
        if p.is_dir() and (p / "SKILL.md").exists():
            out.append(p.name)
        elif p.is_symlink():
            try:
                target = p.resolve()
                if (target / "SKILL.md").exists():
                    out.append(p.name)
            except: pass
        elif p.is_dir():
            # skills anidados sin SKILL.md en root pero con subskills?
            if any((p / sub).is_dir() for sub in os.listdir(p) if False):
                pass
            # incluir igual dirs con nombre válido
            if re.match(r"[a-z][a-z0-9\-]*$", p.name):
                out.append(p.name)
    return sorted(set(out))


def discover_plugins() -> list[str]:
    """Plugins reales de Kateto"""
    plug_dir = KATETO_ROOT / "kateto" / "plugins"
    if not plug_dir.exists():
        return ["audio_input", "audio_processor", "audio_output", "connector", "executor", "system"]
    return sorted([d.name for d in plug_dir.iterdir() if d.is_dir() and not d.name.startswith("_")])


def discover_voices() -> list[str]:
    return ["jane", "doktor", "conquest"]


def discover_builtin_tools() -> list[dict]:
    """Parsea BUILTIN_TOOLS de kateto/voices/tools.py para lista autoritativa"""
    tools_py = KATETO_ROOT / "kateto" / "voices" / "tools.py"
    if not tools_py.exists():
        return []
    txt = tools_py.read_text(encoding="utf-8", errors="ignore")
    # extraer bloques ChatCompletionToolParam con "name": "..."
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', txt)
    # deducir descr cortas
    return [{"name": n} for n in names]


def discover_events() -> list[str]:
    ev = KATETO_ROOT / "kateto" / "core" / "event.py"
    if not ev.exists():
        return []
    txt = ev.read_text(encoding="utf-8", errors="ignore")
    classes = re.findall(r"class (\w+Data)\(EventModel\)", txt)
    # convertir a event_name snake (GenerateData -> Generate)
    return sorted(classes)


def build_toolcalling_queries(skills: list[str], plugins: list[str], voices: list[str], events: list[str]) -> list[tuple[str, str, dict]]:
    """
    Retorna lista de (query, tool, args) cubriendo todas las categorías pedidas.
    Cada categoría tiene al menos N ejemplos para cobertura.
    """
    queries: list[tuple[str, str, dict]] = []

    # 1) Plugins: listar/activar/desactivar
    queries.append(("Lista los plugins activos del runtime", "list_plugins", {}))
    for pl in plugins[:6]:
        queries.append((f"Activa el plugin {pl}", "enable_plugin", {"name": pl}))
        queries.append((f"Desactiva el plugin {pl}", "disable_plugin", {"name": pl}))

    # 2) Eventos entre voces (request_generation, send_event)
    for v in voices:
        queries.append((f"Jane, pedile a {v} que genere el plan del sprint", "request_generation", {"target_voice": v, "prompt": "genera el plan del sprint con fases y deliverables"}))
    # send_event genérico
    for ev in events[:4]:
        queries.append((f"Dispara el evento {ev} con datos de prueba", "send_event", {"event_name": ev, "data": {"test": True}}))
    queries.append(("¿Qué eventos hay disponibles en el bus?", "list_events", {}))

    # 3) Skills
    for sk in skills[:8]:
        queries.append((f"Leé la skill {sk}", "read_file", {"path": f"skills/{sk}/SKILL.md"}))
        queries.append((f"Crea una skill {sk}-v2 con contenido mejorado", "create_skill", {"name": f"{sk}-v2", "content": f"# {sk}-v2\nInstrucciones mejoradas para {sk}"}))
    # update_skill
    if skills:
        queries.append((f"Actualiza la skill {skills[0]} con nuevas instrucciones", "update_skill", {"name": skills[0], "content": "# Updated\nNuevas reglas"}))

    # 4) MCPs: Kateto NO expone mcp_call como voice tool (el acceso MCP va por
    # ExternalMcpManager interno). Se cubre con tools reales del executor.
    queries.extend([
        ("Borra el archivo temporal docs/borrador.md", "delete_file", {"path": "docs/borrador.md"}),
        ("Guarda en memoria que el sprint usa TDD", "refine_memory", {"content": "sprint usa TDD"}),
        ("Leé la skill backlog de Kateto", "read_file", {"path": "config/defaults/skills/backlog/SKILL.md"}),
        ("Leé la skill orchestrator de Kateto", "read_file", {"path": "config/defaults/skills/orchestrator/SKILL.md"}),
    ])

    # 5) Workflows crear/avanzar/terminar
    for v in voices:
        queries.append((f"Crea el workflow review para {v} con 3 fases", "create_workflow", {"name": "review", "voice": v, "content": f"workflow review para {v}"}))
        queries.append((f"Avanzá el workflow review de {v} a la siguiente fase", "update_workflow", {"name": "review", "voice": v, "content": "avance fase 2"}))
    queries.append(("Termina el workflow review marcando todas las fases como done", "update_workflow", {"name": "review", "voice": "doktor", "content": "completed"}))

    # 6) Extras: files, command, schedule, memory
    queries.extend([
        ("Lee el archivo docs/plan.md", "read_file", {"path": "docs/plan.md"}),
        ("Escribe el reporte en docs/reporte.md", "write_file", {"path": "docs/reporte.md", "content": "# Reporte\nContenido"}),
        ("Corre los tests del event bus", "run_command", {"command": "uv run pytest kateto/tests/test_event_bus.py -v"}),
        ("Programa un recordatorio en 30 segundos", "schedule_event", {"event_name": "Reminder", "delay": 30}),
        ("Qué hora es ahora?", "get_current_time", {}),
        ("Actualiza la soul de jane con nuevo rol", "update_soul", {"name": "jane", "content": "# SOUL\nNueva voz"}),
    ])

    return queries


# Formato dual helpers (inspirado en toolcalling.py)

def to_rwkv_answer(tool: str, args: dict, answer: str) -> str:
    """RWKV G1: <think> + <tool_call> + tool_response -> final"""
    call = json.dumps({"name": tool, "arguments": args}, ensure_ascii=False)
    tool_resp = json.dumps({"ok": True, "tool": tool}, ensure_ascii=False)
    # Formato que to_sft espera: answer es string con think/tool_call embebidos
    # Incluimos think, tool_call y simulamos tool_response como parte del answer SFT
    return f"<think>\nNecesito llamar a {tool} para resolver el pedido. Reviso parámetros.\n</think>\n<tool_call>\n{call}\n</tool_call>\n<tool_response>\n{tool_resp}\n</tool_response>\n{answer}"


def to_openai_messages(query: str, tool: str, args: dict, answer: str) -> list[dict]:
    call = json.dumps({"name": tool, "arguments": args}, ensure_ascii=False)
    return [
        {"role": "system", "content": "You are Kateto, an event-driven voice team assistant. Respond ONLY with a valid JSON tool call when a tool is needed."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": call, "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": json.dumps({"ok": True}, ensure_ascii=False)},
        {"role": "assistant", "content": answer},
    ]


async def synthesize_one(query: str, tool: str, args: dict, model: str, base_url: str, timeout: float, api_key: str | None = None) -> str | None:
    """Llama al LLM para generar answer rioplatense breve; retorna answer o None"""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return None
    try:
        import sys as _sys
        _sys.path.insert(0, str(Path.home() / "proyectos" / "OpenaiBuildWeek" / "Kateto"))
        from kateto.tools.dataset_filter.generator import resolve_api_key
        key = resolve_api_key(api_key)
    except Exception:
        key = api_key or os.getenv("OPENAI_API_KEY", "sk-no-key")
    client = AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=1)
    prompt = (
        f"User query: {query}\nTool to call: {tool}({json.dumps(args, ensure_ascii=False)})\n"
        "Escribí UNA sola línea de chat en español rioplatense seco, la frase que le dirías al usuario "
        "para confirmar que la acción se hizo. PROHIBIDO devolver JSON, código, llaves o el tool call: "
        "solo la frase, sin fluff. No inventes datos concretos (horas, nombres, números). "
        "Variá tus aperturas, no repitas fórmulas."
    )
    try:
        resp = await client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=120)
        txt = (resp.choices[0].message.content or "").strip()
        return txt[:300] if txt else None
    except Exception as e:
        print(f"[warn] synthesize {tool} falló: {e}", file=sys.stderr)
        return None


def build_parser():
    p = argparse.ArgumentParser(description="Genera dataset toolcalling dual OpenAI/RWKV con descubrimiento real")
    p.add_argument("--output", default="data/toolcalling_sft.jsonl")
    p.add_argument("--limit", type=int, default=200, help="máx pares")
    p.add_argument("--dry-run", action="store_true", help="no llama a Orion, usa templates determinísticos")
    p.add_argument("--model", default=None, help="default auto (freellmapi)")
    p.add_argument("--base-url", default=None, help="default http://127.0.0.1:3001/v1 (freellmapi)")
    p.add_argument("--api-key", default=None, help="default env OPENAI_API_KEY/FREELLMAPI_KEY u opencode.json")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--concurrency", type=int, default=4, help="4 para freellmapi; 1 para Orion MoE")
    p.add_argument("--seed", type=int, default=42)
    return p


async def main_async(args):
    random.seed(args.seed)
    skills = discover_skills()
    plugins = discover_plugins()
    voices = discover_voices()
    events = discover_events()
    builtin = discover_builtin_tools()

    print(f"[info] skills: {len(skills)} ({skills[:5]})", file=sys.stderr)
    print(f"[info] plugins: {plugins}", file=sys.stderr)
    print(f"[info] events: {events[:6]}", file=sys.stderr)
    print(f"[info] builtin_tools: {[t['name'] for t in builtin][:8]}", file=sys.stderr)

    queries = build_toolcalling_queries(skills, plugins, voices, events)
    # Deduplicar por query
    seen = set()
    uniq = []
    for q in queries:
        if q[0] not in seen:
            seen.add(q[0])
            uniq.append(q)
    queries = uniq
    # Estratificado: garantizar cobertura de cada categoría incluso con limit chico
    # En vez de shuffle ciego, hacemos round-robin por tool
    from collections import defaultdict
    by_tool: dict[str, list] = defaultdict(list)
    for q in queries:
        by_tool[q[1]].append(q)
    # Intercalar round-robin
    interleaved = []
    max_len = max(len(v) for v in by_tool.values()) if by_tool else 0
    for i in range(max_len):
        for tool in sorted(by_tool.keys()):
            if i < len(by_tool[tool]):
                interleaved.append(by_tool[tool][i])
    # Repetir sampleando hasta limit con round-robin
    expanded = []
    idx = 0
    while len(expanded) < args.limit:
        expanded.append(interleaved[idx % len(interleaved)])
        idx += 1
        if idx > args.limit * 3 and len(interleaved) == 0:
            break
    queries = expanded[:args.limit]
    # Shuffle suave preservando estratificación si limit grande
    if args.limit > len(interleaved) * 2:
        random.shuffle(queries)

    model = args.model or os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or "auto"
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:3001/v1"
    if "orion" in model.lower() and args.concurrency != 1:
        print(f"[warn] Orion MoE: forzando concurrency=1", file=sys.stderr)
        args.concurrency = 1

    # probe gateway si no dry-run
    if not args.dry_run:
        try:
            import httpx
            r = httpx.get("http://127.0.0.1:11434/v1/models", timeout=5)
            print(f"[info] gateway probe {r.status_code}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] gateway no responde: {e} — usa --dry-run si slot ocupado", file=sys.stderr)

    # Templates fallback rioplatense seco
    templates = {
        "list_plugins": "Ahí tenés los plugins activos, todos en verde.",
        "enable_plugin": "Listo, plugin activado. Ya está corriendo.",
        "disable_plugin": "Hecho, lo desactivé sin drama.",
        "list_events": "Esos son los eventos disponibles en el bus.",
        "send_event": "Evento disparado, las voces lo van a recibir.",
        "request_generation": "Ya le pedí a la voz que genere eso, en breve responde.",
        "create_skill": "Skill creada, quedó en skills/ lista para usar.",
        "update_skill": "Skill actualizada, ya tomó los cambios.",
        "read_file": "Acá está el contenido del archivo.",
        "write_file": "Archivo escrito, quedó guardado.",
        "mcp_call": "Llamé al MCP y volvió ok.",
        "create_workflow": "Workflow creado para esa voz, fases listas.",
        "update_workflow": "Workflow avanzado a la siguiente fase.",
        "run_command": "Comando ejecutado, salió bien.",
        "schedule_event": "Evento programado, se dispara a tiempo.",
        "get_current_time": "Acá tenés la hora actual.",
        "_default": "Dale, lo ejecuto y te aviso cómo sale.",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    results = []

    async def gen_one(idx, query, tool, targs):
        async with sem:
            if args.dry_run:
                answer = templates.get(tool, templates["_default"])
            else:
                ans = await synthesize_one(query, tool, targs, model, base_url, args.timeout, args.api_key)
                # el modelo a veces devuelve JSON en vez de la frase: eso va al tool_call, no al cierre
                if ans and ans.strip().startswith("{"):
                    ans = None
                answer = ans if ans else templates.get(tool, templates["_default"])
            rwkv_ans = to_rwkv_answer(tool, targs, answer)
            openai_msgs = to_openai_messages(query, tool, targs, answer)
            return {
                "question": query,
                "answer": rwkv_ans,
                "source_label": "toolcalling",
                "tool": tool,
                "arguments": targs,
                "openai": {"messages": openai_msgs},
                "rwkv": {"think": f"Hay que llamar a {tool}", "tool_call": {"name": tool, "arguments": targs}, "tool_response": {"ok": True}, "final": answer},
            }

    tasks = [gen_one(i, q, t, a) for i, (q, t, a) in enumerate(queries)]
    if args.concurrency == 1:
        # secuencial estricto para Orion MoE
        for coro in tasks:
            r = await coro
            results.append(r)
    else:
        results = await asyncio.gather(*tasks)

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Verificación to_sft compatible
    print(f"[info] escritos {len(results)} pares -> {out_path}", file=sys.stderr)
    # Resumen cobertura
    from collections import Counter
    cnt = Counter(r["tool"] for r in results)
    print(f"[info] cobertura tools: {dict(cnt)}", file=sys.stderr)
    cats = {
        "plugins": sum(1 for r in results if r["tool"] in ("list_plugins", "enable_plugin", "disable_plugin")),
        "eventos": sum(1 for r in results if r["tool"] in ("send_event", "list_events", "request_generation")),
        "skills": sum(1 for r in results if r["tool"] in ("create_skill", "update_skill", "read_file")),
        "mcps": sum(1 for r in results if r["tool"] == "mcp_call"),
        "workflows": sum(1 for r in results if r["tool"] in ("create_workflow", "update_workflow")),
    }
    print(f"[info] cobertura categorías: {cats}", file=sys.stderr)
    if not args.dry_run and len(results) == 0:
        print("AVISO: Orion no respondió (slot ocupado) — usar --dry-run verificado", file=sys.stderr)
    return results


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    results = asyncio.run(main_async(args))
    print(f"Done. pairs={len(results)} -> {args.output} dry_run={args.dry_run}")


if __name__ == "__main__":
    main()
