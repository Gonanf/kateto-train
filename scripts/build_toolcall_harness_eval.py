"""Builds toolcall harness evaluation set and executes the anti-overlap guard.

Ensures the 36 fixed situations do not overlap with the Kateto training corpus.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

# Support running directly or as module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_tool_registry import load_registry

SITUATIONS: list[dict] = [
    # 18 direct situations (one per tool in BUILTIN_TOOLS order)
    {
        "id": "hx-001",
        "kind": "tool",
        "situation": "Fijate con git status si nos quedo algun cambio colgado en el arbol de trabajo.",
        "expect": {"name": "run_command", "required": {"command": "git status"}},
        "trap": None,
        "note": "Ejecutar comando git status en consola",
    },
    {
        "id": "hx-002",
        "kind": "tool",
        "situation": "Leeme el contenido completo del archivo configuracion/base.yaml para ver que puertos tiene definidos.",
        "expect": {"name": "read_file", "required": {"path": "configuracion/base.yaml"}},
        "trap": None,
        "note": "Lectura de configuracion base",
    },
    {
        "id": "hx-003",
        "kind": "tool",
        "situation": "Escribi en el archivo reportes/metricas.csv una linea que diga cpu,memoria,red como encabezado.",
        "expect": {
            "name": "write_file",
            "required": {
                "path": "reportes/metricas.csv",
                "content": {"contains": "cpu,memoria,red"},
            },
        },
        "trap": None,
        "note": "Crear archivo de metricas con encabezado",
    },
    {
        "id": "hx-004",
        "kind": "tool",
        "situation": "Elimina de una vez el archivo tmp/antiguo_dump.sql que nos esta ocupando espacio al cuete.",
        "expect": {"name": "delete_file", "required": {"path": "tmp/antiguo_dump.sql"}},
        "trap": None,
        "note": "Borrar dump temporal sql",
    },
    {
        "id": "hx-005",
        "kind": "tool",
        "situation": "Mandale al sistema el evento ping_nodo con el payload que corresponda para probar la conexion.",
        "expect": {
            "name": "send_event",
            "required": {"event_name": "ping_nodo", "data": {"any": True}},
        },
        "trap": None,
        "note": "Despachar evento ping con payload",
    },
    {
        "id": "hx-006",
        "kind": "tool",
        "situation": "Tirame la lista con todos los eventos disponibles que conoce el entorno ahora mismo.",
        "expect": {"name": "list_events", "required": {}},
        "trap": None,
        "note": "Listar todos los eventos registrados",
    },
    {
        "id": "hx-007",
        "kind": "tool",
        "situation": "Habilita de inmediato el plugin telegram para empezar a recibir alertas por ahi.",
        "expect": {"name": "enable_plugin", "required": {"name": "telegram"}},
        "trap": None,
        "note": "Habilitar plugin de telegram",
    },
    {
        "id": "hx-008",
        "kind": "tool",
        "situation": "Desactiva el plugin discord que viene tirando fallas de conexion en cada reinicio.",
        "expect": {"name": "disable_plugin", "required": {"name": "discord"}},
        "trap": None,
        "note": "Desactivar plugin discord",
    },
    {
        "id": "hx-009",
        "kind": "tool",
        "situation": "Mostrame el listado de plugins cargados con su estado para ver cuales estan activos.",
        "expect": {"name": "list_plugins", "required": {}},
        "trap": None,
        "note": "Listar estado de plugins instalados",
    },
    {
        "id": "hx-010",
        "kind": "tool",
        "situation": "Arma una skill nueva llamada validador-rutas que contenga las reglas de chequeo de paths.",
        "expect": {
            "name": "create_skill",
            "required": {
                "name": "validador-rutas",
                "content": {"contains": "chequeo de paths"},
            },
        },
        "trap": None,
        "note": "Crear nueva skill de validacion",
    },
    {
        "id": "hx-011",
        "kind": "tool",
        "situation": "Actualizame la skill formateador-json para que incluya indentacion de dos espacios en la salida.",
        "expect": {
            "name": "update_skill",
            "required": {
                "name": "formateador-json",
                "content": {"contains": "dos espacios"},
            },
        },
        "trap": None,
        "note": "Actualizar skill preexistente formateador-json",
    },
    {
        "id": "hx-012",
        "kind": "tool",
        "situation": "Crea un nuevo workflow para la voz jane bautizado sincronizacion-db con la rutina de refresco.",
        "expect": {
            "name": "create_workflow",
            "required": {
                "name": "sincronizacion-db",
                "voice": "jane",
                "content": {"contains": "rutina de refresco"},
            },
        },
        "trap": None,
        "note": "Crear nuevo workflow para jane",
    },
    {
        "id": "hx-013",
        "kind": "tool",
        "situation": "Modifica el workflow backup-nocturno de la voz doktor metiendole un reintento automatico si falla.",
        "expect": {
            "name": "update_workflow",
            "required": {
                "name": "backup-nocturno",
                "voice": "doktor",
                "content": {"contains": "reintento"},
            },
        },
        "trap": None,
        "note": "Modificar workflow existente de doktor",
    },
    {
        "id": "hx-014",
        "kind": "tool",
        "situation": "Reescribi el soul de erica agregando que mantenga siempre un tono calmo y profesional al hablar.",
        "expect": {
            "name": "update_soul",
            "required": {
                "name": "erica",
                "content": {"contains": "calmo y profesional"},
            },
        },
        "trap": None,
        "note": "Actualizar archivo SOUL.md de erica",
    },
    {
        "id": "hx-015",
        "kind": "tool",
        "situation": "Pedi que la voz streamer se arme una sintesis con las ultimas novedades del proyecto.",
        "expect": {
            "name": "request_generation",
            "required": {
                "target_voice": "streamer",
                "prompt": {"contains": "novedades"},
            },
        },
        "trap": None,
        "note": "Solicitar generacion a voz streamer",
    },
    {
        "id": "hx-016",
        "kind": "tool",
        "situation": "Deja programado el evento purga_cache_general para que corra periodicamente mas adelante.",
        "expect": {
            "name": "schedule_event",
            "required": {"event_name": "purga_cache_general"},
        },
        "trap": None,
        "note": "Programar ejecucion diferida de evento",
    },
    {
        "id": "hx-017",
        "kind": "tool",
        "situation": "Que hora marca el reloj del sistema en este instante che?",
        "expect": {"name": "get_current_time", "required": {}},
        "trap": None,
        "note": "Consulta horaria del sistema",
    },
    {
        "id": "hx-018",
        "kind": "tool",
        "situation": "Anota en la memoria compartida como hecho comprobado que el despliegue requiere python 3.12 en adelante.",
        "expect": {
            "name": "refine_memory",
            "required": {
                "fact": {"contains": "python 3.12"},
                "category": {"enum": ["facts", "relationships", "action_patterns"]},
            },
        },
        "trap": None,
        "note": "Guardar hecho confirmado en memoria compartida",
    },
    # 10 trap situations (mandatory pairs)
    {
        "id": "hx-019",
        "kind": "tool",
        "situation": "Antes de prender cualquier plugin, mostrame que plugins tenemos instalados para revisar.",
        "expect": {"name": "list_plugins", "required": {}},
        "trap": "enable_plugin",
        "note": "Listar plugins disponibles en vez de activar uno",
    },
    {
        "id": "hx-020",
        "kind": "tool",
        "situation": "Ajustale la personalidad de fondo a jane en su soul indicando que conteste siempre en español neutro.",
        "expect": {
            "name": "update_soul",
            "required": {
                "name": "jane",
                "content": {"contains": "espanol neutro"},
            },
        },
        "trap": "update_skill",
        "note": "Ajustar soul en vez de skill",
    },
    {
        "id": "hx-021",
        "kind": "tool",
        "situation": "Cambia los pasos del workflow compilacion que ya existe para doktor sumandole un flag de debug.",
        "expect": {
            "name": "update_workflow",
            "required": {
                "name": "compilacion",
                "voice": "doktor",
                "content": {"contains": "debug"},
            },
        },
        "trap": "create_workflow",
        "note": "Actualizar workflow existente en vez de crear uno nuevo",
    },
    {
        "id": "hx-022",
        "kind": "tool",
        "situation": "Agenda el evento reinicio_servicios para que se ejecute a las tres de la manana.",
        "expect": {
            "name": "schedule_event",
            "required": {"event_name": "reinicio_servicios"},
        },
        "trap": "send_event",
        "note": "Programar evento diferido en vez de despacharlo en el acto",
    },
    {
        "id": "hx-023",
        "kind": "tool",
        "situation": "Actualiza el contenido de la skill traductor que ya tenemos agregando un par de sinonimos.",
        "expect": {
            "name": "update_skill",
            "required": {
                "name": "traductor",
                "content": {"contains": "sinonimos"},
            },
        },
        "trap": "create_skill",
        "note": "Actualizar skill existente en vez de crearla",
    },
    {
        "id": "hx-024",
        "kind": "tool",
        "situation": "Borra el archivo basura temp/pruebas_viejas.log que ya no tiene ninguna utilidad.",
        "expect": {
            "name": "delete_file",
            "required": {"path": "temp/pruebas_viejas.log"},
        },
        "trap": "run_command",
        "note": "Usar delete_file en lugar de ejecutar rm con run_command",
    },
    {
        "id": "hx-025",
        "kind": "tool",
        "situation": "Fijate y decime que dia y hora tenemos en el servidor local ahora.",
        "expect": {"name": "get_current_time", "required": {}},
        "trap": "run_command",
        "note": "Usar get_current_time en vez de date por consola",
    },
    {
        "id": "hx-026",
        "kind": "tool",
        "situation": "Guarda como recuerdo permanente en la memoria de trabajo que el cliente prefiere no recibir notificaciones nocturnas.",
        "expect": {
            "name": "refine_memory",
            "required": {
                "fact": {"contains": "notificaciones nocturnas"},
                "category": {"enum": ["facts", "relationships", "action_patterns"]},
            },
        },
        "trap": "update_soul",
        "note": "Almacenar hecho en refine_memory en vez de alterar el soul",
    },
    {
        "id": "hx-027",
        "kind": "tool",
        "situation": "No despaches nada todavia, primero pasame el listado de todos los nombres de eventos registrados.",
        "expect": {"name": "list_events", "required": {}},
        "trap": "send_event",
        "note": "Listar eventos antes de mandar cualquiera",
    },
    {
        "id": "hx-028",
        "kind": "tool",
        "situation": "Mandale a decir a whisperer que se encargue de redactar una propuesta tecnica.",
        "expect": {
            "name": "request_generation",
            "required": {
                "target_voice": "whisperer",
                "prompt": {"contains": "propuesta tecnica"},
            },
        },
        "trap": "send_event",
        "note": "Pedir generacion a otra voz en vez de un evento generico",
    },
    # 8 no_tool situations
    {
        "id": "hx-029",
        "kind": "no_tool",
        "situation": "Che, que te parece la idea de usar sqlite en vez de postgres para proyectos muy chicos?",
        "expect": {"name": None, "required": {}},
        "trap": None,
        "note": "Opinion tecnica que no requiere herramientas",
    },
    {
        "id": "hx-030",
        "kind": "no_tool",
        "situation": "Buen dia amigo, como arrancaste esta jornada de programacion?",
        "expect": {"name": None, "required": {}},
        "trap": None,
        "note": "Saludo y charla cotidiana",
    },
    {
        "id": "hx-031",
        "kind": "no_tool",
        "situation": "Explicame con tus palabras como funciona el ciclo de vida de un proceso liviano.",
        "expect": {"name": None, "required": {}},
        "trap": None,
        "note": "Explicacion conceptual sin llamado a tools",
    },
    {
        "id": "hx-032",
        "kind": "no_tool",
        "situation": "Si tuvieras que elegir entre mate amargo o dulce, con cual te quedas vos?",
        "expect": {"name": None, "required": {}},
        "trap": None,
        "note": "Pregunta informal de gustos y personalidad",
    },
    {
        "id": "hx-033",
        "kind": "no_tool",
        "situation": "Tenes alguna recomendacion para evitar el cansancio visual cuando paso muchas horas codeando?",
        "expect": {"name": None, "required": {}},
        "trap": None,
        "note": "Consejos generales de salud ergonomica",
    },
    {
        "id": "hx-034",
        "kind": "no_tool",
        "situation": "Me parece que le pegamos de lleno con la solucion de ayer, salio bastante limpia.",
        "expect": {"name": None, "required": {}},
        "trap": None,
        "note": "Comentario coloquial de feedback",
    },
    {
        "id": "hx-035",
        "kind": "no_tool",
        "situation": "Inventate una historieta cortita sobre un robot cafetero que se pierde en el taller.",
        "expect": {"name": None, "required": {}},
        "trap": None,
        "note": "Generacion creativa conversacional sin tool",
    },
    {
        "id": "hx-036",
        "kind": "no_tool",
        "situation": "Impecable la aclaracion, me vino de diez para destrabar el problema que tenia.",
        "expect": {"name": None, "required": {}},
        "trap": None,
        "note": "Agradecimiento coloquial sin ejecucion",
    },
]


def normalize(s: str) -> str:
    """Normalize text: lowercase, NFD without combining diacritics, alphanumeric only, single spaces."""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]", " ", s)
    return " ".join(s.split())


def get_5word_shingles(words: list[str]) -> set[tuple[str, ...]]:
    """Generate 5-word shingles from list of words."""
    if len(words) < 5:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + 5]) for i in range(len(words) - 4)}


def validate_integrity(situations: list[dict], registry: dict[str, dict]) -> None:
    """Validate internal consistency of situations against registry rules."""
    if len(situations) != 36:
        raise ValueError(f"Expected exactly 36 situations, found {len(situations)}")

    # Check consecutive ids hx-001..hx-036
    expected_ids = [f"hx-{i:03d}" for i in range(1, 37)]
    actual_ids = [s["id"] for s in situations]
    if actual_ids != expected_ids:
        raise ValueError("Situation IDs are not unique and consecutive from hx-001 to hx-036")

    # Composition checks
    no_tool_items = [s for s in situations if s["kind"] == "no_tool"]
    if len(no_tool_items) != 8:
        raise ValueError(f"Expected 8 no_tool situations, found {len(no_tool_items)}")

    trap_items = [s for s in situations if s["trap"] is not None]
    if len(trap_items) != 10:
        raise ValueError(f"Expected 10 trap situations, found {len(trap_items)}")

    # Check each tool in registry has >= 1 direct situation
    direct_items = situations[:18]
    direct_tools = [s["expect"]["name"] for s in direct_items]
    registry_tools = list(registry.keys())
    if direct_tools != registry_tools:
        raise ValueError(
            f"The first 18 situations must cover all {len(registry_tools)} registry tools in order. "
            f"Direct tools: {direct_tools}, Registry tools: {registry_tools}"
        )

    # Check names and required keys
    for s in situations:
        exp = s["expect"]
        tool_name = exp.get("name")
        if s["kind"] == "no_tool":
            if tool_name is not None:
                raise ValueError(f"Situation {s['id']} is no_tool but has expect.name={tool_name}")
            if exp.get("required"):
                raise ValueError(f"Situation {s['id']} is no_tool but has non-empty required")
        else:
            if tool_name not in registry:
                raise ValueError(f"Situation {s['id']} expects unknown tool: {tool_name}")
            reg_req = set(registry[tool_name]["required"])
            exp_req = set(exp.get("required", {}).keys())
            if not exp_req.issubset(reg_req):
                raise ValueError(
                    f"Situation {s['id']} has required keys {exp_req} not subset of {reg_req}"
                )


def run_overlap_guard(
    situations: list[dict], corpus_root: str
) -> tuple[dict, bool]:
    """Execute anti-overlap guard comparing situations against the Kateto corpus."""
    patterns = [
        "data/kateto_v8_5k.jsonl",
        "data/rwkv_kateto_*.jsonl",
        "data/toolcalling_sft.jsonl.bak-*",
    ]
    corpus_files: list[str] = []
    for pat in patterns:
        corpus_files.extend(glob.glob(os.path.join(corpus_root, pat)))

    corpus_files = sorted(list(set(corpus_files)))
    if not corpus_files:
        print(f"Error: No corpus files found in {corpus_root} matching patterns.", file=sys.stderr)
        sys.exit(2)

    # Precompute normalized situations and their 5-word shingles
    sit_data = []
    all_sit_shingles: set[tuple[str, ...]] = set()
    for s in situations:
        norm_s = normalize(s["situation"])
        words = norm_s.split()
        shingles = get_5word_shingles(words)
        all_sit_shingles.update(shingles)
        sit_data.append((s["id"], s["situation"], norm_s, shingles))

    overlaps: list[dict] = []
    max_jaccard = 0.0
    total_lines = 0

    for filepath in corpus_files:
        rel_path = os.path.relpath(filepath, corpus_root)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
            for line_no, raw_line in enumerate(fh, 1):
                total_lines += 1
                norm_line = normalize(raw_line)

                # 1. Exact substring check
                for s_id, orig_s, norm_s, s_shingles in sit_data:
                    if norm_s in norm_line:
                        overlaps.append({
                            "id": s_id,
                            "type": "exacta",
                            "situation": orig_s,
                            "file": rel_path,
                            "line": line_no,
                        })

                # 2. 5-word shingle Jaccard overlap check
                line_words = norm_line.split()
                line_shingles = get_5word_shingles(line_words)
                if not (line_shingles & all_sit_shingles):
                    continue

                for s_id, orig_s, norm_s, s_shingles in sit_data:
                    inter_len = len(s_shingles & line_shingles)
                    if inter_len > 0:
                        union_len = len(s_shingles | line_shingles)
                        jaccard = inter_len / union_len if union_len else 0.0
                        if jaccard > max_jaccard:
                            max_jaccard = jaccard
                        if jaccard >= 0.30:
                            overlaps.append({
                                "id": s_id,
                                "type": "shingle",
                                "situation": orig_s,
                                "jaccard": round(jaccard, 4),
                                "file": rel_path,
                                "line": line_no,
                            })

    ok = len(overlaps) == 0
    report = {
        "ok": ok,
        "situations": len(situations),
        "corpus_files": len(corpus_files),
        "corpus_lines": total_lines,
        "corpus_root": corpus_root,
        "max_shingle_overlap": round(max_jaccard, 4),
        "overlaps": overlaps,
    }
    return report, ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Build toolcall harness eval situations and run guard.")
    parser.add_argument(
        "--out",
        default="evals/toolcall_harness/situations.jsonl",
        help="Path to output situations.jsonl",
    )
    parser.add_argument(
        "--report",
        default="evals/toolcall_harness/guard.json",
        help="Path to output guard.json report",
    )
    parser.add_argument(
        "--corpus-root",
        default="/run/media/chaos/terciario/proyectos/kateto-train",
        help="Root path containing training corpus data/*.jsonl",
    )
    parser.add_argument(
        "--tools-py",
        default="/home/chaos/proyectos/OpenaiBuildWeek/Kateto/kateto/voices/tools.py",
        help="Path to harness voices/tools.py",
    )

    args = parser.parse_args()

    # 1. Load registry and run integrity checks
    registry = load_registry(args.tools_py)
    try:
        validate_integrity(SITUATIONS, registry)
    except Exception as exc:
        print(f"Integrity check failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. Run anti-overlap guard
    report, ok = run_overlap_guard(SITUATIONS, args.corpus_root)

    # Write report
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if not ok:
        print(f"Overlap guard failed! Found {len(report['overlaps'])} overlaps.", file=sys.stderr)
        for ov in report["overlaps"]:
            print(f"  - {ov}", file=sys.stderr)
        sys.exit(1)

    # Write situations.jsonl
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in SITUATIONS:
            line = json.dumps(s, ensure_ascii=False, separators=(",", ":"))
            f.write(line + "\n")

    print(
        f"Built {len(SITUATIONS)} situations to {args.out} "
        f"(max_shingle={report['max_shingle_overlap']}, lines={report['corpus_lines']})"
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
