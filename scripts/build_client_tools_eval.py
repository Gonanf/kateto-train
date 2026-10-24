#!/usr/bin/env python3
"""Build client tools evaluation situations.

Generates 45 situations for the 5 client tools + 3 server distractors,
using the existing harness_tool_registry loader with a temp-file adapter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

# Allow importing registry from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_tool_registry import load_registry


def normalize(s: str) -> str:
    """Normalize text: lowercase, NFD without combining diacritics, alphanumeric only, single spaces."""
    import unicodedata

    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]", " ", s)
    return " ".join(s.split())


def load_client_registry(tools_py_path: str) -> dict[str, dict]:
    """Parse client_tools.py by adapting it for harness_tool_registry.load_registry.

    The loader expects 'BUILTIN_TOOLS: tuple' marker. We create a temp file
    with CLIENT_TOOLS replaced by BUILTIN_TOOLS.
    """
    if not os.path.exists(tools_py_path):
        raise SystemExit(f"Tools file not found: {tools_py_path}")

    with open(tools_py_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace CLIENT_TOOLS: tuple with BUILTIN_TOOLS: tuple for the loader
    # The loader searches for this exact marker
    adapted = content.replace("CLIENT_TOOLS: tuple", "BUILTIN_TOOLS: tuple")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(adapted)
        temp_path = tf.name

    try:
        # loader expects at least 18 tools; we only have 5 client tools + will add 3 distractors = 8
        # So we need to temporarily lower the threshold or patch the loader
        # Simpler: monkey-patch the check in the loaded module
        import harness_tool_registry as htr

        original_check = htr.load_registry

        def patched_load_registry(path: str) -> dict[str, dict]:
            # Temporarily lower the minimum tools check
            import harness_tool_registry as htr_module

            # Read the temp file directly with modified minimum
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            marker = "BUILTIN_TOOLS: tuple"
            start_pos = content.find(marker)
            if start_pos == -1:
                raise SystemExit(f"Marker '{marker}' not found in {path}")

            end_pos = content.find("\n)", start_pos)
            if end_pos == -1:
                raise SystemExit(f"Closing tuple delimiter '\\n)' not found after marker in {path}")

            block = content[start_pos:end_pos]
            chunks = block.split("ChatCompletionToolParam(")

            registry: dict[str, dict] = {}

            for chunk in chunks[1:]:
                name_m = re.search(r'"name":\s*"([^"]+)"', chunk)
                if not name_m:
                    continue
                name = name_m.group(1).strip()

                if name in registry:
                    raise SystemExit(f"Duplicate tool name detected: {name}")

                req_m = re.search(r'"required":\s*\[(.*?)\]', chunk, re.DOTALL)
                if req_m:
                    required = [p.strip() for p in re.findall(r'"([^"]+)"', req_m.group(1))]
                else:
                    required = []

                props: dict[str, str] = {}
                p_idx = chunk.find('"properties":')
                if p_idx != -1:
                    brace_start = chunk.find("{", p_idx)
                    if brace_start != -1:
                        depth = 0
                        brace_end = -1
                        for i in range(brace_start, len(chunk)):
                            if chunk[i] == "{":
                                depth += 1
                            elif chunk[i] == "}":
                                depth -= 1
                                if depth == 0:
                                    brace_end = i
                                    break
                        if brace_end != -1:
                            raw_props_str = chunk[brace_start : brace_end + 1]
                            try:
                                import ast

                                parsed_props = ast.literal_eval(raw_props_str)
                                if isinstance(parsed_props, dict):
                                    for prop_name, prop_def in parsed_props.items():
                                        if isinstance(prop_def, dict):
                                            props[prop_name] = prop_def.get("type", "string")
                                        else:
                                            props[prop_name] = "string"
                            except Exception:
                                prop_matches = re.finditer(
                                    r'"([^"]+)":\s*\{([^}]*)\}', raw_props_str
                                )
                                for pm in prop_matches:
                                    p_name = pm.group(1)
                                    body = pm.group(2)
                                    t_match = re.search(r'"type":\s*"([^"]+)"', body)
                                    props[p_name] = t_match.group(1) if t_match else "string"

                registry[name] = {
                    "name": name,
                    "required": required,
                    "props": props,
                }

            return registry

        return patched_load_registry(temp_path)
    finally:
        os.unlink(temp_path)


def add_server_distractors(registry: dict[str, dict]) -> dict[str, dict]:
    """Add 3 server tools as distractors with same schemas as harness."""
    server_tools = {
        "run_command": {
            "name": "run_command",
            "required": ["command"],
            "props": {"command": "string"},
        },
        "read_file": {
            "name": "read_file",
            "required": ["path"],
            "props": {"path": "string"},
        },
        "get_current_time": {
            "name": "get_current_time",
            "required": [],
            "props": {},
        },
    }
    for name, spec in server_tools.items():
        registry[name] = spec
    return registry


def generate_situations(registry: dict[str, dict]) -> list[dict]:
    """Generate exactly 45 situations with the required distribution."""
    situations: list[dict] = []
    sid = 1

    def make_id() -> str:
        nonlocal sid
        i = f"ct-{sid:03d}"
        sid += 1
        return i

    # client_flashlight: 12 (mix on/off, explicit/colloquial)
    flashlight_situations = [
        ("prendeme la linterna", {"on": True}),
        ("apagá la linterna", {"on": False}),
        ("encendé la linterna porfa", {"on": True}),
        ("cortá la linterna", {"on": False}),
        ("necesito luz, prendé la linterna", {"on": True}),
        ("ya no hace falta la linterna, apagala", {"on": False}),
        ("prendeme la linterna que no veo nada", {"on": True}),
        ("apagá la linterna que me deslumbra", {"on": False}),
        ("linterna on", {"on": True}),
        ("linterna off", {"on": False}),
        ("activá la linterna del celu", {"on": True}),
        ("desactivá la linterna", {"on": False}),
    ]
    for situation, required in flashlight_situations:
        situations.append({
            "id": make_id(),
            "kind": "tool",
            "situation": situation,
            "expect": {"name": "client_flashlight", "required": required},
            "trap": None,
            "note": "Control de linterna del cliente",
        })

    # client_location: 8
    location_situations = [
        "¿dónde estoy?",
        "decime mi ubicación actual",
        "pasame las coordenadas de donde estoy",
        "ubicame en el mapa",
        "en qué lugar estoy parado",
        "dame mi ubicación",
        "quiero saber dónde estoy",
        "localizame",
    ]
    for situation in location_situations:
        situations.append({
            "id": make_id(),
            "kind": "tool",
            "situation": situation,
            "expect": {"name": "client_location", "required": {}},
            "trap": None,
            "note": "Obtener ubicación del cliente",
        })

    # client_battery: 6
    battery_situations = [
        "¿cuánta batería me queda?",
        "decime el nivel de batería",
        "estado de la batería del celu",
        "batería",
        "me está muriendo el celu, ¿cuánta carga tiene?",
        "chequeá la batería",
    ]
    for situation in battery_situations:
        situations.append({
            "id": make_id(),
            "kind": "tool",
            "situation": situation,
            "expect": {"name": "client_battery", "required": {}},
            "trap": None,
            "note": "Obtener estado de batería del cliente",
        })

    # client_network: 6
    network_situations = [
        "¿tengo señal?",
        "estado de la red",
        "cómo viene la conexión",
        "red móvil",
        "verificá la conectividad",
        "¿estoy conectado?",
    ]
    for situation in network_situations:
        situations.append({
            "id": make_id(),
            "kind": "tool",
            "situation": situation,
            "expect": {"name": "client_network", "required": {}},
            "trap": None,
            "note": "Obtener estado de red del cliente",
        })

    # client_bluetooth: 6
    bluetooth_situations = [
        "¿está prendido el bluetooth?",
        "estado del bluetooth",
        "decime si el bluetooth está activo",
        "bluetooth",
        "verificá el bluetooth",
        "qué dispositivos tengo conectados por bluetooth",
    ]
    for situation in bluetooth_situations:
        situations.append({
            "id": make_id(),
            "kind": "tool",
            "situation": situation,
            "expect": {"name": "client_bluetooth", "required": {}},
            "trap": None,
            "note": "Obtener estado de bluetooth del cliente",
        })

    # Server distractors: 4 (run_command, read_file, get_current_time)
    distractor_situations = [
        ("ejecutá git status en la consola", "run_command", {"command": "git status"}),
        ("leeme el archivo config.yaml", "read_file", {"path": "config.yaml"}),
        ("qué hora es", "get_current_time", {}),
        ("corré ls -la en el directorio actual", "run_command", {"command": "ls -la"}),
    ]
    for situation, tool_name, required in distractor_situations:
        situations.append({
            "id": make_id(),
            "kind": "tool",
            "situation": situation,
            "expect": {"name": tool_name, "required": required},
            "trap": None,
            "note": f"Distractor servidor: {tool_name}",
        })

    # no_tool: 3
    no_tool_situations = [
        "che, ¿qué tal todo?",
        "contame un chiste corto",
        "qué opinás de la lluvia de hoy",
    ]
    for situation in no_tool_situations:
        situations.append({
            "id": make_id(),
            "kind": "no_tool",
            "situation": situation,
            "expect": {"name": None, "required": {}},
            "trap": None,
            "note": "Charla sin herramienta",
        })

    # Verify count
    assert len(situations) == 45, f"Expected 45 situations, got {len(situations)}"

    # Verify distribution
    by_tool: dict[str, int] = {}
    for s in situations:
        if s["kind"] == "tool":
            name = s["expect"]["name"]
            by_tool[name] = by_tool.get(name, 0) + 1
        else:
            by_tool["no_tool"] = by_tool.get("no_tool", 0) + 1

    expected_dist = {
        "client_flashlight": 12,
        "client_location": 8,
        "client_battery": 6,
        "client_network": 6,
        "client_bluetooth": 6,
        "run_command": 2,
        "read_file": 1,
        "get_current_time": 1,
        "no_tool": 3,
    }
    for tool, expected in expected_dist.items():
        actual = by_tool.get(tool, 0)
        if actual != expected:
            raise SystemExit(f"Distribution mismatch for {tool}: expected {expected}, got {actual}")

    return situations


def validate_situations(
    situations: list[dict], registry: dict[str, dict], existing_situations_path: str
) -> tuple[list[str], list[str]]:
    """Validate situations against requirements.

    Returns (duplicates, collisions_with_existing).
    """
    # 1. Each expect.name exists in registry (for tool situations)
    for s in situations:
        if s["kind"] == "tool":
            name = s["expect"]["name"]
            if name not in registry:
                raise SystemExit(f"Validation failed: expect.name '{name}' not in registry")

    # 2. 45 situations are unique (case/accent-insensitive) and none match existing
    existing_situations: set[str] = set()
    if os.path.exists(existing_situations_path):
        with open(existing_situations_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        existing_situations.add(normalize(json.loads(line)["situation"]))
                    except Exception:
                        pass

    seen: dict[str, str] = {}  # normalized -> original id
    duplicates: list[str] = []
    collisions: list[str] = []

    for s in situations:
        norm = normalize(s["situation"])
        if norm in seen:
            duplicates.append(f"{s['id']} duplicates {seen[norm]}: {s['situation']}")
        elif norm in existing_situations:
            collisions.append(f"{s['id']} collides with existing: {s['situation']}")
        else:
            seen[norm] = s["id"]

    # 3. Distribution already checked in generate_situations

    # 4. Each tool situation has at least one required non-empty OR tool has no args in registry
    for s in situations:
        if s["kind"] == "tool":
            tool_name = s["expect"]["name"]
            registry_required = registry.get(tool_name, {}).get("required", [])
            required = s["expect"].get("required", {})
            if registry_required and not required:
                raise SystemExit(f"Validation failed: {s['id']} has empty required but tool expects args")

    return duplicates, collisions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build client tools evaluation situations.")
    parser.add_argument("--tools-py", required=True, help="Path to client_tools.py")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--report", required=True, help="Report JSON output path")
    parser.add_argument("--n", type=int, default=45, help="Number of situations (must be 45)")
    args = parser.parse_args()

    if args.n != 45:
        raise SystemExit("--n must be 45 (fixed distribution)")

    # Load client registry
    registry = load_client_registry(args.tools_py)

    # Add server distractors
    registry = add_server_distractors(registry)

    # Generate situations
    situations = generate_situations(registry)

    # Validate
    existing_path = "evals/toolcall_harness/situations.jsonl"
    duplicates, collisions = validate_situations(situations, registry, existing_path)

    if duplicates:
        raise SystemExit(f"Duplicate situations found: {duplicates}")
    if collisions:
        raise SystemExit(f"Collisions with existing harness situations: {collisions}")

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in situations:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Write report
    by_tool: dict[str, int] = {}
    for s in situations:
        if s["kind"] == "tool":
            name = s["expect"]["name"]
            by_tool[name] = by_tool.get(name, 0) + 1
        else:
            by_tool["no_tool"] = by_tool.get("no_tool", 0) + 1

    report = {
        "n": len(situations),
        "by_tool": by_tool,
        "duplicados": duplicates,
        "colisiones_con_toolcall_harness": collisions,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(situations)} situations -> {args.out}")
    print(f"Report -> {args.report}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()