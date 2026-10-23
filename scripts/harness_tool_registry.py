"""Harness tool registry extractor for Kateto.

Extracts BUILTIN_TOOLS specifications directly from tools.py without importing
the harness or any external dependencies.
"""

from __future__ import annotations

import ast
import os
import re
import sys


def load_registry(tools_py_path: str) -> dict[str, dict]:
    """Parse tools.py and return dictionary mapping tool names to specifications."""
    if not os.path.exists(tools_py_path):
        raise SystemExit(f"Tools file not found: {tools_py_path}")

    try:
        with open(tools_py_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as exc:
        raise SystemExit(f"Error reading tools file {tools_py_path}: {exc}")

    marker = "BUILTIN_TOOLS: tuple"
    start_pos = content.find(marker)
    if start_pos == -1:
        raise SystemExit(f"Marker '{marker}' not found in {tools_py_path}")

    # Find the closing '\n)' for the BUILTIN_TOOLS tuple
    end_pos = content.find("\n)", start_pos)
    if end_pos == -1:
        raise SystemExit(f"Closing tuple delimiter '\\n)' not found after marker in {tools_py_path}")

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

        # Extract required parameters list
        req_m = re.search(r'"required":\s*\[(.*?)\]', chunk, re.DOTALL)
        if req_m:
            required = [p.strip() for p in re.findall(r'"([^"]+)"', req_m.group(1))]
        else:
            required = []

        # Extract properties dict
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

    if len(registry) < 18:
        raise SystemExit(f"Expected at least 18 tools, found {len(registry)}")

    return registry


if __name__ == "__main__":
    path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/chaos/proyectos/OpenaiBuildWeek/Kateto/kateto/voices/tools.py"
    )
    reg = load_registry(path)
    print(f"Loaded {len(reg)} tools successfully:")
    for name, data in reg.items():
        print(f"  - {name}: required={data['required']}, props={data['props']}")
