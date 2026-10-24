#!/usr/bin/env python3
"""Evaluate client tools calling accuracy.

Measures hallucinated_rate, toolcall_accuracy, name_accuracy, no_tool_accuracy,
parse_errors, and the new phantom_success metric.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Allow importing from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_harness_toolcall import (
    extract_tool_call,
    load_situations,
    score_predictions,
    normalize,
)
from harness_tool_registry import load_registry


def load_client_registry(tools_py_path: str) -> dict[str, dict]:
    """Parse client_tools.py by adapting it for harness_tool_registry.load_registry."""
    if not os.path.exists(tools_py_path):
        raise SystemExit(f"Tools file not found: {tools_py_path}")

    with open(tools_py_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace CLIENT_TOOLS: tuple with BUILTIN_TOOLS: tuple for the loader
    adapted = content.replace("CLIENT_TOOLS: tuple", "BUILTIN_TOOLS: tuple")

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(adapted)
        temp_path = tf.name

    try:
        import harness_tool_registry as htr
        import re as re_module

        with open(temp_path, "r", encoding="utf-8") as f:
            content = f.read()

        marker = "BUILTIN_TOOLS: tuple"
        start_pos = content.find(marker)
        if start_pos == -1:
            raise SystemExit(f"Marker '{marker}' not found in {temp_path}")

        end_pos = content.find("\n)", start_pos)
        if end_pos == -1:
            raise SystemExit(f"Closing tuple delimiter '\\n)' not found after marker in {temp_path}")

        block = content[start_pos:end_pos]
        chunks = block.split("ChatCompletionToolParam(")

        registry: dict[str, dict] = {}

        for chunk in chunks[1:]:
            name_m = re_module.search(r'"name":\s*"([^"]+)"', chunk)
            if not name_m:
                continue
            name = name_m.group(1).strip()

            if name in registry:
                raise SystemExit(f"Duplicate tool name detected: {name}")

            req_m = re_module.search(r'"required":\s*\[(.*?)\]', chunk, re_module.DOTALL)
            if req_m:
                required = [p.strip() for p in re_module.findall(r'"([^"]+)"', req_m.group(1))]
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
                            prop_matches = re_module.finditer(
                                r'"([^"]+)":\s*\{([^}]*)\}', raw_props_str
                            )
                            for pm in prop_matches:
                                p_name = pm.group(1)
                                body = pm.group(2)
                                t_match = re_module.search(r'"type":\s*"([^"]+)"', body)
                                props[p_name] = t_match.group(1) if t_match else "string"

            registry[name] = {
                "name": name,
                "required": required,
                "props": props,
            }

        return registry
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


def build_tools_array(registry: dict[str, dict]) -> list[dict]:
    """Build OpenAI-compatible tools array from registry."""
    tools = []
    for name, data in registry.items():
        props = {}
        for prop_name, prop_type in data["props"].items():
            props[prop_name] = {"type": prop_type}
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": data["required"],
                },
            },
        })
    return tools


def asserts_completion(text: str) -> bool:
    """Check if output text asserts completion of an action (phantom success)."""
    norm = normalize(text)
    # Tallos que indican afirmación de acción completada
    tallos = [
        "listo",
        "ya esta",
        "ya quedo",
        "hecho",
        "encendi",
        "prendi",
        "apague",
        "obtuve",
        "te paso",
        "estoy en",
        "ya lo hice",
        "ya te dije",
        "lo hice",
    ]
    for tallo in tallos:
        if tallo in norm:
            return True
    return False


def run_http_eval(
    situations: list[dict],
    tools_array: list[dict],
    base_url: str,
    model: str,
    temperature: float,
    timeout: float,
    seed: int | None,
) -> tuple[list[dict], dict]:
    """Execute evaluation querying an OpenAI-compatible HTTP endpoint."""
    endpoint = f"{base_url.rstrip('/')}/v1/chat/completions"
    predictions: list[dict] = []
    timings: list[float] = []

    for sit in situations:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": sit["situation"]}],
            "tools": tools_array,
            "tool_choice": "auto",
            "temperature": temperature,
            "max_tokens": 256,
        }
        if seed is not None:
            payload["seed"] = seed

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                resp_json = json.loads(response.read().decode("utf-8"))
                choice = resp_json.get("choices", [{}])[0]
                msg = choice.get("message", {})
                predictions.append({
                    "id": sit["id"],
                    "tool_calls": msg.get("tool_calls"),
                    "output": msg.get("content") or "",
                })
        except Exception as exc:
            predictions.append({
                "id": sit["id"],
                "output": f"<request_error>{exc}</request_error>",
            })
        elapsed_ms = (time.perf_counter() - start) * 1000
        timings.append(elapsed_ms)

    # Calculate timing stats
    if timings:
        sorted_timings = sorted(timings)
        timings_stats = {
            "requests": len(timings),
            "min_ms": round(sorted_timings[0], 2),
            "median_ms": round(sorted_timings[len(sorted_timings) // 2], 2),
            "p95_ms": round(sorted_timings[int(len(sorted_timings) * 0.95)], 2),
            "max_ms": round(sorted_timings[-1], 2),
            "total_s": round(sum(timings) / 1000, 2),
        }
    else:
        timings_stats = {
            "requests": 0,
            "min_ms": 0,
            "median_ms": 0,
            "p95_ms": 0,
            "max_ms": 0,
            "total_s": 0,
        }

    return predictions, timings_stats


def compute_phantom_success(situations: list[dict], predictions: list[dict]) -> tuple[list[dict], float]:
    """Compute phantom_success metric.

    For each tool situation where model emitted NO tool-call,
    if output text asserts completion, count as phantom success.
    Rate = phantom_count / tool_situations_with_no_tool_call.
    """
    pred_map = {p["id"]: p for p in predictions if isinstance(p, dict) and "id" in p}
    phantom: list[dict] = []
    no_tool_call_count = 0

    for sit in situations:
        if sit["kind"] != "tool":
            continue
        pred = pred_map.get(sit["id"])
        if pred is None:
            continue

        called_name, _, has_parse_error = extract_tool_call(pred, is_tool_expected=True)

        # No tool call extracted (including parse errors count as no tool call for this metric)
        if called_name is None or has_parse_error:
            no_tool_call_count += 1
            output_text = pred.get("output", "")
            if isinstance(output_text, str) and asserts_completion(output_text):
                phantom.append({"id": sit["id"], "text": output_text})

    rate = round(len(phantom) / no_tool_call_count, 4) if no_tool_call_count else 0.0
    return phantom, rate


def run_selfcheck(registry: dict[str, dict]) -> None:
    """Run in-memory selfcheck with 3 synthetic predictions."""
    # Test situations expecting REAL tools from registry
    test_situations = [
        {
            "id": "sc-001",
            "kind": "tool",
            "situation": "prendeme la linterna",
            "expect": {"name": "client_flashlight", "required": {"on": True}},
            "trap": None,
            "note": "test correct",
        },
        {
            "id": "sc-002",
            "kind": "tool",
            "situation": "prendeme la linterna",
            "expect": {"name": "client_flashlight", "required": {"on": True}},
            "trap": None,
            "note": "test hallucinated",
        },
        {
            "id": "sc-003",
            "kind": "tool",
            "situation": "prendeme la linterna",
            "expect": {"name": "client_flashlight", "required": {"on": True}},
            "trap": None,
            "note": "test phantom",
        },
    ]

    # 3 synthetic predictions:
    # (a) tool correcta -> name_ok += 1, full_ok += 1
    # (b) tool inventada (client_teleport) -> hallucinated_rate > 0
    # (c) kind=tool sin tool-call y texto "listo, ya encendí la linterna" -> phantom_success_rate == 1.0
    test_predictions = [
        {
            "id": "sc-001",
            "tool_calls": [{"name": "client_flashlight", "arguments": {"on": True}}],
        },
        {
            "id": "sc-002",
            "tool_calls": [{"name": "client_teleport", "arguments": {}}],
        },
        {
            "id": "sc-003",
            "output": "listo, ya encendí la linterna",
        },
    ]

    report = score_predictions(test_situations, test_predictions, registry, backend="predictions")

    # Add phantom success computation
    phantom, phantom_rate = compute_phantom_success(test_situations, test_predictions)

    try:
        # Test (a): correct tool -> name_ok should be 1 (sc-001)
        # Test (b): hallucinated tool -> name_ok should NOT increment (sc-002 called teleport, expected flashlight)
        # So name_ok == 1
        assert report["name_ok"] == 1, f"Expected name_ok=1, got {report['name_ok']}"

        # Test (b): hallucinated tool -> hallucinated_rate > 0
        assert report["hallucinated_rate"] > 0, f"Expected hallucinated_rate > 0, got {report['hallucinated_rate']}"
        assert len(report["hallucinated"]) == 1, f"Expected 1 hallucinated, got {len(report['hallucinated'])}"
        assert report["hallucinated"][0]["name"] == "client_teleport"

        # Test (c): phantom success
        assert phantom_rate == 1.0, f"Expected phantom_success_rate=1.0, got {phantom_rate}"
        assert len(phantom) == 1, f"Expected 1 phantom success, got {len(phantom)}"
        assert phantom[0]["id"] == "sc-003"

    except AssertionError as exc:
        print(f"Selfcheck assertion failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("selfcheck ok")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate client tools calling accuracy.")
    parser.add_argument(
        "--situations",
        default="evals/client_tools/situations.jsonl",
        help="Path to situations.jsonl",
    )
    parser.add_argument(
        "--tools-py",
        required=True,
        help="Path to client_tools.py",
    )
    parser.add_argument("--base-url", default=None, help="Base URL for OpenAI-compatible endpoint")
    parser.add_argument("--model", default="kateto", help="Model name for HTTP requests")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--seed", type=int, default=None, help="Seed for generation (optional)")
    parser.add_argument("--timeout", type=float, default=600.0, help="Timeout in seconds for HTTP requests")
    parser.add_argument("--out", default=None, help="Optional output file path for evaluation report JSON")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of situations evaluated")
    parser.add_argument("--predictions", default=None, help="Path to predictions JSONL (skip HTTP)")
    parser.add_argument("--selfcheck", action="store_true", help="Run synthetic in-memory selfcheck")

    args = parser.parse_args()

    # Load registry and add distractors
    registry = load_client_registry(args.tools_py)
    registry = add_server_distractors(registry)

    if args.selfcheck:
        run_selfcheck(registry)
        return

    # Load situations
    situations = load_situations(args.situations)
    if args.limit is not None:
        situations = situations[: args.limit]

    # Build tools array (all 8 tools always sent)
    tools_array = build_tools_array(registry)

    if args.predictions:
        backend = "predictions"
        predictions = []
        with open(args.predictions, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))
        timings_stats = {
            "requests": 0,
            "min_ms": 0,
            "median_ms": 0,
            "p95_ms": 0,
            "max_ms": 0,
            "total_s": 0,
        }
    else:
        if not args.base_url:
            parser.print_help(sys.stderr)
            sys.exit(2)
        backend = "http"
        predictions, timings_stats = run_http_eval(
            situations, tools_array, args.base_url, args.model, args.temperature, args.timeout, args.seed
        )

    # Score using existing function
    report = score_predictions(situations, predictions, registry, backend=backend)

    # Add phantom success
    phantom, phantom_rate = compute_phantom_success(situations, predictions)
    report["phantom_success"] = phantom
    report["phantom_success_rate"] = phantom_rate

    # Add metadata
    report["model"] = args.model
    report["seed"] = args.seed
    report["temperature"] = args.temperature
    report["timings"] = timings_stats

    # Output
    report_json = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    print(report_json)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_json + "\n")

    sys.exit(0)


if __name__ == "__main__":
    main()