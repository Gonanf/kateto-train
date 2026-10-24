"""Evaluates harness toolcalling metric for Kateto.

Measures toolcall accuracy, hallucinated tool names rate, and validates against
the fixed BUILTIN_TOOLS registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow importing registry from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_tool_registry import load_registry


def modelo_servido(base_url: str | None) -> str:
    """Fetch served model ID from /v1/models endpoint. Returns 'desconocido' on any failure."""
    if base_url is None:
        return "desconocido"
    b = base_url.rstrip("/")
    url = f"{b}/models" if b.endswith("/v1") else f"{b}/v1/models"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = data.get("data")
        if isinstance(models, list) and len(models) > 0:
            first = models[0]
            if isinstance(first, dict):
                model_id = first.get("id")
                if isinstance(model_id, str):
                    return model_id
    except Exception:
        pass
    return "desconocido"


def git_head() -> str:
    """Devuelve el sha del commit medido (git rev-parse HEAD) o 'desconocido' en error."""
    try:
        root = Path(__file__).resolve().parent.parent
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"advertencia: fallo git rev-parse HEAD ({exc})", file=sys.stderr)
        return "desconocido"


def normalize(s: str) -> str:
    """Normalize text: lowercase, NFD without combining diacritics, alphanumeric only, single spaces."""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z0-9]", " ", s)
    return " ".join(s.split())


def check_args_spec(spec_dict: dict, actual_args: dict) -> bool:
    """Check if actual_args satisfies the expected required parameters specification."""
    if not isinstance(actual_args, dict):
        return False

    for key, spec in spec_dict.items():
        if key not in actual_args:
            return False
        val = actual_args[key]

        if isinstance(spec, dict):
            if spec.get("any") is True:
                if val is None or val == "" or val == [] or val == {}:
                    return False
            elif "enum" in spec:
                norm_val = normalize(str(val))
                norm_enum = [normalize(str(x)) for x in spec["enum"]]
                if norm_val not in norm_enum:
                    return False
            elif "contains" in spec:
                norm_sub = normalize(str(spec["contains"]))
                norm_val = normalize(str(val))
                if norm_sub not in norm_val:
                    return False
            else:
                if not isinstance(val, dict):
                    return False
                if not check_args_spec(spec, val):
                    return False
        elif isinstance(spec, str):
            if normalize(str(val)) != normalize(spec):
                return False
        else:
            if type(val) is not type(spec) or val != spec:
                return False

    return True


def is_request_error(pred: dict) -> bool:
    """True si la prediccion es un error de transporte, no una respuesta del modelo."""
    out = pred.get("output")
    return isinstance(out, str) and "<request_error>" in out


def extract_tool_call(
    pred: dict, is_tool_expected: bool = True
) -> tuple[str | None, dict | None, bool]:
    """Extract tool call name and arguments from prediction dict.

    Order of extraction:
    1. tool_calls in prediction
    2. <tool_call>{...}</tool_call> in output
    3. output as plain JSON object

    Returns:
        (name, arguments, has_parse_error)
    """
    # 1. tool_calls in pred
    tool_calls = pred.get("tool_calls")
    if isinstance(tool_calls, list) and len(tool_calls) > 0:
        tc = tool_calls[0]
        if isinstance(tc, dict):
            func = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            name = func.get("name")
            raw_args = func.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except Exception:
                    return None, None, True
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            if isinstance(args, dict):
                return name, args, False
            return None, None, True

    if is_request_error(pred):
        return None, None, True

    output = pred.get("output", "")
    if not isinstance(output, str):
        output = str(output) if output is not None else ""

    # 2. <tool_call>{...}</tool_call> in output
    tc_match = re.search(r"<tool_call>(.*?)</tool_call>", output, re.DOTALL)
    if tc_match:
        raw_json = tc_match.group(1).strip()
        try:
            data = json.loads(raw_json)
        except Exception:
            return None, None, True

        if isinstance(data, dict):
            func = data.get("function") if isinstance(data.get("function"), dict) else data
            name = func.get("name")
            raw_args = func.get("arguments", func.get("parameters"))
            if raw_args is None:
                raw_args = {
                    k: v
                    for k, v in data.items()
                    if k not in ("name", "function", "parameters", "arguments")
                }
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except Exception:
                    return None, None, True
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}
            if isinstance(args, dict):
                return name, args, False
        return None, None, True

    # 3. output as plain JSON object
    stripped = output.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                func = data.get("function") if isinstance(data.get("function"), dict) else data
                name = func.get("name")
                raw_args = func.get("arguments", func.get("parameters"))
                if raw_args is None:
                    raw_args = {
                        k: v
                        for k, v in data.items()
                        if k not in ("name", "function", "parameters", "arguments")
                    }
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args)
                    except Exception:
                        return None, None, True
                elif isinstance(raw_args, dict):
                    args = raw_args
                else:
                    args = {}
                if isinstance(args, dict):
                    return name, args, False
        except Exception:
            if is_tool_expected:
                return None, None, True

    # No tool call extracted
    if is_tool_expected:
        return None, None, True
    return None, None, False


def score_predictions(
    situations: list[dict],
    predictions: list[dict] | dict[str, dict],
    registry: dict[str, dict],
    backend: str = "predictions",
) -> dict:
    """Score model predictions against expected situations and tool registry."""
    if isinstance(predictions, list):
        pred_map = {p["id"]: p for p in predictions if isinstance(p, dict) and "id" in p}
    else:
        pred_map = predictions

    n = len(situations)
    tool_n = 0
    no_tool_n = 0
    name_ok = 0
    full_ok = 0
    no_tool_ok = 0
    parse_errors = 0
    missing = 0
    hallucinated: list[dict] = []
    tool_n_llamados = 0
    n_request_errors = 0
    n_request_errors_tool = 0
    n_request_errors_no_tool = 0
    full_ok_tool = 0
    tool_validas = 0

    for sit in situations:
        s_id = sit["id"]
        kind = sit.get("kind", "tool")
        exp = sit.get("expect", {})
        exp_name = exp.get("name")
        exp_req = exp.get("required", {})

        if kind == "no_tool":
            no_tool_n += 1
        else:
            tool_n += 1

        pred = pred_map.get(s_id)
        if pred is None:
            missing += 1
            if kind == "tool":
                parse_errors += 1
            continue

        if is_request_error(pred):
            n_request_errors += 1
            if kind == "no_tool":
                n_request_errors_no_tool += 1
            else:
                n_request_errors_tool += 1
            continue

        if kind != "no_tool":
            tool_validas += 1

        called_name, called_args, has_parse_error = extract_tool_call(
            pred, is_tool_expected=(kind == "tool")
        )

        if has_parse_error:
            parse_errors += 1
            continue

        if called_name is not None:
            tool_n_llamados += 1
            if called_name not in registry:
                hallucinated.append({"id": s_id, "name": called_name})

        if kind == "no_tool":
            if called_name is None:
                name_ok += 1
                full_ok += 1
                no_tool_ok += 1
        else:
            if called_name is not None:
                is_name_ok = called_name == exp_name
                if is_name_ok:
                    name_ok += 1
                    if check_args_spec(exp_req, called_args):
                        full_ok += 1
                        full_ok_tool += 1

    name_accuracy = round(name_ok / n, 4) if n else 0.0
    toolcall_accuracy = round(full_ok / n, 4) if n else 0.0
    no_tool_accuracy = round(no_tool_ok / no_tool_n, 4) if no_tool_n else 0.0
    hallucinated_rate = (
        round(len(hallucinated) / tool_n_llamados, 4) if tool_n_llamados else 0.0
    )

    n_validas = n - n_request_errors
    celda_valida = n_request_errors == 0
    toolcall_accuracy_validas = round(full_ok / n_validas, 4) if n_validas > 0 else 0.0
    accuracy_tool_validas = round(full_ok_tool / tool_validas, 4) if tool_validas > 0 else 0.0

    return {
        "n": n,
        "tool_n": tool_n,
        "no_tool_n": no_tool_n,
        "name_ok": name_ok,
        "name_accuracy": name_accuracy,
        "full_ok": full_ok,
        "toolcall_accuracy": toolcall_accuracy,
        "no_tool_ok": no_tool_ok,
        "no_tool_accuracy": no_tool_accuracy,
        "parse_errors": parse_errors,
        "missing": missing,
        "hallucinated": hallucinated,
        "hallucinated_rate": hallucinated_rate,
        "backend": backend,
        "n_request_errors": n_request_errors,
        "n_request_errors_tool": n_request_errors_tool,
        "n_request_errors_no_tool": n_request_errors_no_tool,
        "n_validas": n_validas,
        "celda_valida": celda_valida,
        "full_ok_tool": full_ok_tool,
        "toolcall_accuracy_validas": toolcall_accuracy_validas,
        "accuracy_tool_validas": accuracy_tool_validas,
    }


def compute_extra_metrics(
    situations: list[dict],
    predictions: list[dict] | dict[str, dict],
    registry: dict[str, dict],
) -> dict:
    """Compute extra toolcall metrics: call counts, accuracy on calls, and spec parseability ceiling."""
    if isinstance(predictions, list):
        pred_map = {p["id"]: p for p in predictions if isinstance(p, dict) and "id" in p}
    else:
        pred_map = predictions

    n_llamadas_tools = 0
    full_ok_tool = 0
    evaluables = 0
    no_evaluables = 0
    tool_n = 0
    exceptions_list: list[Exception] = []

    for sit in situations:
        s_id = sit.get("id")
        kind = sit.get("kind", "tool")
        exp = sit.get("expect", {})
        exp_name = exp.get("name")
        exp_req = exp.get("required")

        if kind != "no_tool":
            tool_n += 1
            try:
                check_args_spec(exp_req, {})
                evaluables += 1
            except Exception as exc:
                no_evaluables += 1
                exceptions_list.append(exc)

        pred = pred_map.get(s_id)
        if pred is None:
            continue

        called_name, called_args, has_parse_error = extract_tool_call(
            pred, is_tool_expected=(kind == "tool")
        )

        if has_parse_error:
            continue

        if called_name is not None:
            n_llamadas_tools += 1

        if kind != "no_tool":
            if called_name is not None and called_name == exp_name:
                try:
                    if check_args_spec(exp_req, called_args):
                        full_ok_tool += 1
                except Exception:
                    pass

    accuracy_sobre_llamadas = (
        round(full_ok_tool / n_llamadas_tools, 4) if n_llamadas_tools > 0 else 0.0
    )
    techo_parseable = round(evaluables / tool_n, 4) if tool_n > 0 else 0.0
    techo_parseable_n = {
        "evaluables": evaluables,
        "no_evaluables": no_evaluables,
        "tool_n": tool_n,
    }

    if no_evaluables == 0:
        techo_parseable_motivo = (
            f"{evaluables}/{tool_n} evaluables: expect.required es dict en todas"
        )
    else:
        err_counts: dict[str, int] = {}
        for exc in exceptions_list:
            key = f"{type(exc).__name__}: {exc}"
            err_counts[key] = err_counts.get(key, 0) + 1
        sorted_errs = sorted(err_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        err_msg = ", ".join(f"{k} (x{v})" for k, v in sorted_errs)
        techo_parseable_motivo = f"{no_evaluables}/{tool_n} no evaluables: {err_msg}"

    return {
        "n_llamadas_tools": n_llamadas_tools,
        "full_ok_tool": full_ok_tool,
        "accuracy_sobre_llamadas": accuracy_sobre_llamadas,
        "techo_parseable": techo_parseable,
        "techo_parseable_n": techo_parseable_n,
        "techo_parseable_motivo": techo_parseable_motivo,
    }


def build_raw_prompt(situation: str) -> str:
    """Wrapper crudo del dataset de entrenamiento: '<|im_user|>' + situation + '<|im_end|>\n<|im_start|>seco\n'"""
    return f"<|im_user|>{situation}<|im_end|>\n<|im_start|>seco\n"


def build_tools_schema(registry: dict[str, dict]) -> list[dict]:
    """Build OpenAI-compatible tool schema definitions from tool registry."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {p: {"type": t} for p, t in data["props"].items()},
                    "required": data["required"],
                },
            },
        }
        for name, data in registry.items()
    ]


def load_situations(situations_path: str) -> list[dict]:
    """Load situations list from JSONL file."""
    if not os.path.exists(situations_path):
        raise SystemExit(f"Situations file not found: {situations_path}")

    situations: list[dict] = []
    with open(situations_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                situations.append(json.loads(line))
    return situations


def run_http_eval(
    situations: list[dict],
    registry: dict[str, dict],
    base_url: str,
    model: str,
    limit: int | None,
    temperature: float,
    timeout: float,
    transport: str = "chat",
    stop_mode: str = "default",
    dump_predictions: str | Path | None = None,
    max_tokens: int = 256,
) -> list[dict]:
    """Execute evaluation querying an OpenAI-compatible HTTP endpoint."""
    if limit is not None:
        situations = situations[:limit]

    tools_schema = build_tools_schema(registry)

    b = base_url.rstrip("/")
    if transport == "raw":
        endpoint = f"{b}/completions" if b.endswith("/v1") else f"{b}/v1/completions"
    else:
        endpoint = f"{b}/chat/completions" if b.endswith("/v1") else f"{b}/v1/chat/completions"

    predictions: list[dict] = []

    for sit in situations:
        if transport == "raw":
            raw_prompt = build_raw_prompt(sit["situation"])
            payload = {
                "model": model,
                "prompt": raw_prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
            }
            if stop_mode == "im_end_only":
                payload["stop"] = ["<|im_end|>"]
                payload["repeat_penalty"] = 1.0

            prompt_sha256 = hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()
        else:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": sit["situation"]}],
                "tools": tools_schema,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if stop_mode == "im_end_only":
                payload["stop"] = ["<|im_end|>"]
                payload["repeat_penalty"] = 1.0

            chat_prompt = {
                "messages": payload["messages"],
                "tools": payload["tools"],
            }
            canon_json = json.dumps(
                chat_prompt, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            )
            prompt_sha256 = hashlib.sha256(canon_json.encode("utf-8")).hexdigest()

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                resp_json = json.loads(response.read().decode("utf-8"))
                choice = resp_json.get("choices", [{}])[0]
                finish_reason = choice.get("finish_reason")
                usage = resp_json.get("usage")
                tokens = (
                    usage.get("completion_tokens")
                    if isinstance(usage, dict)
                    else None
                )

                if transport == "raw":
                    output = choice.get("text") or ""
                    tool_calls = None
                else:
                    msg = choice.get("message", {})
                    output = msg.get("content") or ""
                    tool_calls = msg.get("tool_calls")

                predictions.append({
                    "id": sit["id"],
                    "kind": sit.get("kind", "tool"),
                    "output": output,
                    "tool_calls": tool_calls,
                    "finish_reason": finish_reason,
                    "tokens": tokens,
                    "prompt_sha256": prompt_sha256,
                })
        except Exception as exc:
            # Count request_error in parse_errors by recording a malformed output
            predictions.append({
                "id": sit["id"],
                "kind": sit.get("kind", "tool"),
                "output": f"<request_error>{exc}</request_error>",
                "tool_calls": None,
                "finish_reason": None,
                "tokens": None,
                "prompt_sha256": prompt_sha256,
            })

    if dump_predictions is not None:
        dump_p = Path(dump_predictions)
        dump_p.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_p, "w", encoding="utf-8") as f:
            for pred in predictions:
                record = {
                    "id": pred["id"],
                    "kind": pred.get("kind", "tool"),
                    "output": pred.get("output", ""),
                    "tool_calls": pred.get("tool_calls"),
                    "finish_reason": pred.get("finish_reason"),
                    "tokens": pred.get("tokens"),
                    "prompt_sha256": pred.get("prompt_sha256"),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return predictions


def compute_traceability_metrics(
    situations: list[dict],
    predictions: list[dict] | dict[str, dict],
    transport: str = "chat",
    stop_mode: str = "default",
    tools_schema: list[dict] | None = None,
) -> dict:
    """Compute matrix traceability metrics: hashes, finish reasons, token lengths, marker distributions."""
    situaciones_data = [[s["id"], s["situation"]] for s in situations]
    sit_json = json.dumps(
        situaciones_data, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    )
    situaciones_sha256 = hashlib.sha256(sit_json.encode("utf-8")).hexdigest()

    if isinstance(predictions, list):
        pred_map = {p["id"]: p for p in predictions if isinstance(p, dict) and "id" in p}
    else:
        pred_map = predictions

    prompt_shas: list[str] = []
    finish_reason_counts: dict[str, int] = {}
    n_truncados = 0
    n_vacios = 0
    n_request_errors_tc = 0
    marcadores = {
        "con_tool_call_completo": 0,
        "con_tool_call_sin_cerrar": 0,
        "sin_marcador": 0,
        "vacios": 0,
        "request_error": 0,
    }

    for sit in situations:
        pred = pred_map.get(sit["id"]) or {}

        p_sha = pred.get("prompt_sha256")
        if not p_sha:
            if transport == "raw":
                raw_prompt = build_raw_prompt(sit["situation"])
                p_sha = hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()
            else:
                chat_prompt = {
                    "messages": [{"role": "user", "content": sit["situation"]}],
                    "tools": tools_schema or [],
                }
                canon_json = json.dumps(
                    chat_prompt, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
                p_sha = hashlib.sha256(canon_json.encode("utf-8")).hexdigest()
        prompt_shas.append(p_sha)

        fr = pred.get("finish_reason")
        fr_key = "null" if fr is None else str(fr)
        finish_reason_counts[fr_key] = finish_reason_counts.get(fr_key, 0) + 1
        if fr == "length":
            n_truncados += 1

        if is_request_error(pred):
            n_request_errors_tc += 1
            marcadores["request_error"] += 1
            continue

        tc_list = pred.get("tool_calls")
        if isinstance(tc_list, list) and len(tc_list) > 0:
            marcadores["con_tool_call_completo"] += 1
        else:
            out = pred.get("output")
            out_str = str(out) if out is not None else ""
            is_vacio = not out_str.strip()
            if is_vacio:
                n_vacios += 1
                marcadores["vacios"] += 1
            else:
                if re.search(r"<tool_call>.*?</tool_call>", out_str, re.DOTALL):
                    marcadores["con_tool_call_completo"] += 1
                elif "<tool_call>" in out_str and "</tool_call>" not in out_str:
                    marcadores["con_tool_call_sin_cerrar"] += 1
                else:
                    marcadores["sin_marcador"] += 1

    request_sha256 = hashlib.sha256("".join(prompt_shas).encode("utf-8")).hexdigest()

    return {
        "transport": transport,
        "stop_mode": stop_mode,
        "situaciones_sha256": situaciones_sha256,
        "request_sha256": request_sha256,
        "finish_reason_counts": finish_reason_counts,
        "n_truncados": n_truncados,
        "n_vacios": n_vacios,
        "marcadores": marcadores,
        "n_request_errors": n_request_errors_tc,
        "celda_valida": n_request_errors_tc == 0,
    }


def build_report(
    situations: list[dict],
    predictions: list[dict] | dict[str, dict],
    registry: dict[str, dict],
    backend: str = "http",
    transport: str = "chat",
    stop_mode: str = "default",
    base_url: str | None = None,
    model: str = "kateto",
    cmd: str = "",
) -> dict:
    """Build full evaluation report including accuracy, registry validation, extra metrics, and matrix traceability."""
    report = score_predictions(situations, predictions, registry, backend=backend)

    report.update({
        "modelo_pedido": model,
        "modelo_servido": modelo_servido(base_url),
        "base_url": base_url if base_url is not None else "n/a",
        "medido": datetime.now(timezone(timedelta(hours=-3))).isoformat(timespec="seconds"),
        "cmd": cmd,
        "git_head": git_head(),
    })

    report.update(compute_extra_metrics(situations, predictions, registry))
    tools_schema = build_tools_schema(registry)
    report.update(
        compute_traceability_metrics(
            situations,
            predictions,
            transport=transport,
            stop_mode=stop_mode,
            tools_schema=tools_schema,
        )
    )
    return report


def run_selfcheck(registry: dict[str, dict], situations_path: str) -> None:
    """Run in-memory selfcheck with 6 synthetic predictions and assert exact metric counts."""
    situations = load_situations(situations_path)

    # 6 synthetic predictions:
    # 1. Perfecta
    # 2. Nombre equivocado del registro (run_command instead of read_file)
    # 3. Nombre inventado read_doc
    # 4. Falta un required (empty arguments)
    # 5. no_tool bien (plain text)
    # 6. no_tool llamando una tool (calls get_current_time)
    test_predictions = [
        {
            "id": "hx-001",
            "tool_calls": [{"name": "run_command", "arguments": {"command": "git status"}}],
        },
        {
            "id": "hx-002",
            "tool_calls": [{"name": "run_command", "arguments": {"command": "cat docs/plan.md"}}],
        },
        {
            "id": "hx-003",
            "tool_calls": [{"name": "read_doc", "arguments": {"path": "report.txt"}}],
        },
        {
            "id": "hx-004",
            "tool_calls": [{"name": "delete_file", "arguments": {}}],
        },
        {
            "id": "hx-029",
            "output": "El clima parece bastante agradable hoy en la ciudad.",
        },
        {
            "id": "hx-030",
            "tool_calls": [{"name": "get_current_time", "arguments": {}}],
        },
    ]

    report = score_predictions(
        situations, test_predictions, registry, backend="predictions"
    )

    try:
        assert report["n"] == 36, f"Expected n=36, got {report['n']}"
        assert report["tool_n"] == 28, f"Expected tool_n=28, got {report['tool_n']}"
        assert report["no_tool_n"] == 8, f"Expected no_tool_n=8, got {report['no_tool_n']}"
        assert report["name_ok"] == 3, f"Expected name_ok=3, got {report['name_ok']}"
        assert report["name_accuracy"] == 0.0833, f"Expected name_acc=0.0833, got {report['name_accuracy']}"
        assert report["full_ok"] == 2, f"Expected full_ok=2, got {report['full_ok']}"
        assert report["toolcall_accuracy"] == 0.0556, f"Expected toolcall_acc=0.0556, got {report['toolcall_accuracy']}"
        assert report["no_tool_ok"] == 1, f"Expected no_tool_ok=1, got {report['no_tool_ok']}"
        assert report["no_tool_accuracy"] == 0.125, f"Expected no_tool_acc=0.125, got {report['no_tool_accuracy']}"
        assert report["parse_errors"] == 24, f"Expected parse_errors=24, got {report['parse_errors']}"
        assert report["missing"] == 30, f"Expected missing=30, got {report['missing']}"
        assert len(report["hallucinated"]) == 1, f"Expected 1 hallucinated, got {len(report['hallucinated'])}"
        assert report["hallucinated"][0]["name"] == "read_doc", f"Expected hallucinated read_doc, got {report['hallucinated']}"
        assert report["hallucinated_rate"] == 0.2, f"Expected hallucinated_rate=0.2, got {report['hallucinated_rate']}"
    except AssertionError as exc:
        print(f"Selfcheck assertion failed: {exc}", file=sys.stderr)
        sys.exit(1)

    extra = compute_extra_metrics(situations, test_predictions, registry)
    try:
        assert extra["n_llamadas_tools"] == 5, f"Expected n_llamadas_tools=5, got {extra['n_llamadas_tools']}"
        assert extra["full_ok_tool"] == 1, f"Expected full_ok_tool=1, got {extra['full_ok_tool']}"
        assert extra["accuracy_sobre_llamadas"] == 0.2, f"Expected accuracy_sobre_llamadas=0.2, got {extra['accuracy_sobre_llamadas']}"
        assert extra["techo_parseable"] == 1.0, f"Expected techo_parseable=1.0, got {extra['techo_parseable']}"
        assert extra["techo_parseable_n"] == {"evaluables": 28, "no_evaluables": 0, "tool_n": 28}, f"Expected techo_parseable_n={{'evaluables': 28, 'no_evaluables': 0, 'tool_n': 28}}, got {extra['techo_parseable_n']}"
    except AssertionError as exc:
        print(f"Selfcheck assertion failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print("selfcheck ok")
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate harness toolcalling accuracy for Kateto.")
    parser.add_argument(
        "--situations",
        default="evals/toolcall_harness/situations.jsonl",
        help="Path to situations.jsonl",
    )
    parser.add_argument(
        "--tools-py",
        default="/home/chaos/proyectos/OpenaiBuildWeek/Kateto/kateto/voices/tools.py",
        help="Path to voices/tools.py",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        help="Path to model predictions JSONL",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--model",
        default="kateto",
        help="Model name for HTTP requests",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of situations evaluated",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for HTTP generation",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Timeout in seconds for HTTP requests",
    )
    parser.add_argument(
        "--transport",
        choices=["chat", "raw"],
        default="chat",
        help="Transport protocol: chat (/v1/chat/completions) or raw (/v1/completions)",
    )
    parser.add_argument(
        "--stop-mode",
        choices=["default", "im_end_only"],
        default="default",
        help="Stop sequence mode: default or im_end_only (<|im_end|> and repeat_penalty 1.0)",
    )
    parser.add_argument(
        "--dump-predictions",
        default=None,
        help="Path to dump JSONL predictions with traceability keys",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output file path for evaluation report JSON",
    )
    parser.add_argument(
        "--fail-on-hallucination",
        action="store_true",
        help="Exit with code 1 if hallucinated_rate > 0",
    )
    parser.add_argument(
        "--selfcheck",
        action="store_true",
        help="Run synthetic in-memory selfcheck and verify metric scoring",
    )

    args = parser.parse_args()

    registry = load_registry(args.tools_py)

    if args.selfcheck:
        run_selfcheck(registry, args.situations)
        return

    if not args.predictions and not args.base_url:
        parser.print_help(sys.stderr)
        sys.exit(2)

    situations = load_situations(args.situations)
    if args.limit is not None:
        situations = situations[: args.limit]

    if args.predictions:
        backend = "predictions"
        predictions = []
        with open(args.predictions, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))
        if args.dump_predictions:
            dump_p = Path(args.dump_predictions)
            dump_p.parent.mkdir(parents=True, exist_ok=True)
            with open(dump_p, "w", encoding="utf-8") as f:
                for pred in predictions:
                    record = {
                        "id": pred["id"],
                        "kind": pred.get("kind", "tool"),
                        "output": pred.get("output", ""),
                        "tool_calls": pred.get("tool_calls"),
                        "finish_reason": pred.get("finish_reason"),
                        "tokens": pred.get("tokens"),
                        "prompt_sha256": pred.get("prompt_sha256"),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        backend = "http"
        predictions = run_http_eval(
            situations,
            registry,
            args.base_url,
            args.model,
            args.limit,
            args.temperature,
            args.timeout,
            transport=args.transport,
            stop_mode=args.stop_mode,
            dump_predictions=args.dump_predictions,
        )

    report = build_report(
        situations=situations,
        predictions=predictions,
        registry=registry,
        backend=backend,
        transport=args.transport,
        stop_mode=args.stop_mode,
        base_url=args.base_url,
        model=args.model,
        cmd=" ".join(sys.argv),
    )

    report_json = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    print(report_json)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_json + "\n")

    if args.fail_on-hallucination if hasattr(args, "fail_on-hallucination") else args.fail_on_hallucination:
        if report["hallucinated_rate"] > 0:
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
