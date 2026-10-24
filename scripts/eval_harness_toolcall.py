"""Evaluates harness toolcalling metric for Kateto.

Measures toolcall accuracy, hallucinated tool names rate, and validates against
the fixed BUILTIN_TOOLS registry. Supports optional GBNF grammar constraining
and dual-arm (control vs grammar) evaluation.
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
    url = f"{base_url.rstrip('/')}/v1/models"
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


def get_chat_endpoint(base_url: str) -> str:
    """Return OpenAI-compatible chat completions endpoint URL."""
    b_url = base_url.rstrip("/")
    if not b_url.endswith("/v1"):
        return f"{b_url}/v1/chat/completions"
    return f"{b_url}/chat/completions"


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

    name_accuracy = round(name_ok / n, 4) if n else 0.0
    toolcall_accuracy = round(full_ok / n, 4) if n else 0.0
    no_tool_accuracy = round(no_tool_ok / no_tool_n, 4) if no_tool_n else 0.0
    hallucinated_rate = (
        round(len(hallucinated) / tool_n_llamados, 4) if tool_n_llamados else 0.0
    )

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
    grammar: str | None = None,
) -> list[dict]:
    """Execute evaluation querying an OpenAI-compatible HTTP endpoint."""
    if limit is not None:
        situations = situations[:limit]

    tools_schema = [
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

    endpoint = get_chat_endpoint(base_url)
    predictions: list[dict] = []

    for sit in situations:
        payload: dict[str, any] = {
            "model": model,
            "messages": [{"role": "user", "content": sit["situation"]}],
            "tools": tools_schema,
            "temperature": temperature,
            "max_tokens": 256,
        }
        if grammar is not None:
            payload["grammar"] = grammar

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=max(timeout, 600.0)) as response:
                resp_json = json.loads(response.read().decode("utf-8"))
                choice = resp_json.get("choices", [{}])[0]
                msg = choice.get("message", {})
                predictions.append({
                    "id": sit["id"],
                    "tool_calls": msg.get("tool_calls"),
                    "output": msg.get("content") or "",
                })
        except Exception as exc:
            # Count request_error in parse_errors by recording a malformed output
            predictions.append({
                "id": sit["id"],
                "output": f"<request_error>{exc}</request_error>",
            })

    return predictions


def probe_grammar(
    base_url: str,
    model: str,
    registry: dict[str, dict],
    gbnf_content: str,
    temperature: float,
    timeout: float,
) -> dict:
    """Verify that the HTTP endpoint honors GBNF grammar as required."""
    endpoint = get_chat_endpoint(base_url)

    timeout = max(timeout, 600.0)
    # 1. Invalid grammar probe: must return HTTP >= 400
    payload_inv = {
        "model": model,
        "messages": [{"role": "user", "content": "herramientas"}],
        "temperature": temperature,
        "max_tokens": 16,
        "grammar": 'root ::= "abc',
    }
    req_inv = urllib.request.Request(
        endpoint,
        data=json.dumps(payload_inv).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    invalida_status = 0
    invalida_raw = ""
    try:
        with urllib.request.urlopen(req_inv, timeout=timeout) as resp:
            invalida_status = resp.status
            invalida_raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        invalida_status = exc.code
        invalida_raw = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        invalida_raw = str(exc)

    if invalida_status < 400:
        print(
            f"ERROR: Grammar probe invalid check failed. Expected HTTP >= 400, got {invalida_status}: {invalida_raw}",
            file=sys.stderr,
        )
        sys.exit(3)

    # 2. Valid grammar probe: must return 200 and conforming content
    payload_val = {
        "model": model,
        "messages": [{"role": "user", "content": "herramientas"}],
        "temperature": temperature,
        "max_tokens": 64,
        "grammar": gbnf_content,
    }
    req_val = urllib.request.Request(
        endpoint,
        data=json.dumps(payload_val).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    valida_status = 0
    valida_raw = ""
    valida_conforme = False
    try:
        with urllib.request.urlopen(req_val, timeout=timeout) as resp:
            valida_status = resp.status
            valida_raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        valida_status = exc.code
        valida_raw = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        valida_raw = str(exc)

    if valida_status == 200:
        try:
            resp_json = json.loads(valida_raw)
            choice = resp_json.get("choices", [{}])[0]
            msg = choice.get("message", {})
            out_content = (msg.get("content") or "").strip()
            pred = {
                "id": "probe",
                "tool_calls": msg.get("tool_calls"),
                "output": msg.get("content") or "",
            }
            _, _, parse_err = extract_tool_call(pred, is_tool_expected=True)
            if not parse_err:
                valida_conforme = True
            elif not out_content.startswith("{") and not out_content.startswith("<tool_call"):
                valida_conforme = True
            else:
                valida_conforme = False
        except Exception:
            valida_conforme = False
    else:
        valida_conforme = False

    if not valida_conforme:
        print(
            f"ERROR: Grammar probe valid check non-conforming. HTTP {valida_status}: {valida_raw}",
            file=sys.stderr,
        )
        sys.exit(3)

    return {
        "invalida_status": invalida_status,
        "valida_status": valida_status,
        "valida_conforme": valida_conforme,
        "endpoint": "/v1/chat/completions",
    }


def dump_predictions(predictions: list[dict], path: str | Path) -> None:
    """Dump predictions to JSONL file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for pred in predictions:
            rec = {
                "id": pred.get("id"),
                "tool_calls": pred.get("tool_calls"),
                "output": pred.get("output") or "",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_selfcheck_grammar(
    registry: dict[str, dict], tools_py: str, gbnf_path: str = "scripts/toolcall.gbnf"
) -> None:
    """Verify GBNF grammar generation, file identity, and structural constraints."""
    try:
        from build_toolcall_gbnf import generate_gbnf
    except ImportError:
        try:
            from build_toolcall_gbnf import build_grammar as generate_gbnf
        except ImportError as exc:
            print(f"Selfcheck-grammar failed: cannot import build_toolcall_gbnf: {exc}", file=sys.stderr)
            sys.exit(1)

    generated = generate_gbnf(tools_py)
    gen_bytes = generated.encode("utf-8")

    # (a) Check scripts/toolcall.gbnf exists and is byte-identical
    p = Path(gbnf_path)
    if not p.exists():
        print(f"Selfcheck-grammar failed: {gbnf_path} does not exist", file=sys.stderr)
        sys.exit(1)
    disk_bytes = p.read_bytes()
    if disk_bytes != gen_bytes:
        print(f"Selfcheck-grammar failed: {gbnf_path} does not match generated content byte-for-byte", file=sys.stderr)
        sys.exit(1)

    # (b) Check that it contains all 18 tool names from the registry
    if len(registry) < 18:
        print(f"Selfcheck-grammar failed: expected at least 18 tools in registry, got {len(registry)}", file=sys.stderr)
        sys.exit(1)
    for name in registry.keys():
        literal = f'"\\"{name}\\""'
        if literal not in generated:
            print(f"Selfcheck-grammar failed: tool name literal {literal} not found in grammar", file=sys.stderr)
            sys.exit(1)

    # (c) Check branches: tool-call, json-plano, texto
    for rule in ["tool-call", "json-plano", "texto"]:
        if f"{rule} ::=" not in generated:
            print(f"Selfcheck-grammar failed: rule '{rule} ::=' missing in grammar", file=sys.stderr)
            sys.exit(1)
    cuerpo_m = re.search(r"cuerpo\s*::=\s*(.+)", generated)
    if not cuerpo_m:
        print("Selfcheck-grammar failed: rule 'cuerpo ::=' missing in grammar", file=sys.stderr)
        sys.exit(1)
    cuerpo_branches = [b.strip() for b in cuerpo_m.group(1).split("|")]
    for rule in ["tool-call", "json-plano", "texto"]:
        if rule not in cuerpo_branches:
            print(f"Selfcheck-grammar failed: branch '{rule}' missing in cuerpo", file=sys.stderr)
            sys.exit(1)

    # (d) Check that texto-char does not accept '{', '}', '<'
    texto_char_m = re.search(r"texto-char\s*::=\s*\[([^\]]+)\]", generated)
    if not texto_char_m:
        print("Selfcheck-grammar failed: 'texto-char ::=' character class missing", file=sys.stderr)
        sys.exit(1)
    raw_class = texto_char_m.group(1)
    for forbidden in ["{", "}", "<"]:
        if forbidden in raw_class:
            print(f"Selfcheck-grammar failed: texto-char class contains '{forbidden}'", file=sys.stderr)
            sys.exit(1)
    compiled = re.compile(f"^[{raw_class}]$")
    for forbidden in ["{", "}", "<"]:
        if compiled.match(forbidden):
            print(f"Selfcheck-grammar failed: texto-char accepts forbidden char '{forbidden}'", file=sys.stderr)
            sys.exit(1)

    # (e) Check str-char rule and cadena rule
    if "str-char ::=" not in generated:
        print("Selfcheck-grammar failed: rule 'str-char ::=' missing in grammar", file=sys.stderr)
        sys.exit(1)
    cadena_m = re.search(r"cadena\s*::=\s*(.+)", generated)
    if not cadena_m or cadena_m.group(1).strip() != '"\\"" str-char* "\\""':
        print('Selfcheck-grammar failed: rule cadena must be \'"\\"" str-char* "\\""\'', file=sys.stderr)
        sys.exit(1)
    str_char_m = re.search(r"str-char\s*::=\s*\[\^([^\]]+)\]", generated)
    if not str_char_m:
        print("Selfcheck-grammar failed: negated character class '[^...]' missing in str-char rule", file=sys.stderr)
        sys.exit(1)
    str_char_negated = str_char_m.group(1)
    for forbidden in ["/", "_"]:
        if forbidden in str_char_negated:
            print(f"Selfcheck-grammar failed: str-char negated character class contains '{forbidden}' (must allow it)", file=sys.stderr)
            sys.exit(1)

    print("selfcheck-grammar ok")
    sys.exit(0)


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
        default=300.0,
        help="Timeout in seconds for HTTP requests",
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
    parser.add_argument(
        "--grammar",
        default="none",
        help="Grammar to apply: 'none' or path to GBNF file (default: none)",
    )
    parser.add_argument(
        "--gbnf",
        default="scripts/toolcall.gbnf",
        help="Path to GBNF grammar file for --grammar-both (default: scripts/toolcall.gbnf)",
    )
    parser.add_argument(
        "--grammar-both",
        action="store_true",
        help="Run both control and grammar arms in the same process",
    )
    parser.add_argument(
        "--dump-predictions",
        default=None,
        help="Path to dump predictions JSONL",
    )
    parser.add_argument(
        "--selfcheck-grammar",
        action="store_true",
        help="Verify GBNF grammar generation and constraints",
    )

    args = parser.parse_args()

    registry = load_registry(args.tools_py)

    if args.selfcheck:
        run_selfcheck(registry, args.situations)
        return

    if args.selfcheck_grammar:
        run_selfcheck_grammar(registry, args.tools_py, args.gbnf)
        return

    if not args.predictions and not args.base_url:
        parser.print_help(sys.stderr)
        sys.exit(2)

    situations = load_situations(args.situations)
    if args.limit is not None:
        situations = situations[: args.limit]

    servido = modelo_servido(args.base_url)
    head_sha = git_head()
    now_iso = datetime.now(timezone(timedelta(hours=-3))).isoformat(timespec="seconds")
    base_url_str = args.base_url if args.base_url is not None else "n/a"

    if args.grammar_both:
        if not args.base_url:
            print("Error: --grammar-both requires --base-url", file=sys.stderr)
            sys.exit(2)

        gbnf_file_path = Path(args.gbnf)
        if not gbnf_file_path.exists():
            print(f"Error: GBNF file not found: {args.gbnf}", file=sys.stderr)
            sys.exit(1)
        gbnf_content = gbnf_file_path.read_text(encoding="utf-8")
        grammar_sha256 = hashlib.sha256(gbnf_content.encode("utf-8")).hexdigest()

        # Probe grammar once before measuring
        probe_res = probe_grammar(
            args.base_url,
            args.model,
            registry,
            gbnf_content,
            args.temperature,
            args.timeout,
        )

        # Arm 1: Control (grammar absent)
        preds_control = run_http_eval(
            situations,
            registry,
            args.base_url,
            args.model,
            args.limit,
            args.temperature,
            args.timeout,
            grammar=None,
        )
        rep_control = score_predictions(situations, preds_control, registry, backend="http")
        rep_control.update({
            "modelo_pedido": args.model,
            "modelo_servido": servido,
            "base_url": base_url_str,
            "medido": now_iso,
            "cmd": " ".join(sys.argv),
            "git_head": head_sha,
            "grammar_aplicada": False,
            "grammar_sha256": None,
            "grammar_file": None,
        })
        rep_control.update(compute_extra_metrics(situations, preds_control, registry))

        # Arm 2: Gramatica (grammar present)
        preds_grammar = run_http_eval(
            situations,
            registry,
            args.base_url,
            args.model,
            args.limit,
            args.temperature,
            args.timeout,
            grammar=gbnf_content,
        )
        rep_grammar = score_predictions(situations, preds_grammar, registry, backend="http")
        rep_grammar.update({
            "modelo_pedido": args.model,
            "modelo_servido": servido,
            "base_url": base_url_str,
            "medido": now_iso,
            "cmd": " ".join(sys.argv),
            "git_head": head_sha,
            "grammar_aplicada": True,
            "grammar_sha256": grammar_sha256,
            "grammar_file": str(args.gbnf),
            "grammar_endpoint": "/v1/chat/completions",
            "grammar_probe": probe_res,
        })
        rep_grammar.update(compute_extra_metrics(situations, preds_grammar, registry))

        if args.dump_predictions:
            dump_p = Path(args.dump_predictions)
            ctrl_dump = dump_p.parent / f"{dump_p.stem}-control{dump_p.suffix}"
            gram_dump = dump_p.parent / f"{dump_p.stem}-gramatica{dump_p.suffix}"
            dump_predictions(preds_control, ctrl_dump)
            dump_predictions(preds_grammar, gram_dump)

        delta_toolcall_accuracy = round(
            rep_grammar["toolcall_accuracy"] - rep_control["toolcall_accuracy"], 4
        )
        report = {
            "delta_toolcall_accuracy": delta_toolcall_accuracy,
            "n": rep_grammar["n"],
            "n_llamadas_tools": rep_grammar["n_llamadas_tools"],
            "toolcall_accuracy": rep_grammar["toolcall_accuracy"],
            "parse_errors": rep_grammar["parse_errors"],
            "grammar_aplicada": True,
            "grammar_sha256": grammar_sha256,
            "modelo_pedido": args.model,
            "modelo_servido": servido,
            "base_url": base_url_str,
            "medido": now_iso,
            "cmd": " ".join(sys.argv),
            "git_head": head_sha,
            "grammar_endpoint": "/v1/chat/completions",
            "grammar_probe": probe_res,
            "brazos": {
                "control": rep_control,
                "gramatica": rep_grammar,
            },
        }

        report_json = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
        print(report_json)

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report_json + "\n")

        if args.fail_on_hallucination:
            if rep_grammar.get("hallucinated_rate", 0.0) > 0:
                sys.exit(1)

        sys.exit(0)

    # Single arm execution
    use_grammar = args.grammar != "none"
    gbnf_content = None
    grammar_sha256 = None
    probe_res = None

    if use_grammar:
        gbnf_file_path = Path(args.grammar)
        if not gbnf_file_path.exists():
            print(f"Error: Grammar file not found: {args.grammar}", file=sys.stderr)
            sys.exit(1)
        gbnf_content = gbnf_file_path.read_text(encoding="utf-8")
        grammar_sha256 = hashlib.sha256(gbnf_content.encode("utf-8")).hexdigest()

        if args.base_url:
            probe_res = probe_grammar(
                args.base_url,
                args.model,
                registry,
                gbnf_content,
                args.temperature,
                args.timeout,
            )

    if args.predictions:
        backend = "predictions"
        predictions = []
        with open(args.predictions, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))
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
            grammar=gbnf_content if use_grammar else None,
        )

    if args.dump_predictions:
        dump_predictions(predictions, args.dump_predictions)

    report = score_predictions(situations, predictions, registry, backend=backend)

    report.update({
        "modelo_pedido": args.model,
        "modelo_servido": servido,
        "base_url": base_url_str,
        "medido": now_iso,
        "cmd": " ".join(sys.argv),
        "git_head": head_sha,
        "grammar_aplicada": use_grammar,
        "grammar_sha256": grammar_sha256 if use_grammar else None,
        "grammar_file": str(args.grammar) if use_grammar else None,
    })

    if use_grammar and probe_res is not None:
        report["grammar_endpoint"] = "/v1/chat/completions"
        report["grammar_probe"] = probe_res

    report.update(compute_extra_metrics(situations, predictions, registry))

    report_json = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    print(report_json)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_json + "\n")

    if args.fail_on_hallucination:
        if report.get("hallucinated_rate", 0.0) > 0:
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
