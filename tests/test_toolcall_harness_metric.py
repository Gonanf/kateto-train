"""Tests for Kateto harness toolcall metric and evaluation suite."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import pytest


def import_module_from_path(name: str, path: str | Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_PY = Path("/home/chaos/proyectos/OpenaiBuildWeek/Kateto/kateto/voices/tools.py")
CORPUS_ROOT = Path("/run/media/chaos/terciario/proyectos/kateto-train")

registry_mod = import_module_from_path(
    "harness_tool_registry", REPO_ROOT / "scripts" / "harness_tool_registry.py"
)
builder_mod = import_module_from_path(
    "build_toolcall_harness_eval", REPO_ROOT / "scripts" / "build_toolcall_harness_eval.py"
)
eval_mod = import_module_from_path(
    "eval_harness_toolcall", REPO_ROOT / "scripts" / "eval_harness_toolcall.py"
)


def test_1_load_registry():
    """load_registry returns exactly the 18 expected tools and required parameters match samples."""
    reg = registry_mod.load_registry(str(TOOLS_PY))

    expected_names = {
        "run_command",
        "read_file",
        "write_file",
        "delete_file",
        "send_event",
        "list_events",
        "enable_plugin",
        "disable_plugin",
        "list_plugins",
        "create_skill",
        "update_skill",
        "create_workflow",
        "update_workflow",
        "update_soul",
        "request_generation",
        "schedule_event",
        "get_current_time",
        "refine_memory",
    }
    assert set(reg.keys()) == expected_names
    assert len(reg) == 18

    # Check 6 required parameter samples
    assert reg["write_file"]["required"] == ["path", "content"]
    assert reg["send_event"]["required"] == ["event_name", "data"]
    assert reg["request_generation"]["required"] == ["target_voice", "prompt"]
    assert reg["create_workflow"]["required"] == ["name", "voice", "content"]
    assert reg["schedule_event"]["required"] == ["event_name"]
    assert reg["list_events"]["required"] == []


def test_2_builder_determinism(tmp_path: Path):
    """Running builder twice outputs identical bytes and sha256."""
    sits = builder_mod.SITUATIONS
    reg = registry_mod.load_registry(str(TOOLS_PY))
    builder_mod.validate_integrity(sits, reg)

    out1 = tmp_path / "sits1.jsonl"
    out2 = tmp_path / "sits2.jsonl"

    import json
    with open(out1, "w", encoding="utf-8") as f:
        for s in sits:
            f.write(json.dumps(s, ensure_ascii=False, separators=(",", ":")) + "\n")

    with open(out2, "w", encoding="utf-8") as f:
        for s in sits:
            f.write(json.dumps(s, ensure_ascii=False, separators=(",", ":")) + "\n")

    bytes1 = out1.read_bytes()
    bytes2 = out2.read_bytes()
    assert bytes1 == bytes2
    sha1 = hashlib.sha256(bytes1).hexdigest()
    sha2 = hashlib.sha256(bytes2).hexdigest()
    assert sha1 == sha2


def test_3_set_integrity():
    """Set integrity: 36 items, 18 direct covering all tools, 10 traps, 8 no_tool, valid required."""
    sits = builder_mod.SITUATIONS
    reg = registry_mod.load_registry(str(TOOLS_PY))

    assert len(sits) == 36
    # Check IDs are hx-001..hx-036
    expected_ids = [f"hx-{i:03d}" for i in range(1, 37)]
    assert [s["id"] for s in sits] == expected_ids

    # 18 direct items covering all registry tools in order
    direct_tools = [s["expect"]["name"] for s in sits[:18]]
    assert direct_tools == list(reg.keys())

    # 10 trap items
    traps = [s for s in sits if s["trap"] is not None]
    assert len(traps) == 10

    # 8 no_tool items
    no_tools = [s for s in sits if s["kind"] == "no_tool"]
    assert len(no_tools) == 8

    # All required keys are subsets of registry required
    for s in sits:
        exp_name = s["expect"]["name"]
        exp_req = set(s["expect"].get("required", {}).keys())
        if exp_name is not None:
            assert exp_name in reg
            reg_req = set(reg[exp_name]["required"])
            assert exp_req.issubset(reg_req)
        else:
            assert s["kind"] == "no_tool"
            assert exp_req == set()


def test_4_guard_passes_real_corpus():
    """Anti-overlap guard passes on the real Kateto corpus."""
    guard_json_path = REPO_ROOT / "evals" / "toolcall_harness" / "guard.json"
    assert guard_json_path.exists(), "guard.json must exist"

    import json
    report = json.loads(guard_json_path.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["corpus_lines"] > 1000
    assert report["overlaps"] == []
    assert report["max_shingle_overlap"] < 0.30


def test_5_guard_detects_overlap(tmp_path: Path):
    """Anti-overlap guard detects exact overlap in a temporary corpus."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    fake_corpus_file = data_dir / "kateto_v8_5k.jsonl"

    target_sit = builder_mod.SITUATIONS[0]["situation"]
    # Write line containing target situation
    fake_corpus_file.write_text(
        f'{{"prompt": "Usuario dice: {target_sit}", "response": "ok"}}\n',
        encoding="utf-8",
    )

    report, ok = builder_mod.run_overlap_guard(builder_mod.SITUATIONS, str(tmp_path))
    assert ok is False
    assert len(report["overlaps"]) > 0
    overlap_ids = [ov["id"] for ov in report["overlaps"]]
    assert "hx-001" in overlap_ids


def test_6_scoring_exact():
    """Exact scoring fixture of 5 items: perfecta / nombre equivocado / nombre inventado / falta required / no_tool ok."""
    reg = registry_mod.load_registry(str(TOOLS_PY))

    fixture_situations = [
        {
            "id": "fix-001",
            "kind": "tool",
            "situation": "Corre git status.",
            "expect": {"name": "run_command", "required": {"command": "git status"}},
        },
        {
            "id": "fix-002",
            "kind": "tool",
            "situation": "Lee docs/plan.md.",
            "expect": {"name": "read_file", "required": {"path": "docs/plan.md"}},
        },
        {
            "id": "fix-003",
            "kind": "tool",
            "situation": "Guarda el reporte.",
            "expect": {"name": "write_file", "required": {"path": "report.txt", "content": {"contains": "ok"}}},
        },
        {
            "id": "fix-004",
            "kind": "tool",
            "situation": "Borra el archivo tmp.log.",
            "expect": {"name": "delete_file", "required": {"path": "tmp.log"}},
        },
        {
            "id": "fix-005",
            "kind": "no_tool",
            "situation": "Que opinas del clima?",
            "expect": {"name": None, "required": {}},
        },
    ]

    fixture_predictions = [
        # 1. Perfecta
        {
            "id": "fix-001",
            "tool_calls": [{"name": "run_command", "arguments": {"command": "git status"}}],
        },
        # 2. Nombre equivocado del registro (run_command en vez de read_file)
        {
            "id": "fix-002",
            "tool_calls": [{"name": "run_command", "arguments": {"command": "cat docs/plan.md"}}],
        },
        # 3. Nombre inventado (read_doc)
        {
            "id": "fix-003",
            "tool_calls": [{"name": "read_doc", "arguments": {"path": "report.txt"}}],
        },
        # 4. Falta required (delete_file sin 'path')
        {
            "id": "fix-004",
            "tool_calls": [{"name": "delete_file", "arguments": {}}],
        },
        # 5. no_tool ok
        {
            "id": "fix-005",
            "output": "Hoy esta soleado y templado.",
        },
    ]

    report = eval_mod.score_predictions(
        fixture_situations, fixture_predictions, reg, backend="predictions"
    )

    assert report["n"] == 5
    assert report["tool_n"] == 4
    assert report["no_tool_n"] == 1
    assert report["name_ok"] == 3  # fix-001, fix-004, fix-005
    assert report["full_ok"] == 2  # fix-001, fix-005
    assert report["no_tool_ok"] == 1  # fix-005
    assert report["parse_errors"] == 0
    assert len(report["hallucinated"]) == 1
    assert report["hallucinated"][0]["name"] == "read_doc"
    assert report["hallucinated_rate"] == 0.25  # 1 hallucinated out of 4 calls


def test_7_scoring_partial_predictions():
    """Partial predictions for 2 ids (1 tool correct, 1 no_tool calling a tool) on the 36 situations."""
    reg = registry_mod.load_registry(str(TOOLS_PY))
    situations = eval_mod.load_situations(
        str(REPO_ROOT / "evals" / "toolcall_harness" / "situations.jsonl")
    )
    preds = [
        {
            "id": "hx-001",
            "tool_calls": [{"name": "run_command", "arguments": {"command": "git status"}}],
        },
        {
            "id": "hx-030",
            "tool_calls": [{"name": "get_current_time", "arguments": {}}],
        },
    ]
    report = eval_mod.score_predictions(situations, preds, reg, backend="predictions")
    assert report["missing"] == 34
    assert report["no_tool_ok"] == 0
    assert report["name_ok"] == 1


def test_8_scoring_empty_predictions(tmp_path: Path):
    """Empty predictions file -> parse_errors == tool_n (28), no_tool_ok == 0, missing == 36, toolcall_accuracy == 0.0."""
    import json

    reg = registry_mod.load_registry(str(TOOLS_PY))
    situations = eval_mod.load_situations(
        str(REPO_ROOT / "evals" / "toolcall_harness" / "situations.jsonl")
    )
    empty_preds_file = tmp_path / "empty_preds.jsonl"
    empty_preds_file.write_text("", encoding="utf-8")

    preds = []
    with open(empty_preds_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))

    report = eval_mod.score_predictions(situations, preds, reg, backend="predictions")
    assert report["parse_errors"] == report["tool_n"]
    assert report["tool_n"] == 28
    assert report["no_tool_ok"] == 0
    assert report["missing"] == 36
    assert report["toolcall_accuracy"] == 0.0


import http.server
import json
import threading
from contextlib import contextmanager


class _MockHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        payload = json.loads(body.decode("utf-8")) if body else {}
        self.server.requests.append({"path": self.path, "payload": payload})

        resp = self.server.response_fn(self.path, payload)
        data = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        resp = json.dumps({"data": [{"id": "kateto"}]}).encode("utf-8")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


@contextmanager
def run_mock_server(response_fn=None):
    if response_fn is None:
        def response_fn(path, payload):
            if "chat" in path:
                return {
                    "choices": [{
                        "message": {"role": "assistant", "content": "mock text", "tool_calls": None},
                        "finish_reason": "stop",
                    }],
                    "usage": {"completion_tokens": 10},
                }
            else:
                return {
                    "choices": [{
                        "text": "mock text",
                        "finish_reason": "stop",
                    }],
                    "usage": {"completion_tokens": 10},
                }

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MockHandler)
    server.requests = []
    server.response_fn = response_fn
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    try:
        yield server, base_url
    finally:
        server.shutdown()
        server.server_close()


def test_10_raw_prompt_es_el_wrapper_del_dataset():
    assert (
        eval_mod.build_raw_prompt("Corré git status.")
        == "<|im_user|>Corré git status.<|im_end|>\n<|im_start|>seco\n"
    )


def test_11_transport_chat_sigue_pegandole_a_chat_completions():
    reg = registry_mod.load_registry(str(TOOLS_PY))
    sit = {"id": "hx-001", "situation": "Corré git status.", "kind": "tool"}
    with run_mock_server() as (server, base_url):
        eval_mod.run_http_eval(
            [sit],
            reg,
            base_url,
            model="kateto",
            limit=1,
            temperature=0.0,
            timeout=5.0,
            transport="chat",
            stop_mode="default",
        )
        assert len(server.requests) == 1
        req = server.requests[0]
        assert req["path"] == "/v1/chat/completions"
        assert "messages" in req["payload"]
        assert "tools" in req["payload"]
        assert "stop" not in req["payload"]
        assert "repeat_penalty" not in req["payload"]


def test_12_transport_raw_pega_a_completions_sin_tools():
    reg = registry_mod.load_registry(str(TOOLS_PY))
    sit = {"id": "hx-001", "situation": "Corré git status.", "kind": "tool"}
    with run_mock_server() as (server, base_url):
        eval_mod.run_http_eval(
            [sit],
            reg,
            base_url,
            model="kateto",
            limit=1,
            temperature=0.0,
            timeout=5.0,
            transport="raw",
            stop_mode="default",
            max_tokens=256,
        )
        assert len(server.requests) == 1
        req = server.requests[0]
        assert req["path"] == "/v1/completions"
        assert req["payload"]["prompt"] == eval_mod.build_raw_prompt(sit["situation"])
        assert "tools" not in req["payload"]
        assert "messages" not in req["payload"]
        assert req["payload"]["n_predict"] == 256


def test_13_raw_extrae_tool_call_del_texto():
    reg = registry_mod.load_registry(str(TOOLS_PY))
    sit = {
        "id": "hx-001",
        "kind": "tool",
        "situation": "Corre git status.",
        "expect": {"name": "run_command", "required": {"command": "git status"}},
    }
    raw_text = '<tool_call>\n{"name": "run_command", "arguments": "{\\"command\\": \\"git status\\"}"}\n</tool_call>'

    def resp_fn(path, payload):
        return {
            "choices": [{"text": raw_text, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 15},
        }

    with run_mock_server(resp_fn) as (server, base_url):
        preds = eval_mod.run_http_eval(
            [sit],
            reg,
            base_url,
            model="kateto",
            limit=1,
            temperature=0.0,
            timeout=5.0,
            transport="raw",
        )
        report = eval_mod.score_predictions([sit], preds, reg, backend="http")
        extra = eval_mod.compute_extra_metrics([sit], preds, reg)
        assert report["toolcall_accuracy"] > 0
        assert extra["n_llamadas_tools"] >= 1
        assert report["parse_errors"] == 0


def test_14_stop_mode_im_end_only():
    reg = registry_mod.load_registry(str(TOOLS_PY))
    sit = {"id": "hx-001", "situation": "Corré git status.", "kind": "tool"}
    with run_mock_server() as (server, base_url):
        # 1. chat with im_end_only
        eval_mod.run_http_eval(
            [sit],
            reg,
            base_url,
            model="kateto",
            limit=1,
            temperature=0.0,
            timeout=5.0,
            transport="chat",
            stop_mode="im_end_only",
        )
        req_chat_im = server.requests[-1]
        assert req_chat_im["payload"]["stop"] == ["<|im_end|>"]
        assert req_chat_im["payload"]["repeat_penalty"] == 1.0

        # 2. raw with im_end_only
        eval_mod.run_http_eval(
            [sit],
            reg,
            base_url,
            model="kateto",
            limit=1,
            temperature=0.0,
            timeout=5.0,
            transport="raw",
            stop_mode="im_end_only",
        )
        req_raw_im = server.requests[-1]
        assert req_raw_im["payload"]["stop"] == ["<|im_end|>"]
        assert req_raw_im["payload"]["repeat_penalty"] == 1.0

        # 3. chat with default
        eval_mod.run_http_eval(
            [sit],
            reg,
            base_url,
            model="kateto",
            limit=1,
            temperature=0.0,
            timeout=5.0,
            transport="chat",
            stop_mode="default",
        )
        req_chat_def = server.requests[-1]
        assert "stop" not in req_chat_def["payload"]
        assert "repeat_penalty" not in req_chat_def["payload"]

        # 4. raw with default
        eval_mod.run_http_eval(
            [sit],
            reg,
            base_url,
            model="kateto",
            limit=1,
            temperature=0.0,
            timeout=5.0,
            transport="raw",
            stop_mode="default",
        )
        req_raw_def = server.requests[-1]
        assert "stop" not in req_raw_def["payload"]
        assert "repeat_penalty" not in req_raw_def["payload"]


def test_15_situaciones_sha256_igual_entre_transportes():
    reg = registry_mod.load_registry(str(TOOLS_PY))
    situations = eval_mod.load_situations(
        str(REPO_ROOT / "evals" / "toolcall_harness" / "situations.jsonl")
    )
    with run_mock_server() as (server, base_url):
        preds_chat = eval_mod.run_http_eval(
            situations,
            reg,
            base_url,
            model="kateto",
            limit=None,
            temperature=0.0,
            timeout=5.0,
            transport="chat",
        )
        preds_raw = eval_mod.run_http_eval(
            situations,
            reg,
            base_url,
            model="kateto",
            limit=None,
            temperature=0.0,
            timeout=5.0,
            transport="raw",
        )
        report_chat = eval_mod.build_report(
            situations,
            preds_chat,
            reg,
            backend="http",
            transport="chat",
            base_url=base_url,
        )
        report_raw = eval_mod.build_report(
            situations,
            preds_raw,
            reg,
            backend="http",
            transport="raw",
            base_url=base_url,
        )
        assert report_chat["situaciones_sha256"] == report_raw["situaciones_sha256"]
        assert report_chat["request_sha256"] != report_raw["request_sha256"]


def test_16_dump_predictions_escribe_36_lineas(tmp_path: Path):
    reg = registry_mod.load_registry(str(TOOLS_PY))
    situations = eval_mod.load_situations(
        str(REPO_ROOT / "evals" / "toolcall_harness" / "situations.jsonl")
    )
    dump_file = tmp_path / "dump.jsonl"
    with run_mock_server() as (server, base_url):
        eval_mod.run_http_eval(
            situations,
            reg,
            base_url,
            model="kateto",
            limit=None,
            temperature=0.0,
            timeout=5.0,
            transport="chat",
            dump_predictions=dump_file,
        )
    assert dump_file.exists()
    lines = [
        line.strip()
        for line in dump_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 36
    expected_keys = {
        "id",
        "kind",
        "output",
        "tool_calls",
        "finish_reason",
        "tokens",
        "prompt_sha256",
    }
    for line in lines:
        data = json.loads(line)
        assert set(data.keys()) == expected_keys
        assert isinstance(data["prompt_sha256"], str) and len(data["prompt_sha256"]) > 0


def test_17_marcadores_distinguen_sin_cerrar():
    reg = registry_mod.load_registry(str(TOOLS_PY))
    sit = {
        "id": "hx-001",
        "kind": "tool",
        "situation": "Corre git status.",
        "expect": {"name": "run_command", "required": {"command": "git status"}},
    }
    unclosed_text = '<tool_call>\n{"name": "run_command", "arguments": {"command": "git status"'

    def resp_fn(path, payload):
        return {
            "choices": [{"text": unclosed_text, "finish_reason": "length"}],
            "usage": {"completion_tokens": 256},
        }

    with run_mock_server(resp_fn) as (server, base_url):
        preds = eval_mod.run_http_eval(
            [sit],
            reg,
            base_url,
            model="kateto",
            limit=1,
            temperature=0.0,
            timeout=5.0,
            transport="raw",
        )
        report = eval_mod.build_report(
            [sit],
            preds,
            reg,
            backend="http",
            transport="raw",
            base_url=base_url,
        )
        assert report["marcadores"]["con_tool_call_sin_cerrar"] == 1
        assert report["n_truncados"] == 1
