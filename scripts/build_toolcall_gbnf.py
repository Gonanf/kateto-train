#!/usr/bin/env python3
"""Deterministic generator for toolcall.gbnf grammar."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Allow importing registry from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_tool_registry import load_registry


def generate_gbnf(tools_py: str) -> str:
    """Build GBNF grammar string deterministically from tools registry."""
    registry = load_registry(tools_py)
    tool_names = sorted(registry.keys())
    nombre_alternatives = " | ".join(f'"\\"{name}\\""' for name in tool_names)

    lines = [
        "# toolcall.gbnf — GENERADO por scripts/build_toolcall_gbnf.py desde harness_tool_registry. No editar a mano.",
        "root ::= ws cuerpo ws",
        "cuerpo ::= tool-call | json-plano | texto",
        'tool-call ::= "<tool_call>" ws "{" ws "\\"name\\"" ws ":" ws nombre ws "," ws "\\"arguments\\"" ws ":" ws objeto ws "}" ws "</tool_call>"',
        'json-plano ::= "{" ws "\\"name\\"" ws ":" ws nombre ws "," ws "\\"arguments\\"" ws ":" ws objeto ws "}"',
        f"nombre ::= {nombre_alternatives}",
        'objeto ::= "{" ws "}" | "{" ws par (ws "," ws par)* ws "}"',
        'par ::= cadena ws ":" ws valor',
        'valor ::= cadena | numero | "true" | "false" | "null" | objeto | lista',
        'lista ::= "[" ws "]" | "[" ws valor (ws "," ws valor)* ws "]"',
        'cadena ::= "\\"" str-char* "\\""',
        r'str-char ::= [^"\\\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{4} )',
        'numero ::= "-"? ("0" | [1-9] [0-9]{0,15}) ("." [0-9]{1,15})? ([eE] [-+]? [0-9]{1,3})?',
        "texto ::= texto-char{8,400}",
        'texto-char ::= [a-zA-Z0-9 .,;:!?\'"()\\-]',
        "ws ::= [ \\t\\n]{0,3}",
    ]
    return "\n".join(lines) + "\n"


build_grammar = generate_gbnf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate toolcall.gbnf grammar deterministically."
    )
    parser.add_argument(
        "--out",
        default="scripts/toolcall.gbnf",
        help="Output path for the generated GBNF file (default: scripts/toolcall.gbnf)",
    )
    parser.add_argument(
        "--tools-py",
        default="/home/chaos/proyectos/OpenaiBuildWeek/Kateto/kateto/voices/tools.py",
        help="Path to voices/tools.py",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if generated grammar matches file on disk without writing",
    )

    args = parser.parse_args()

    content = generate_gbnf(args.tools_py)
    gen_bytes = content.encode("utf-8")
    gen_sha = hashlib.sha256(gen_bytes).hexdigest()

    out_path = Path(args.out)

    if args.check:
        if not out_path.exists():
            print("FALTA")
            sys.exit(1)
        disk_bytes = out_path.read_bytes()
        if disk_bytes == gen_bytes:
            print(f"sha256={gen_sha}")
            sys.exit(0)
        else:
            print("DESACTUALIZADO")
            sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(gen_bytes)
    print(f"sha256={gen_sha}")
    sys.exit(0)


if __name__ == "__main__":
    main()
