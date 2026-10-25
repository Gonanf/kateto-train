"""Arma data/toolcalling_kateto_v2.jsonl concatenando data/toolcalling_v2/*.jsonl.
Solo rebusca archivos cambiados segun manifest, asi Minecraft se toca sin regenerar todo.
Uso: python3 scripts/build_v2_split.py [--force]
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

TDIR = Path(__file__).resolve().parent.parent
SDIR = TDIR / "data" / "toolcalling_v2"
DST = TDIR / "data" / "toolcalling_kateto_v2.jsonl"
MANIFEST = SDIR / ".manifest.json"


def file_hash(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main(force: bool = False) -> None:
    manifest: dict[str, str] = {}
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    changed = []
    for p in sorted(SDIR.glob("*.jsonl")):
        h = file_hash(p)
        if force or manifest.get(p.name) != h:
            changed.append(p.name)
        manifest[p.name] = h
    rows = []
    for p in sorted(SDIR.glob("*.jsonl")):
        for line in p.open(encoding="utf-8"):
            if line.strip():
                rows.append(line)
    DST.write_text("".join(rows), encoding="utf-8")
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"archivos={len(list(SDIR.glob('*.jsonl')))} filas={len(rows)} cambiados={changed} -> {DST}")


if __name__ == "__main__":
    main("--force" in sys.argv)
