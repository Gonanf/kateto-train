#!/usr/bin/env python3
"""
scripts/auditoria-arq.py — Auditoría de modelos locales, soporte de motor y candidatos HF
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import time
import urllib.error
import urllib.request

LOCAL_MODELS = [
    ("rwkv15", "/run/media/chaos/terciario/proyectos/kateto-train/out/kateto-15b-Q8_0.gguf"),
    ("rwkv29", "/run/media/chaos/terciario/proyectos/kateto-train/out/kateto-rwkv29-Q8_0.gguf"),
    ("mamba14", "/run/media/chaos/terciario/chaos-cache/mamba-bench/mamba-1.4b-hf.Q8_0.gguf"),
    ("mamba14inst", "/run/media/chaos/terciario/chaos-cache/mamba-bench/mamba-1.4b-instruct-hf.Q8_0.gguf"),
    ("mamba28", "/run/media/chaos/terciario/chaos-cache/mamba-bench/mamba-2.8b-hf-Q8_0.gguf"),
]

BIN_PATH = "/home/chaos/.local/bin/llama-bench"
USER_AGENT = "hermes-t_0240ba88"


def get_binary_info():
    build_str = "b32-f2cc728"
    try:
        res = subprocess.run([BIN_PATH, "--version"], capture_output=True, text=True, timeout=10)
        out = (res.stdout + "\n" + res.stderr).strip()
        m = re.search(r"build (\d+),\s*commit ([0-9a-f]+)", out)
        if m:
            build_str = f"b{m.group(1)}-{m.group(2)}"
    except Exception:
        pass
    return BIN_PATH, build_str


def find_libllama_so(bin_path):
    candidates = [
        Path(bin_path).parent / "libllama.so",
        Path(os.path.realpath(bin_path)).parent / "libllama.so",
        Path("/run/media/chaos/secundario/llamacpp-conv/build/bin/libllama.so"),
        Path("/home/chaos/.local/lib/libllama.so"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def check_engine_support(bin_path):
    target_archs = ["mamba", "mamba2", "jamba", "nemotron_h", "falcon", "rwkv7", "zamba2"]
    support = {a: False for a in target_archs}
    lib_path = find_libllama_so(bin_path)
    if lib_path:
        try:
            res = subprocess.run(["strings", lib_path], capture_output=True, text=True, timeout=15)
            content = res.stdout
            lines = set(content.splitlines())
            for a in target_archs:
                if a in lines or a in content:
                    support[a] = True
        except Exception:
            pass
    return support


def compute_sha256_extremos(filepath):
    p = Path(filepath)
    if not p.is_file():
        return None
    size = p.stat().st_size
    with open(filepath, "rb") as f:
        if size <= 2 * 1024 * 1024:
            data = f.read()
        else:
            head = f.read(1024 * 1024)
            f.seek(size - 1024 * 1024)
            tail = f.read(1024 * 1024)
            data = head + tail
    return hashlib.sha256(data).hexdigest()


def parse_gguf_metadata(filepath):
    p = Path(filepath)
    if not p.is_file():
        return {}
    with open(filepath, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            return {}
        (version,) = struct.unpack("<I", f.read(4))
        tensor_count, kv_count = struct.unpack("<QQ", f.read(16))
        kv = {}

        def read_val(vt):
            if vt == 0:
                return struct.unpack("<B", f.read(1))[0]
            elif vt == 1:
                return struct.unpack("<b", f.read(1))[0]
            elif vt == 2:
                return struct.unpack("<H", f.read(2))[0]
            elif vt == 3:
                return struct.unpack("<h", f.read(2))[0]
            elif vt == 4:
                return struct.unpack("<I", f.read(4))[0]
            elif vt == 5:
                return struct.unpack("<i", f.read(4))[0]
            elif vt == 6:
                return struct.unpack("<f", f.read(4))[0]
            elif vt == 7:
                return struct.unpack("<?", f.read(1))[0]
            elif vt == 8:
                (slen,) = struct.unpack("<Q", f.read(8))
                return f.read(slen).decode("utf-8", errors="replace")
            elif vt == 9:
                elem_type, arr_len = struct.unpack("<IQ", f.read(12))
                return [read_val(elem_type) for _ in range(arr_len)]
            elif vt == 10:
                return struct.unpack("<Q", f.read(8))[0]
            elif vt == 11:
                return struct.unpack("<q", f.read(8))[0]
            elif vt == 12:
                return struct.unpack("<d", f.read(8))[0]
            else:
                raise ValueError(f"Unknown GGUF type {vt}")

        for _ in range(kv_count):
            (klen,) = struct.unpack("<Q", f.read(8))
            key = f.read(klen).decode("utf-8", errors="replace")
            (vtype,) = struct.unpack("<I", f.read(4))
            val = read_val(vtype)
            kv[key] = val
        return kv


def http_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data)
        except Exception as e:
            if attempt == 2:
                return {"error": str(e)}
            time.sleep(2)
    return {"error": "Timeout o falla de red"}


def inspect_hf_candidates():
    queries = [
        "mamba-130m",
        "mamba-370m",
        "mamba-790m",
        "mamba-1.4b",
        "mamba-2.8b",
        "mamba2",
        "falcon-mamba",
        "zamba2",
        "codestral-mamba",
        "jamba",
        "nemotron_h",
    ]

    candidatos = []
    seen_repos = set()

    for q in queries:
        time.sleep(1.5)
        search_url = f"https://huggingface.co/api/models?search={q}&limit=100&expand[]=downloads"
        res = http_get_json(search_url)
        if isinstance(res, dict) and "error" in res:
            continue
        if not isinstance(res, list):
            continue

        # Filter repos with gguf in name or siblings
        matching = []
        for item in res:
            rid = item.get("id", "")
            if "gguf" in rid.lower() or any(
                s.get("rfilename", "").endswith(".gguf") for s in item.get("siblings", [])
            ):
                matching.append(item)

        # Sort by downloads descending
        matching.sort(key=lambda x: x.get("downloads", 0), reverse=True)

        # Pick top 3 unique repos per family
        picked = 0
        for item in matching:
            rid = item.get("id")
            if not rid or rid in seen_repos:
                continue
            seen_repos.add(rid)

            time.sleep(1.5)
            detail_url = f"https://huggingface.co/api/models/{rid}?blobs=true"
            detail = http_get_json(detail_url)
            if isinstance(detail, dict) and "error" in detail:
                continue

            siblings = detail.get("siblings", [])
            gguf_files = []
            for s in siblings:
                fname = s.get("rfilename", "")
                if fname.endswith(".gguf"):
                    gguf_files.append({
                        "file": fname,
                        "bytes": s.get("size", 0),
                    })

            is_instruct = bool(
                re.search(r"(instruct|chat|sft|dpo|orpo)", rid, re.IGNORECASE)
            )

            candidatos.append({
                "familia": q,
                "repo": rid,
                "instruct": is_instruct,
                "gguf": gguf_files,
                "downloads": detail.get("downloads", item.get("downloads", 0)),
                "lastModified": detail.get("lastModified", ""),
                "pipeline_tag": detail.get("pipeline_tag"),
                "fuente": f"https://huggingface.co/api/models/{rid}",
            })

            picked += 1
            if picked >= 3:
                break

    return candidatos


def main():
    parser = argparse.ArgumentParser(description="Auditoría de arquitecturas locales y candidatos HF")
    parser.add_argument("--out", default="./auditoria-arq.json", help="Ruta del JSON de salida")
    args = parser.parse_args()

    bin_path, build_str = get_binary_info()
    archs_soportados = check_engine_support(bin_path)

    modelos_locales = []
    for m_id, gguf_path in LOCAL_MODELS:
        p = Path(gguf_path)
        if not p.is_file():
            modelos_locales.append({
                "id": m_id,
                "gguf": gguf_path,
                "existe": False,
                "error": "Archivo no encontrado",
            })
            continue

        size_bytes = p.stat().st_size
        sha_ext = compute_sha256_extremos(gguf_path)
        kv = parse_gguf_metadata(gguf_path)

        arch = kv.get("general.architecture", "")
        name = kv.get("general.name", p.name)
        size_label = kv.get("general.size_label", "")
        block_count = kv.get(f"{arch}.block_count") or kv.get("block_count")
        embedding_length = kv.get(f"{arch}.embedding_length") or kv.get("embedding_length")
        file_type = kv.get("general.file_type")

        modelos_locales.append({
            "id": m_id,
            "gguf": gguf_path,
            "bytes": size_bytes,
            "sha256_extremos": sha_ext,
            "arch": arch,
            "name": name,
            "size_label": size_label,
            "block_count": block_count,
            "embedding_length": embedding_length,
            "file_type": file_type,
        })

    candidatos = inspect_hf_candidates()

    out_data = {
        "fecha": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "engine": {
            "bin": bin_path,
            "build": build_str,
            "archs_soportados": archs_soportados,
        },
        "modelos_locales": modelos_locales,
        "candidatos": candidatos,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
