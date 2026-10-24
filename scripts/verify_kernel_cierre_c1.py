#!/usr/bin/env python3
"""Verificación y validación de contrato para el kernel Kaggle A/B de cierre few-shot.

Especificación: Brief t_7cbaf7c5 (SPEC 3).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rwkv_pipeline.chat_template import render_chat, render_historial


def calcular_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def normalizar(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Verificación del kernel eval29_cierre_c1")
    parser.add_argument(
        "--prefix",
        type=Path,
        default=_REPO_ROOT / "kaggle/kernels/eval29_cierre_c1/prefix.json",
        help="Ruta a prefix.json",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=_REPO_ROOT / "kaggle/kernels/eval29_cierre_c1/reference-v7-as-run.ipynb",
        help="Ruta a reference-v7-as-run.ipynb",
    )
    parser.add_argument(
        "--v7-orig",
        type=Path,
        default=Path("/tmp/kpull-v7/kateto-eval29-v7-2xt4.ipynb"),
        help="Ruta al notebook real v7 original",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=_REPO_ROOT / "kaggle/kernels/eval29_cierre_c1/eval29_cierre_c1.ipynb",
        help="Ruta al notebook generado",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=_REPO_ROOT / "kaggle/kernels/eval29_cierre_c1/kernel-metadata.json",
        help="Ruta a kernel-metadata.json",
    )
    parser.add_argument(
        "--eval-prompts",
        type=Path,
        default=Path("/home/chaos/.hermes/profiles/runner/cache/scratch/eval29-ngram/ds/data/eval_prompts.jsonl"),
        help="Ruta a eval_prompts.jsonl",
    )
    parser.add_argument(
        "--eval-v1",
        type=Path,
        default=Path("/run/media/chaos/terciario/proyectos/kateto-train/out/eval/prompts_v1.json"),
        help="Ruta a prompts_v1.json",
    )
    args = parser.parse_args()

    # Cargar los 60 prompts de evaluacion
    assert args.eval_v1.exists(), f"Falta {args.eval_v1}"
    assert args.eval_prompts.exists(), f"Falta {args.eval_prompts}"
    eval_prompts = []
    v1_data = json.loads(args.eval_v1.read_text(encoding="utf-8"))
    for it in v1_data["generation"][:30]:
        eval_prompts.append(it["prompt"])
    for line in args.eval_prompts.read_text(encoding="utf-8").splitlines():
        if line.strip():
            eval_prompts.append(json.loads(line)["prompt"])
    assert len(eval_prompts) == 60, f"Se esperaban 60 prompts de evaluacion, obtenidos {len(eval_prompts)}"

    # 1. prefix.json
    assert args.prefix.exists(), f"Falta {args.prefix}"
    prefix_data = json.loads(args.prefix.read_text(encoding="utf-8"))
    prefijo_texto = prefix_data["prefijo_texto"]
    assert prefix_data["n_chars_prefijo"] == len(prefijo_texto), "n_chars_prefijo no coincide"
    assert prefix_data["sha256_prefijo"] == hashlib.sha256(prefijo_texto.encode("utf-8")).hexdigest(), (
        "sha256_prefijo incorrecto"
    )
    assert prefijo_texto.count("<|im_end|>") == 4, "Debe tener 4 <|im_end|>"
    assert prefijo_texto.count("<|im_user|>") == 2, "Debe tener 2 <|im_user|>"
    assert prefijo_texto.count("<|im_start|>") == 2, "Debe tener 2 <|im_start|>"
    turnos = prefix_data["turnos"]
    assert len(turnos) == 2, f"Se esperaban 2 turnos, obtenidos {len(turnos)}"
    for t in turnos:
        assert "<|" not in t[0] and "<|" not in t[1], f"Turno contiene '<|': {t}"
    hist = [tuple(t) for t in turnos]
    assert render_historial(hist, "seco") == prefijo_texto, "render_historial no coincide con prefijo_texto"

    prefijo_norm = normalizar(prefijo_texto)
    for ep in eval_prompts:
        ep_norm = normalizar(ep)
        assert ep_norm not in prefijo_norm, f"Prompt de evaluacion aparece en prefijo: {ep}"
        for t in turnos:
            u_norm = normalizar(t[0])
            assert u_norm != ep_norm, f"Turno coincide con prompt: {t[0]}"
            assert u_norm not in ep_norm, f"Turno es substring de prompt: {t[0]}"
            assert ep_norm not in u_norm, f"Turno es superstring de prompt: {t[0]}"
    print("Check 1: prefix.json valido (2 turnos, marcadores 4/2/2, exclusion estricta)")

    # 2. reference-v7-as-run.ipynb
    assert args.ref.exists(), f"Falta {args.ref}"
    assert args.v7_orig.exists(), f"Falta {args.v7_orig}"
    ref_sha = calcular_sha256(args.ref)
    orig_sha = calcular_sha256(args.v7_orig)
    assert ref_sha == orig_sha, f"SHA256 no coincide: {ref_sha} vs {orig_sha}"
    ref_nb = json.loads(args.ref.read_text(encoding="utf-8"))
    assert len(ref_nb["cells"]) == 8, f"reference debe tener 8 celdas, tiene {len(ref_nb['cells'])}"
    print("Check 2: reference-v7-as-run.ipynb identico byte-exacto a v7 (8 celdas)")

    # 3. Notebook generado
    assert args.notebook.exists(), f"Falta {args.notebook}"
    gen_nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    assert "nbformat" in gen_nb, "Falta nbformat"
    assert "kernelspec" in gen_nb.get("metadata", {}), "Falta kernelspec en metadata"
    assert len(gen_nb["cells"]) == 8, f"Notebook debe tener 8 celdas, tiene {len(gen_nb['cells'])}"
    assert gen_nb["cells"][0]["cell_type"] == "markdown", "Celda 0 debe ser markdown"
    for idx in range(1, 8):
        c = gen_nb["cells"][idx]
        assert c["cell_type"] == "code", f"Celda {idx} debe ser code"
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        ast.parse(src)
    for idx in range(1, 6):
        assert gen_nb["cells"][idx]["source"] == ref_nb["cells"][idx]["source"], (
            f"Celda {idx} difiere de reference"
        )
    print("Check 3: Notebook generado valido (8 celdas, ast.parse OK, celdas 1..5 identicas a reference)")

    # 4. Invariante brazo B
    for p in eval_prompts:
        chat_b = render_chat(hist, p, "seco")
        chat_a = render_chat([], p, "seco")
        assert chat_b == prefijo_texto + chat_a, f"Invariante brazo B violado para prompt: {p!r}"
        assert chat_a == f"<|im_user|>{p.strip()}<|im_end|>\n<|im_start|>seco\n", (
            f"render_chat([], p, 'seco') no coincide con formato canonico v7"
        )
    print("Check 4: Invariante brazo B verificado sobre los 60 prompts de evaluacion")

    # 5. Invariante brazo A
    c6_src = gen_nb["cells"][6]["source"]
    c6_str = c6_src if isinstance(c6_src, str) else "".join(c6_src)
    assert 'render_chat([], p, "seco")' in c6_str, "Falta llamada render_chat([], p, 'seco') en celda 6"
    assert 'render_chat(HIST, p, "seco")' in c6_str, "Falta llamada render_chat(HIST, p, 'seco') en celda 6"

    # Verificar función fmt
    ast_c6 = ast.parse(c6_str)
    fmt_def = next((n for n in ast.walk(ast_c6) if isinstance(n, ast.FunctionDef) and n.name == "fmt"), None)
    assert fmt_def is not None, "No se encontro la funcion fmt en celda 6"

    # Validar ejecucion de fmt en namespace controlado
    ns = {
        "HIST": hist,
        "render_chat": render_chat,
    }
    for stmt in ast_c6.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "fmt":
            exec(compile(ast.Module(body=[stmt], type_ignores=[]), filename="<cell>", mode="exec"), ns)
            break
    fmt_fn = ns.get("fmt")
    assert fmt_fn is not None
    test_p = "test prompt"
    assert fmt_fn(test_p, "A") == render_chat([], test_p, "seco"), "fmt(p, 'A') no coincide con render_chat([], p, 'seco')"
    assert fmt_fn(test_p, "B") == render_chat(hist, test_p, "seco"), "fmt(p, 'B') no coincide con render_chat(HIST, p, 'seco')"
    print("Check 5: Invariante brazo A verificado (fmt llama render_chat([], p, 'seco') y nada mas)")

    # 6. kernel-metadata.json
    assert args.metadata.exists(), f"Falta {args.metadata}"
    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    assert meta.get("id") == "gabrielsolotorevsky/kateto-eval29-cierre-c1-2xt4", f"id incorrecto: {meta.get('id')}"
    assert meta.get("code_file") == "eval29_cierre_c1.ipynb", f"code_file incorrecto: {meta.get('code_file')}"
    assert meta.get("machine_shape") == "NvidiaTeslaT4x2", f"machine_shape incorrecto: {meta.get('machine_shape')}"
    assert meta.get("dataset_sources") == [
        "gabrielsolotorevsky/kateto-eval29",
        "gabrielsolotorevsky/kateto-eval29-v7-code",
    ], f"dataset_sources incorrecto: {meta.get('dataset_sources')}"
    print("Check 6: kernel-metadata.json verificado (id, code_file, T4x2, 2 datasets exactos)")

    # 7. Restricciones y limpieza
    for idx, cell in enumerate(gen_nb["cells"]):
        src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        assert "/run/media/chaos" not in src, f"Celda {idx} contiene ruta absoluta /run/media/chaos"
        assert "kateto-eval29-ngram" not in src, f"Celda {idx} contiene kateto-eval29-ngram"
        for line in src.splitlines():
            line_s = line.strip()
            if line_s.startswith("!") and "kaggle" in line_s:
                raise AssertionError(f"Celda {idx} contiene comando CLI kaggle: {line}")
            if line_s.startswith("import kaggle") or line_s.startswith("from kaggle"):
                raise AssertionError(f"Celda {idx} importa kaggle: {line}")
        # Celdas PMI / behavior excluidas
        assert "pmi29_chatml" not in src, f"Celda {idx} contiene pmi29_chatml"
        assert "pmi29_world" not in src, f"Celda {idx} contiene pmi29_world"
        assert "behavior29" not in src, f"Celda {idx} contiene behavior29"
        assert "probes.json" not in src, f"Celda {idx} contiene probes.json"

    # Verificar que celdas nuevas (0, 6, 7) no contengan 'kaggle'
    for idx in (0, 6, 7):
        src = gen_nb["cells"][idx]["source"] if isinstance(gen_nb["cells"][idx]["source"], str) else "".join(gen_nb["cells"][idx]["source"])
        assert "kaggle" not in src.lower(), f"Celda {idx} contiene palabra 'kaggle'"

    print("Check 7: Restricciones verificadas (sin /run/media/chaos, sin kateto-eval29-ngram, sin PMI ni behavior)")
    print("VERIFY OK")


if __name__ == "__main__":
    main()
