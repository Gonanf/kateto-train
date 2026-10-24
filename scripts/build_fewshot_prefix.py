#!/usr/bin/env python3
"""Genera el prefijo few-shot de 2 turnos cerrados a partir del set de entrenamiento.

Especificación: Brief t_7cbaf7c5 (SPEC 1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Raíz del repo en sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rwkv_pipeline.chat_template import render_historial, render_turn, slot_guard
from scripts.analyze_eval29_coherence import check_degenerate

RX_TURNO = re.compile(
    r"^<\|im_user\|>(?P<u>.+?)<\|im_end\|>\n<\|im_start\|>seco\n(?P<a>.+?)<\|im_end\|>$",
    re.S,
)
RX_PALABRA = re.compile(r"[^\W\d_]+")


def normalizar_texto(s: str) -> str:
    """Normaliza colapsando espacios y pasando a minúsculas."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def calcular_sha256_archivo(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def cargar_prompts_eval(path_ep: Path, path_v1: Path) -> list[str]:
    prompts = []
    # 30 prompts de eval_prompts.jsonl
    for line in path_ep.read_text(encoding="utf-8").splitlines():
        if line.strip():
            prompts.append(json.loads(line)["prompt"])
    assert len(prompts) == 30, f"Se esperaban 30 prompts en {path_ep}, hay {len(prompts)}"

    # 30 prompts de prompts_v1.json ["generation"][:30]
    v1_data = json.loads(path_v1.read_text(encoding="utf-8"))
    for it in v1_data["generation"][:30]:
        prompts.append(it["prompt"])

    assert len(prompts) == 60, f"Se esperaban 60 prompts de evaluacion, hay {len(prompts)}"
    return prompts


def rel_path(p: Path, standard_rel: str) -> str:
    """Devuelve la ruta canónica relativa para no filtrar rutas del host a Kaggle."""
    s = str(p)
    if standard_rel in s:
        return standard_rel
    return p.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir prefijo few-shot de 2 turnos cerrados")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("/run/media/chaos/terciario/proyectos/kateto-train/data/rwkv_kateto_seco.jsonl"),
        help="Ruta al archivo rwkv_kateto_seco.jsonl de entrenamiento",
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
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "kaggle/kernels/eval29_cierre_c1/prefix.json",
        help="Ruta del prefix.json de salida",
    )
    args = parser.parse_args()

    assert args.train.exists(), f"No existe el archivo de train: {args.train}"
    assert args.eval_prompts.exists(), f"No existe el archivo eval prompts: {args.eval_prompts}"
    assert args.eval_v1.exists(), f"No existe el archivo eval v1: {args.eval_v1}"

    eval_prompts = cargar_prompts_eval(args.eval_prompts, args.eval_v1)
    norm_eval_prompts = [normalizar_texto(p) for p in eval_prompts]

    sha256_archivo = calcular_sha256_archivo(args.train)

    candidates = []
    n_lineas_archivo = 0

    with open(args.train, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            n_lineas_archivo += 1
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj["text"]
            m = RX_TURNO.match(text.strip())
            if not m:
                continue
            u = m.group("u")
            a = m.group("a")

            # 3.a 20 <= len(u) <= 100 y 60 <= len(a) <= 240
            if not (20 <= len(u) <= 100 and 60 <= len(a) <= 240):
                continue
            # 3.b u y a sin '<' ni '>'
            if "<" in u or ">" in u or "<" in a or ">" in a:
                continue
            # 3.c a[-1] en .!?»"'
            if a[-1] not in (".", "!", "?", "»", '"', "'"):
                continue
            # 3.d heurístico de degeneración
            is_deg, _ = check_degenerate(a)
            if is_deg:
                continue
            # 3.e ninguna palabra de a aparece 3 o más veces
            palabras = RX_PALABRA.findall(a.lower())
            counts = {}
            for p in palabras:
                counts[p] = counts.get(p, 0) + 1
            if any(cnt >= 3 for cnt in counts.values()):
                continue
            # 3.f round-trip
            if render_turn(u, a, "seco") != text.strip():
                continue
            # 3.g exclusión de evaluación
            u_norm = normalizar_texto(u)
            overlap = False
            for ep_n in norm_eval_prompts:
                if u_norm == ep_n or u_norm in ep_n or ep_n in u_norm:
                    overlap = True
                    break
            if overlap:
                continue

            candidates.append((len(u), idx, u, a, text))

    # 4. Elegir los 2 primeros candidatos ordenados por (len(u), indice_de_linea) con u distintos
    candidates.sort(key=lambda x: (x[0], x[1]))
    chosen = []
    for c in candidates:
        if not chosen or c[2] != chosen[0][2]:
            chosen.append(c)
        if len(chosen) == 2:
            break

    assert len(chosen) == 2, f"No se encontraron 2 candidatos validos, encontrados {len(chosen)}"
    c1, c2 = chosen[0], chosen[1]

    # 5. slot_guard sobre los 4 slots
    u1, a1 = slot_guard(c1[2], "user"), slot_guard(c1[3], "answer")
    u2, a2 = slot_guard(c2[2], "user"), slot_guard(c2[3], "answer")

    # 6. prefijo_texto y asserts
    prefijo_texto = render_historial([(u1, a1), (u2, a2)], "seco")
    assert prefijo_texto.count("<|im_end|>") == 4, "Debe haber 4 <|im_end|>"
    assert prefijo_texto.count("<|im_user|>") == 2, "Debe haber 2 <|im_user|>"
    assert prefijo_texto.count("<|im_start|>seco") == 2, "Debe haber 2 <|im_start|>seco"
    assert prefijo_texto.endswith("<|im_end|>"), "El prefijo debe terminar en <|im_end|>"

    prefijo_norm = normalizar_texto(prefijo_texto)
    for ep_n in norm_eval_prompts:
        assert ep_n not in prefijo_norm, f"Prompt de evaluacion filtrado en prefijo: {ep_n}"
        assert ep_n not in prefijo_texto, f"Prompt de evaluacion filtrado en prefijo crudo: {ep_n}"

    sha256_prefijo = hashlib.sha256(prefijo_texto.encode("utf-8")).hexdigest()
    sha256_l1 = hashlib.sha256(c1[4].encode("utf-8")).hexdigest()
    sha256_l2 = hashlib.sha256(c2[4].encode("utf-8")).hexdigest()

    # Rutas sanitizadas para metadata de prefix.json
    path_train_meta = rel_path(args.train, "data/rwkv_kateto_seco.jsonl")
    path_ep_meta = rel_path(args.eval_prompts, "data/eval_prompts.jsonl")
    path_v1_meta = rel_path(args.eval_v1, "out/eval/prompts_v1.json")

    out_obj = {
        "schema": "eureka-fewshot-prefix/v1",
        "fuente": {
            "path": path_train_meta,
            "sha256_archivo": sha256_archivo,
            "n_lineas_archivo": n_lineas_archivo,
            "lineas": [c1[1], c2[1]],
            "sha256_lineas": [sha256_l1, sha256_l2],
        },
        "excluidos": {
            "path_eval_prompts": path_ep_meta,
            "path_eval_v1": path_v1_meta,
            "n_prompts": len(norm_eval_prompts),
        },
        "voice": "seco",
        "turnos": [[u1, a1], [u2, a2]],
        "prefijo_texto": prefijo_texto,
        "sha256_prefijo": sha256_prefijo,
        "n_chars_prefijo": len(prefijo_texto),
        "marcadores": {"<|im_end|>": 4, "<|im_user|>": 2, "<|im_start|>": 2},
        "nota_anti_hack": (
            "Los marcadores los emite unicamente el template (render_turn/render_historial). "
            "slot_guard verifica que ni el slot user ni el slot answer contienen marcadores, "
            "asi que el <|im_end|> del ejemplo vive SOLO en el turno del assistant: "
            "no hay marcador dentro del prompt del user."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Resumen
    print(f"Lineas elegidas: {c1[1]} y {c2[1]}, n_chars_prefijo={len(prefijo_texto)}")


if __name__ == "__main__":
    main()
