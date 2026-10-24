#!/usr/bin/env python3
"""Convierte el dataset de preferencias v8 al formato SFT consumable por sft_generate_h5.py.

Lee data/kateto_orpo_v8.jsonl y genera moe_rlhf/data/sft_text.jsonl con {"text": prompt + chosen}.
"""

import os
import sys
import json
import math
import hashlib
import argparse


def compute_percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[int(lo)] * (hi - k)
    d1 = sorted_vals[int(hi)] * (k - lo)
    return float(d0 + d1)


def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_and_convert(rows):
    converted_texts = []
    n_chosen_con_im_end = 0
    n_chosen_sin_im_end = 0
    n_prompt_vacio = 0
    n_textos_mayores_a_1024_chars = 0
    lengths = []

    for idx, row in enumerate(rows, 1):
        for req_key in ("prompt", "chosen", "rejected"):
            if req_key not in row:
                raise ValueError(
                    f"Fila inválida en línea {idx}: falta la clave obligatoria '{req_key}'. "
                    f"Claves encontradas: {list(row.keys())}"
                )

        prompt = row["prompt"]
        chosen = row["chosen"]

        if len(prompt.strip()) == 0:
            n_prompt_vacio += 1

        if chosen.endswith("<|im_end|>"):
            n_chosen_con_im_end += 1
        else:
            n_chosen_sin_im_end += 1

        text = prompt + chosen
        text_len = len(text)
        lengths.append(text_len)

        if text_len > 1024:
            n_textos_mayores_a_1024_chars += 1

        converted_texts.append(text)

    sorted_lens = sorted(lengths)
    p50 = compute_percentile(sorted_lens, 50)
    p95 = compute_percentile(sorted_lens, 95)
    max_len = max(lengths) if lengths else 0

    stats = {
        "n": len(rows),
        "n_chosen_con_im_end": n_chosen_con_im_end,
        "n_chosen_sin_im_end": n_chosen_sin_im_end,
        "chars_p50": p50,
        "chars_p95": p95,
        "chars_max": max_len,
        "n_textos_mayores_a_1024_chars": n_textos_mayores_a_1024_chars,
        "n_prompt_vacio": n_prompt_vacio,
    }

    return converted_texts, stats


def run_selfcheck():
    # 1. Conversión de 2 filas sintéticas
    synthetic_rows = [
        {
            "prompt": "User: Pregunta 1\nAssistant: ",
            "chosen": "Respuesta 1 válida<|im_end|>",
            "rejected": "Respuesta 1 descartada",
        },
        {
            "prompt": "User: Pregunta 2\nAssistant: ",
            "chosen": "Respuesta 2 sin im_end",
            "rejected": "Respuesta 2 descartada",
        },
    ]

    converted_texts, stats = validate_and_convert(synthetic_rows)
    assert len(converted_texts) == 2, f"Expected 2 texts, got {len(converted_texts)}"
    assert converted_texts[0] == "User: Pregunta 1\nAssistant: Respuesta 1 válida<|im_end|>"
    assert converted_texts[1] == "User: Pregunta 2\nAssistant: Respuesta 2 sin im_end"
    assert stats["n_chosen_con_im_end"] == 1, f"Expected 1 con_im_end, got {stats['n_chosen_con_im_end']}"
    assert stats["n_chosen_sin_im_end"] == 1, f"Expected 1 sin_im_end, got {stats['n_chosen_sin_im_end']}"
    assert stats["n"] == 2

    # 2. Validación de fallo ante fila sin 'chosen'
    invalid_row = [{"prompt": "User: Solo prompt", "rejected": "Rechazado"}]
    failed = False
    try:
        validate_and_convert(invalid_row)
    except ValueError:
        failed = True

    assert failed, "El par sin 'chosen' debió haber fallado la validación"

    print("selfcheck OK")
    sys.exit(0)


def process_dataset(src_path, out_path, stats_path):
    if not os.path.exists(src_path):
        sys.stderr.write(f"Error: no existe el archivo de origen '{src_path}'.\n")
        sys.exit(1)

    rows = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                row_obj = json.loads(line_str)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"Error al decodificar JSON en línea {line_num}: {e}\n")
                sys.exit(1)
            rows.append(row_obj)

    if len(rows) != 550:
        sys.stderr.write(
            f"Error: se esperaban exactamente 550 líneas en {src_path}, pero se encontraron {len(rows)}.\n"
        )
        sys.exit(1)

    converted_texts, stats = validate_and_convert(rows)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f_out:
        for t in converted_texts:
            f_out.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    stats["sha256"] = compute_sha256(out_path)

    os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f_stats:
        json.dump(stats, f_stats, indent=2, ensure_ascii=False)

    print(f"n: {stats['n']}")
    print(f"n_chosen_con_im_end: {stats['n_chosen_con_im_end']}")
    print(f"n_chosen_sin_im_end: {stats['n_chosen_sin_im_end']}")
    print(f"chars_p50: {stats['chars_p50']}")
    print(f"chars_p95: {stats['chars_p95']}")
    print(f"chars_max: {stats['chars_max']}")
    print(f"n_textos_mayores_a_1024_chars: {stats['n_textos_mayores_a_1024_chars']}")
    print(f"n_prompt_vacio: {stats['n_prompt_vacio']}")
    print(f"sha256: {stats['sha256']}")
    print("[ok]")


def main():
    parser = argparse.ArgumentParser(description="Construye sft_text.jsonl a partir de preferencias v8")
    parser.add_argument("--src", default="data/kateto_orpo_v8.jsonl", help="Ruta al archivo origen")
    parser.add_argument("--out", default="moe_rlhf/data/sft_text.jsonl", help="Ruta al archivo sft salida")
    parser.add_argument("--stats", default="moe_rlhf/data/stats.json", help="Ruta al archivo de estadísticas")
    parser.add_argument("--selfcheck", action="store_true", help="Ejecuta selfcheck de validación lógica")
    args = parser.parse_args()

    if args.selfcheck:
        run_selfcheck()
    else:
        process_dataset(args.src, args.out, args.stats)


if __name__ == "__main__":
    main()
