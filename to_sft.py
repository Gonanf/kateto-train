"""Convierte dataset_filter jsonl a formato SFT de RWKV-PEFT.

Entrada: dataset.jsonl con {"question","answer",...} (incluye followups y tools).
Salida: sft_train.jsonl + sft_eval.jsonl con {"query","response"} (split 90/10
deterministico por hash, los seeds de eval nunca se ven en train) y calib.txt
con texto plano para la imatrix del quant.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def main(src: str, out_dir: str) -> None:
    src_p = Path(src)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_f = (out / "sft_train.jsonl").open("w", encoding="utf-8")
    eval_f = (out / "sft_eval.jsonl").open("w", encoding="utf-8")
    calib_f = (out / "calib.txt").open("w", encoding="utf-8")
    n_train = n_eval = 0
    with src_p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except Exception:
                continue
            q = (p.get("question") or "").strip()
            a = (p.get("answer") or "").strip()
            if not q or not a:
                continue
            rec = {"query": q, "response": a}
            h = hashlib.md5((q + "\x00" + a).encode()).hexdigest()
            if int(h[:2], 16) < 26:  # ~10% a eval
                eval_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_eval += 1
            else:
                train_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_train += 1
            calib_f.write(q + "\n" + a + "\n\n")
    print(f"train={n_train} eval={n_eval} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
