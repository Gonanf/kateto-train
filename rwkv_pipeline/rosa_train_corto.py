#!/usr/bin/env python3
"""ROSA corto: CE next-token sobre un subconjunto chico del dataset, con SOLO
el head ROSA entrenable. Los pesos van a `out/rosa_head_corto/`.

Por que no `opd_train.py`: ese loop necesita un teacher (ollama) y su
`ckpt_every` filtra por `"lora" in k`, asi que los params de ROSA se perdian al
guardar. Este script no pide teacher (CE contra el dataset) y guarda el head
completo.

Reutiliza lo que ya existe, no reimplementa nada:
  - `opd_rollout.build_student`  -> carga el RWKV7 real (WKV=triton, el unico
    camino que anda en esta box; ver su docstring).
  - `opd_rollout._pad_len`       -> padding a bucket de 64 (bug de cola del
    kernel triton con T % 8 != 0, medido en out/opd/tlen_probe.md).
  - `opd_train.wire_rosa`        -> enganche identico al de infer/eval detras
    de `--rosa`.
  - `data_utils.build_labels`    -> loss-mask qa, el mismo que
    train_lora_base.sh pasa como `--loss_mask qa`.

    venv-unsloth-qwen/bin/python rwkv_pipeline/rosa_train_corto.py \
        --ckpt models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
        --rosa --steps 2 --out out/rosa_head_smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from rwkv_pipeline import opd_rollout as OR            # noqa: E402
from rwkv_pipeline import opd_train                    # noqa: E402
from rwkv_pipeline import data_utils as DU             # noqa: E402
from rwkv_pipeline.rosa_module import RosaHead         # noqa: E402


def load_subset(data: str, n: int) -> list[dict]:
    """Primeras `n` filas con campo `text`. Orden fijo: run reproducible."""
    out: list[dict] = []
    with open(data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("text"):
                out.append(row)
            if len(out) >= n:
                break
    if not out:
        raise SystemExit(f"no hay filas con 'text' en {data}")
    return out


def build_example(tok, text: str, ctx_len: int, delims: dict):
    """(ids, labels) de una fila: trunca a ctx_len, rellena a bucket, -100 fuera
    de la respuesta. `ids` truncado antes de `build_labels` para que el
    loss-mask vea la misma secuencia que el forward.

    Devuelve None si la fila no aporta ningun token supervisado. No es un
    detalle: `F.cross_entropy` con todos los targets en -100 promedia cero
    elementos y sale NaN (medido: 3 de las primeras 5 filas de
    rwkv_kateto_base_train_plus.jsonl). La causa es que en las filas de
    tool-calling el `<|im_end|>` final no tokeniza a la misma secuencia de 7 ids
    que `chatml_seqs`, asi que la guarda anti-fuga de `build_labels` revierte
    todo el span. Descartar la fila equivale a lo que el mask ya decidio: ahi no
    hay gradiente."""
    ids = list(tok.encode(text))[:ctx_len]
    if len(ids) < 2:
        return None
    labels = DU.build_labels(ids, delims=delims)
    if not any(l != DU.IGNORE for l in labels):
        return None
    t_pad = OR._pad_len(len(ids), ctx_len=ctx_len)
    ids += [0] * (t_pad - len(ids))
    labels += [DU.IGNORE] * (t_pad - len(labels))
    return ids, labels


def main() -> int:
    ap = argparse.ArgumentParser(description="ROSA: run corto, head ROSA entrenable")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=str(PROJECT / "data/rwkv_kateto_base_train_plus.jsonl"))
    ap.add_argument("--out", default=str(PROJECT / "out/rosa_head_corto"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--subset", type=int, default=64)
    ap.add_argument("--ctx-len", type=int, default=512)
    ap.add_argument("--retrieval-dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3,
                    help="3e-3 y no 1e-5: los params del head viven en bf16 "
                         "(misma precision que --precision bf16 del repo), y a "
                         "1e-5 el update no mueve el mantissa.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rosa", action="store_true",
                    help="engancha ROSA al head (apagado por default, como en "
                         "opd_train / infer_kateto / eval_multiturn)")
    args = ap.parse_args()
    if not args.rosa:
        raise SystemExit("--rosa es obligatorio en este script: sin la rama el "
                         "head entrenado seria el head de la base, no ROSA")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1337)

    student, tok, info = OR.build_student(args.ckpt, device=args.device, lo_r=0)
    raw = getattr(student, "_raw", student)
    n_rosa = opd_train.wire_rosa(student, True)
    if not isinstance(raw.head, RosaHead):
        raise SystemExit(f"head no es RosaHead: {type(raw.head).__name__}")
    # base congelada: lo unico que aprende este run es el head ROSA, asi que el
    # artefacto guardado es toda la historia y el A/B contra la base es limpio.
    for p in raw.parameters():
        p.requires_grad_(False)
    rosa_params = [p for p in raw.head.rosa.parameters()]
    for p in rosa_params:
        p.requires_grad_(True)

    # ROSA SI estaba en el grafo (`forward_normal` acaba en `self.head(x)` y
    # `attach_rosa` reemplaza `self.head`): los 8.78M params tienen `.grad` no-None
    # en todos los pasos. Lo que moria era la magnitud, porque la mezcla es
    # multiplicativa y el gate se cerraba: `rosa_grad_norm == 0.0` en 177 de los 300
    # pasos de out/rosa_head_corto/metrics.jsonl. Al cerrarse el sigmoid se lleva
    # el gradiente de proj_q/k/v y out_head, y el suyo propio tampoco escapa
    # (~ grad_logits * rosa_logits * sigma'), dejando un llano muerto.
    # out_head=0 (residual zero-init, ReZero/LoRA B=0) deja la rama como no-op
    # exacto en vez de sumar ruido de |rosa_logits| ~18 sobre logits base ~85, y
    # gate sin gradiente quita el multiplicador que se cerraba. El gate queda
    # congelado en su init pero el checkpoint lo guarda igual, asi que inferencia
    # mezcla con el mismo gate que se entreno (load_state_dict lo sobreescribe).
    with torch.no_grad():
        raw.head.rosa.out_head.weight.zero_()
    raw.head.rosa.gate.weight.requires_grad_(False)
    raw.head.rosa.gate.bias.requires_grad_(False)
    # RosaHead envuelve al Linear original: el peso base (device/dtype) vive un
    # nivel mas adentro, en raw.head.head.
    base_head = raw.head.head
    dev = base_head.weight.device
    print(f"[rosa] base {info['arch']['n_embd']}d x{info['arch']['n_layer']}l "
          f"vocab={info['arch']['vocab']} device={dev} dtype={base_head.weight.dtype}")
    print(f"[rosa] head=RosaHead +{n_rosa:,} params trainables "
          f"(retrieval_dim={args.retrieval_dim})")

    rows = load_subset(args.data, args.subset)
    delims = DU.chatml_seqs(tok)
    examples = [build_example(tok, r["text"], args.ctx_len, delims) for r in rows]
    examples = [e for e in examples if e is not None]
    if not examples:
        raise SystemExit("el subconjunto no dejo ninguna secuencia de >=2 tokens")
    print(f"[rosa] subconjunto: {len(examples)}/{len(rows)} filas usables "
          f"(ctx_len={args.ctx_len}, T real={examples[0][0].__len__()}->"
          f"{OR._pad_len(examples[0][0].__len__(), ctx_len=args.ctx_len)} con pad)")

    opt = torch.optim.AdamW(rosa_params, lr=args.lr, weight_decay=0.0)
    # la prueba de que hay senal NO es que la loss sea finita: es que los pesos
    # se movieron. Se mide el delta L2 contra el estado inicial.
    init_norm = float(sum(float(p.detach().float().norm() ** 2)
                          for p in rosa_params) ** 0.5)
    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    print(f"[rosa] out={out} steps={args.steps} lr={args.lr} "
          f"rosa_norm_init={init_norm:.6f}")

    for step in range(1, args.steps + 1):
        t0 = time.time()
        ids, labels = examples[(step - 1) % len(examples)]
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        y = torch.tensor([labels], dtype=torch.long, device=dev)
        logits = raw.forward_normal(x)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]),
            y[:, 1:].reshape(-1), ignore_index=DU.IGNORE)
        if not torch.isfinite(loss).item():
            raise SystemExit(f"[rosa] loss no finito en step {step}: "
                             f"{float(loss.detach())} — abortando "
                             f"(no metric invented)")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = sum(float(p.grad.detach().float().norm() ** 2)
                   for p in rosa_params if p.grad is not None) ** 0.5
        torch.nn.utils.clip_grad_norm_(rosa_params, 1.0)
        opt.step()
        ntok = int((y[:, 1:] != DU.IGNORE).sum())
        rec = {"step": step, "loss": round(float(loss.detach()), 5),
               "label_tokens": ntok, "rosa_grad_norm": round(grad, 6),
               "wall_s": round(time.time() - t0, 2)}
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[rosa] step {step}/{args.steps} " + json.dumps(rec), flush=True)

    final_norm = float(sum(float(p.detach().float().norm() ** 2)
                           for p in rosa_params) ** 0.5)
    # state_dict del modulo rosa pelado (proj_q/proj_k/proj_v/out_head/gate), no
    # prefijado con `head.rosa.` -> es lo que infer_kateto.RosaAssociativeMemory
    # puede cargar con un `load_state_dict` directo.
    ck = out / "rosa_head.pth"
    torch.save({"step": args.steps, "retrieval_dim": args.retrieval_dim,
                "vocab_size": info["arch"]["vocab"], "n_embd": info["arch"]["n_embd"],
                "n_layer": info["arch"]["n_layer"],
                "rosa_norm_init": init_norm, "rosa_norm_final": final_norm,
                "state_dict": {k: v.detach().cpu().clone()
                               for k, v in raw.head.rosa.state_dict().items()}}, ck)
    print(f"[rosa] guardado {ck}")
    print(f"[rosa] rosa_norm {init_norm:.6f} -> {final_norm:.6f} "
          f"(delta {abs(final_norm - init_norm):.6f})")
    moved = abs(final_norm - init_norm) > 0.0
    print("VERDICT:", "OK" if moved else "NO-GO (los pesos no se movieron)")
    return 0 if moved else 1


if __name__ == "__main__":
    sys.exit(main())
