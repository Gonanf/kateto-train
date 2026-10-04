#!/usr/bin/env python3
"""Smoke de 2 pasos en CPU con ROSA encendida detras de `--rosa`.

El wiring es el MISMO que usa el entrenamiento (`opd_train.wire_rosa` ->
`rosa_module.attach_rosa` sobre `model.head`); lo unico que cambia es el student:
`MiniRWKV`, porque el RWKV7 de RWKV-PEFT necesita el kernel triton/cuda y no
corre en CPU (medido: ZeroDivisionError en `rwkvop.py`, WKV=triton, tensores CPU).

La loss es CE next-token sobre el batch de train. Sin teacher: el item R2 mide el
cableado (gradiente vivo, sin NaN), no la calidad del distillation.

    venv/bin/python rwkv_pipeline/smoke_rosa_cpu.py --rosa --steps 2
    venv/bin/python rwkv_pipeline/smoke_rosa_cpu.py            # control
"""
import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import torch
import torch.nn.functional as F

from rwkv_pipeline.mini_rwkv import MiniRWKV
from rwkv_pipeline.opd_train import wire_rosa


def run(steps: int = 2, rosa: bool = False, seed: int = 1337,
        vocab: int = 1024, n_embd: int = 64, n_layer: int = 2,
        seq: int = 16, lr: float = 1e-3) -> dict:
    torch.manual_seed(seed)
    student = MiniRWKV(vocab=vocab, n_embd=n_embd, n_layer=n_layer, ctx_len=seq)
    n_rosa = wire_rosa(student, rosa)
    head = student.head
    rosa_mod = getattr(head, "rosa", None)
    params = [p for p in (rosa_mod.parameters() if rosa_mod else []) if p.requires_grad]
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0)
    losses, grads = [], []
    for _ in range(steps):
        ids = torch.randint(0, vocab, (1, seq))
        logits = student.forward_normal(ids)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, vocab),
                               ids[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads.append(sum(float(p.grad.abs().sum()) for p in params) if params else 0.0)
        opt.step()
        losses.append(float(loss.detach()))
    gate = (rosa_mod.forward_train(torch.randn(1, 4, n_embd))[1]
            if rosa_mod else torch.zeros(1))
    return {"losses": losses, "rosa_params": n_rosa,
            "gate_mean": float(gate.mean().detach()), "head_type": type(head).__name__,
            "rosa_grad": grads}


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke ROSA CPU (2 pasos)")
    ap.add_argument("--rosa", action="store_true")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--seq", type=int, default=16)
    args = ap.parse_args()

    r = run(steps=args.steps, rosa=args.rosa, seed=args.seed, vocab=args.vocab,
            n_embd=args.n_embd, n_layer=args.n_layer, seq=args.seq)
    print(f"rosa={args.rosa} head={r['head_type']} rosa_params={r['rosa_params']:,} "
          f"gate_mean={r['gate_mean']:.6f} seed={args.seed}")
    for i, (loss, grad) in enumerate(zip(r["losses"], r["rosa_grad"]), 1):
        print(f"step {i} loss {loss:.8f} finite={bool(torch.isfinite(torch.tensor(loss)))} "
              f"rosa_grad_abs_sum {grad:.6f}")
    finite = all(torch.isfinite(torch.tensor(x)) for x in r["losses"])
    if args.rosa:
        finite = finite and r["rosa_params"] > 0 and all(g > 0.0 for g in r["rosa_grad"])
    print("VERDICT:", "OK" if finite else "NO-GO")
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())