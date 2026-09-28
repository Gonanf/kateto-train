#!/usr/bin/env python3
"""On-policy rollout of the RWKV-7 student with per-position logits — Phases 0.3.

Builds the **autograd** student (RWKV-PEFT `rwkvt.rwkv7.model.RWKV7`, NOT the
no_grad+torchscript `infer_kateto.RWKV_RNN` — rule 6), generates the response
on-policy (sampling) and provides:

  * `rollout_from_prompt(...)` -> sampled response + per-position prefixes +
                                  ` thinking`-channel stats (rule 9).
  * `rescore_response(...)`    -> ONE grad-enabled forward over prompt+response
                                  returning logits aligned to response tokens.

Loss over only the response tokens (assistant-only) is applied in opd_train.py
via `data_utils.build_labels(..., mode="legacy")` (World format), per §3/§1.2
(base 1.5B uses `User: {q}\n\nAssistant:`; ChatML is NOT used for the base).

Env (rule 7) is configured here before `import torch`.
"""
from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Rule 7: ROCm/torch pre-import env (must precede `import torch`).
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# RWKV-PEFT custom-op env (the training scripts use exactly these).
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_TRAIN_TYPE", "none")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")
# WKV must be "triton" on this box, NOT "cuda":
#   - "cuda" takes the else-branch in RWKV-PEFT/rwkvt/operator/rwkvop.py:296,
#     which JIT-builds cuda/rwkv7_clampw.cu. That build cannot succeed here:
#     it passes nvcc flags (-res-usage, --use_fast_math, -Xptxas,
#     --extra-device-vectorization) to clang++/hipcc, --offload-arch has no
#     gfx1034, and /opt/rocm/include/hip/amd_detail/amd_hip_bf16.h fails to
#     compile. It also hangs forever first, on the stale
#     ~/.cache/torch_extensions/*/lock via torch.utils.file_baton.
#   - "fla" is unavailable: rwkvfla is not installed in venv-unsloth-qwen.
#   - "triton" is the path train_lora_base.sh itself uses (--op triton); the
#     gfx1034 LDS K=8 patch lives there.
os.environ.setdefault("WKV", "triton")
os.environ.setdefault("FUSED_KERNEL", "0")

import torch
import torch.nn.functional as F
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parent.parent
# When run as a script, sys.path[0] is rwkv_pipeline/ and the project root is
# not importable (`from rwkv_pipeline import infer_kateto` would fail).
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "RWKV-PEFT"))

# RWKV7 is imported lazily inside build_student() because doing so triggers the
# `rwkvt.operator.rwkvop` custom CUDA op build at import time (needs CWD =
# RWKV-PEFT and the x070 env). Keeping it lazy lets pure-math callers (e.g.
# opd_train --selftest) import this module without touching the toolchain.

_RWKV7 = None


def _get_rwkv7():
    """Lazy import of the RWKV-PEFT trainable RWKV7.

    RWKV-PEFT reads its env contract at IMPORT time (rwkvt/operator/rwkvop.py:28
    reads WKV; att.py reads FUSED_KERNEL / RWKV_TRAIN_TYPE). train.py sets those
    from CLI args before importing. Importing the model directly without them
    either KeyErrors or takes the CUDA branch, which JIT-builds a native
    extension into ~/.cache/torch_extensions and then blocks forever on a stale
    `lock` file via torch.utils.file_baton (observed: hung >8 min at 0% CPU).

    WKV=triton is the path that works on this box: `rwkvfla` is NOT installed in
    venv-unsloth-qwen, and train_lora_base.sh itself passes `--op triton` (the
    gfx1034 LDS K=8 patch lives in the triton path). setdefault so an explicit
    caller export still wins.
    """
    global _RWKV7
    if _RWKV7 is None:
        for k, v in (("WKV", "triton"), ("RWKV_MY_TESTING", "x070"),
                     ("RWKV_TRAIN_TYPE", "none"), ("FUSED_KERNEL", "0"),
                     ("RWKV_FLOAT_MODE", "bf16"), ("RWKV_JIT_ON", "0")):
            os.environ.setdefault(k, v)
        cwd = os.getcwd()
        try:
            os.chdir(str(PROJECT / "RWKV-PEFT"))  # custom-op `load()` uses cuda/
            from rwkvt.rwkv7.model import RWKV7  # noqa: F401
            _RWKV7 = RWKV7
        finally:
            os.chdir(cwd)
    return _RWKV7


@dataclass
class Rollout:
    prompt_text: str
    prompt_ids: List[int] = field(default_factory=list)
    response_ids: List[int] = field(default_factory=list)
    prefixes: List[str] = field(default_factory=list)   # len == len(response_ids)
    think_tokens: int = 0
    wall_s: float = 0.0
    tok_per_s: float = 0.0
    finished: bool = False

    @property
    def think_fraction(self) -> float:
        if not self.response_ids:
            return 0.0
        return self.think_tokens / len(self.response_ids)


def _is_think(tok_bytes: bytes) -> bool:
    try:
        return tok_bytes.decode("utf-8", "replace").lstrip().startswith("think")
    except Exception:
        return False


def _load_tokenizer():
    from rwkv_pipeline import infer_kateto
    vocab = PROJECT / "RWKV-PEFT/rwkv_vocab_v20230424.txt"
    return infer_kateto.RWKV_TOKENIZER(str(vocab))


def _infer_arch(ckpt_path: str) -> dict:
    """Read architecture from the checkpoint (measured, not assumed)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    n_embd = ck["emb.weight"].shape[1]
    nl = 0
    while f"blocks.{nl}.att.r_k" in ck:
        nl += 1
    _, hs = ck["blocks.0.att.r_k"].shape
    vocab = ck["head.weight"].shape[0]
    del ck
    gc.collect()
    return {"n_embd": n_embd, "n_layer": nl, "head_size": hs, "vocab": vocab}


def _make_args(n_embd: int, n_layer: int, head_size: int, vocab: int,
               ctx_len: int = 512, grad_cp: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        vocab_size=vocab, n_embd=n_embd, n_layer=n_layer, dim_att=n_embd,
        head_size_a=head_size, head_size_divisor=8, grad_cp=grad_cp,
        dropout=0.0, train_type="none", peft="none", ctx_len=ctx_len,
        head_qk=0, pre_ffn=0, tiny_att_layer=-999, my_pos_emb=0,
        my_att_shift=1, my_ffn_shift=1, l2warp_sparse=0,
        my_testing="x070", dim_ffn=0, n_head=0)


def build_student(ckpt_path: str, device: str = "cuda", dtype: str = "bf16",
                  ctx_len: int = 512, grad_cp: int = 1, lo_r: int = 16,
                  lo_alpha: int = 32, lo_dropout: float = 0.01):
    """Load the autograd RWKV7 student; optionally wrap in LoRA (rule 8: cuda).

    Full fine-tune does not fit in 4 GB VRAM (weights alone ~3.15 GB, §1.3),
    so the trainable params are LoRA adapters, matching the repo's train path.
    Returns (model_with_peft, tokenizer, info_dict).
    """
    arch = _infer_arch(ckpt_path)
    args = _make_args(arch["n_embd"], arch["n_layer"], arch["head_size"],
                      arch["vocab"], ctx_len=ctx_len, grad_cp=grad_cp)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = _get_rwkv7()(args)
    missing, unexpected = model.load_state_dict(ck, strict=False)
    del ck
    gc.collect()
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
    model.to(dtype=torch_dtype, device=device)
    model.train()

    if lo_r > 0:
        from peft import LoraConfig, TaskType, get_peft_model
        cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=lo_r, lora_alpha=lo_alpha,
            lora_dropout=lo_dropout,
            target_modules=["receptance", "key", "value", "output"])
        raw = model                       # LoRA adapters are added in-place
        model = get_peft_model(model, cfg)
        # object.__setattr__ = atributo PELADO, sin registrar en _modules.
        # La asignacion normal (`model._raw = raw`) mete `raw` en el arbol de
        # modulos OTRA VEZ (ya esta anidado adentro del PeftModel), asi que
        # .parameters() lo cuenta dos veces.
        object.__setattr__(model, "_raw", raw)
        model.print_trainable_parameters()
    else:
        # object.__setattr__ es OBLIGATORIO aca. `model._raw = model` registra el
        # modulo como hijo DE SI MISMO (nn.Module.__setattr__ hace add_module),
        # el arbol queda ciclado y .eval()/.train() recursan hasta RecursionError
        # (medido: 986 niveles antes de reventar). Con el atributo pelado,
        # getattr(model, "_raw", model) sigue resolviendo a `model`.
        object.__setattr__(model, "_raw", model)
    info = {"arch": arch, "device": device, "dtype": dtype,
            "missing": len(missing), "unexpected": len(unexpected)}
    return model, _load_tokenizer(), info


def _forward(student, x) -> torch.Tensor:
    """forward_normal on the underlying RWKV7 (LoRA adapters integrated)."""
    raw = getattr(student, "_raw", student)
    return raw.forward_normal(x)


def _trainable_params(student):
    for p in getattr(student, "_raw", student).parameters():
        if p.requires_grad:
            yield p


def _pad_len(n: int, bucket: int = 64, ctx_len: int = 512) -> int:
    """Target length for a forward: next multiple of `bucket` (>= bucket).

    Why padding is REQUIRED on this stack (measured, see out/opd/tlen_probe.md):
    the WKV=triton path (RWKV-PEFT/rwkvt/operator/rwkvop.py, TritonRWKV7)
    processes the sequence in chunks of K=8 with `T` as a tl.constexpr and
    `s = zeros(B,H,T//K,C,C)`: for T % 8 != 0 the tail positions are never
    written (uninitialized memory) and every distinct T triggers a fresh
    Triton JIT compile (~6s each). Padding to a bucket of 64 both fixes the
    tail and bounds the number of kernel compilations to ctx_len/64.
    Causality (mask2 = t >= t.trans()) means appended pad tokens cannot
    affect logits at earlier positions.
    """
    target = max(bucket, ((n + bucket - 1) // bucket) * bucket)
    return min(target, ctx_len)


def _forward_padded(student, token_ids: List[int], ctx_len: int) -> torch.Tensor:
    """forward over token_ids zero-padded to the bucket length; (1, T_pad, vocab)."""
    dev = next(_trainable_params(student)).device
    t_pad = _pad_len(len(token_ids), ctx_len=ctx_len)
    ids = list(token_ids)[:ctx_len]
    ids = ids + [0] * (t_pad - len(ids))
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    with torch.no_grad():
        return _forward(student, x)


def _next_logits(student, token_ids: List[int], ctx_len: int) -> torch.Tensor:
    logits = _forward_padded(student, token_ids, ctx_len)
    return logits[0, len(token_ids) - 1, :]   # real position, padding is after


def _sample_next(logits: torch.Tensor, temperature: float, top_p: float,
                 top_k: int, block_think: bool, tokenizer) -> int:
    lp = logits.float().clone()
    if block_think:
        v, i = lp.topk(min(top_k * 4 if top_k else 128, lp.shape[0]))
        for t in i.tolist():
            if _is_think(tokenizer.decodeBytes([t])):
                lp[t] = -float("inf")
    if temperature != 1.0:
        lp = lp / temperature
    if top_p < 1.0:
        sorted_lp, sorted_idx = torch.sort(lp, descending=True)
        cum = torch.softmax(sorted_lp, dim=-1).cumsum(dim=-1)
        sp = torch.softmax(sorted_lp, dim=-1)
        sorted_lp[(cum - sp) > top_p] = -float("inf")
        lp = torch.zeros_like(lp).scatter_(0, sorted_idx, sorted_lp)
    if top_k > 0:
        v, i = torch.topk(lp, min(top_k, lp.shape[0]))
        mask = torch.full_like(lp, -float("inf")).scatter_(0, i, v)
        lp = mask
    probs = F.softmax(lp, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def _decode_str(tokenizer, ids: List[int]) -> str:
    try:
        return tokenizer.decodeBytes(list(ids)).decode("utf-8", errors="replace")
    except Exception:
        return tokenizer.decodeBytes(list(ids)).decode("latin-1", errors="replace")


def rollout_from_prompt(student, tokenizer, prompt_text: str,
                        max_len: int = 32, temperature: float = 0.8,
                        top_p: float = 0.8, top_k: int = 0,
                        ctx_len: int = 512, block_think: bool = False,
                        end_token: int = 0) -> Rollout:
    """Sample response on-policy; record per-position prefixes and think stats."""
    prompt_ids = tokenizer.encode(prompt_text)
    roll = Rollout(prompt_text=prompt_text, prompt_ids=prompt_ids[:])
    t0 = time.time()
    token_ids = list(prompt_ids)
    while len(roll.response_ids) < max_len:
        lg = _next_logits(student, token_ids, ctx_len)
        tok = _sample_next(lg, temperature, top_p, top_k, block_think, tokenizer)
        if tok == end_token:
            roll.finished = True
            break
        roll.response_ids.append(tok)
        roll.prefixes.append(_decode_str(tokenizer, list(prompt_ids) + roll.response_ids))
        if _is_think(tokenizer.decodeBytes([tok])):
            roll.think_tokens += 1
        token_ids.append(tok)
    roll.wall_s = round(time.time() - t0, 3)
    roll.tok_per_s = round(len(roll.response_ids) / roll.wall_s, 3) if roll.wall_s else 0.0
    return roll


def rescore_response(student, tokenizer, prompt_ids: List[int],
                     response_ids: List[int], ctx_len: int = 512):
    """ONE grad-enabled forward over prompt+response.

    Returns (rows, prompt_len): rows = (T_resp, vocab) with row j predicting
    response[j], and autograd flows through the student weights.
    """
    dev = next(_trainable_params(student)).device
    keep = (list(prompt_ids) + list(response_ids))[-ctx_len:]
    n_real = len(keep)
    t_pad = _pad_len(n_real, ctx_len=ctx_len)
    ids = keep + [0] * (t_pad - n_real)
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    with torch.enable_grad():
        out = _forward(student, x)         # (1, T_pad, vocab)
    # rows for real positions only; padded rows are dropped (loss never sees
    # them), and causality makes their content irrelevant to the kept rows.
    rows = out[0, len(prompt_ids) - 1:len(prompt_ids) - 1 + len(response_ids), :]
    return rows.contiguous(), len(prompt_ids)


def run_gate_b(ckpt_path: str, prompts: List[str], max_len: int = 24,
               device: str = "cuda", lo_r: int = 16) -> dict:
    """Phase-0.3 Gate B: rollout on `prompts`; report length + tokens/s."""
    student, tokenizer, info = build_student(ckpt_path, device=device, lo_r=lo_r)
    lens, tok, think, fin = [], [], [], []
    for p in prompts:
        r = rollout_from_prompt(student, tokenizer, p, max_len=max_len)
        lens.append(len(r.response_ids))
        tok.append(r.tok_per_s)
        think.append(r.think_fraction)
        fin.append(1.0 if r.finished else 0.0)
    n = max(1, len(prompts))
    return {"n_prompts": len(prompts),
            "mean_response_len": round(float(sum(lens) / len(lens)), 2),
            "mean_tokens_per_s": round(float(sum(tok) / len(tok)), 2),
            "think_fraction_mean": round(float(sum(think)) / n, 4),
            "finished_fraction": round(float(sum(fin)) / n, 4),
            "student_info": info}


def main() -> None:
    import argparse
    import json
    import random
    ap = argparse.ArgumentParser(description="OPD rollout Gate B")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=24)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--qa", default=str(PROJECT / "data/kateto_qa.jsonl"))
    ap.add_argument("--lo-r", type=int, default=16)
    args = ap.parse_args()

    prompts = []
    with open(args.qa, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line).get("question", "")
            if q and not q.startswith("[DRY"):
                prompts.append(f"User: {q}\n\nAssistant:")
    random.seed(1337)
    random.shuffle(prompts)
    prompts = prompts[: args.n_prompts]
    print(json.dumps(run_gate_b(args.ckpt, prompts, max_len=args.max_len,
                                device=args.device, lo_r=args.lo_r), indent=2))


if __name__ == "__main__":
    main()