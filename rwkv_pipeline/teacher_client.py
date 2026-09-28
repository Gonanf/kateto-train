#!/usr/bin/env python3
"""Teacher client for On-Policy Distillation (OPD) — Phase 0.2 (docs/opd-brief-phase0.md).

Two backends, same per-position contract:

    for response position j (0-based over the continuation):
        "distribution the TEACHER assigns to the token at position j"

Mode A (hf, default):  local transformers AutoModelForCausalLM on CPU / given
    device. `logits_at(prompt, continuation)` returns dense logits
    (T_cont, vocab_teacher): row j = logits predicting continuation[j],
    aligned with the student's response position j.

Mode B (router): HTTP client to llama.cpp `/v1/chat/completions`
    (`top_logprobs=K`). `top_logprobs_at(prefix)` issues one request per
    position (prefix = prompt + continuation[:j]) and returns the teacher's
    top-K next-token distribution, i.e. the distribution over continuation[j].

The reverse-KL support is aligned on byte-strings in opd_train.py (the
teacher/student tokenizers are different families, per §2 of the brief).

Backend chosen with --teacher-backend {hf,router} (default hf) in opd_train.py.
Mode A deps: transformers + torch (in venv-unsloth-qwen). Mode B: stdlib only.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any, List, Optional, Tuple

# ROCm/torch pre-import env (must precede `import torch`).
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch


class TeacherGateError(RuntimeError):
    """Raised when logits_at / top_logprobs fails the Phase-0.2 Gate A."""


def _softmax_sum_check(t: torch.Tensor, atol: float = 1e-3, label: str = "") -> None:
    """Gate A (part 2): log_softmax over last dim, exp sums ~1.0 within atol."""
    lsm = torch.log_softmax(t.float(), dim=-1)
    s = lsm.exp().sum(dim=-1)
    bad = (s - 1.0).abs().max().item()
    if bad > atol:
        raise TeacherGateError(
            f"Gate A softmax-sum failed ({label}): max|sum-1|={bad:.2e} > {atol:.1e}")
    return None


def _token_str(item: dict) -> str:
    """Pack a llama.cpp logprob entry into a lossless str carrier.

    llama.cpp sends both `token` (str, may mangle partial-UTF-8 tokens) and
    `bytes` (the authoritative byte list). We pack the byte list as latin-1 so
    RouterTeacherBackend.decode_token() round-trips it byte-exactly. Falls back
    to `token` when `bytes` is absent (other OpenAI-compatible servers).
    """
    b = item.get("bytes")
    if isinstance(b, list) and b:
        try:
            return bytes(b).decode("latin-1")
        except (ValueError, TypeError):
            pass
    return item.get("token", "") or ""


class HFTEACHERBackend:
    """Mode A — transformers AutoModelForCausalLM (local, CPU by default)."""

    def __init__(self, model_id: str, device: str = "cpu", dtype: str = "bf16"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:  # pragma: no cover
            raise TeacherGateError(f"transformers not importable: {e}") from e
        torch_dtype = torch.bfloat16 if dtype == "bf16" else (
            torch.float32 if dtype == "fp32" else torch.float16)
        self._t0 = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, device_map=device,
            low_cpu_mem_usage=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        emb = self.model.get_output_embeddings()
        self.vocab_size = emb.weight.shape[0] if emb is not None \
            else self.model.config.vocab_size
        self.model.eval()
        print(f"[teacher hf] loaded '{model_id}' on {device} "
              f"in {time.time()-self._t0:.1f}s, vocab={self.vocab_size}")

    @torch.no_grad()
    def logits_at(self, prompt_text: str, continuation_text: str) -> torch.Tensor:
        """One forward over prompt+continuation; return (T_cont, vocab_teacher).

        Row j = logits predicting continuation[j] (j = 0..T_cont-1).
        """
        ids = self.tokenizer(prompt_text + continuation_text, return_tensors="pt")
        input_ids = ids["input_ids"].to(self.model.device)
        T_total = input_ids.shape[1]
        ids_c = self.tokenizer(continuation_text, add_special_tokens=False)["input_ids"]
        T_cont = len(ids_c)
        out = self.model(input_ids=input_ids)
        logits = out.logits[0]                       # (T_total, vocab)
        # row i predicts token i+1; continuation[0] at T_total-T_cont (prompt end).
        start = T_total - T_cont
        slice_rows = logits[start:start + T_cont]    # (T_cont, vocab)
        return slice_rows.to("cpu")

    def decode_token(self, token_id: int) -> bytes:
        return self.tokenizer.decode([token_id]).encode("utf-8", errors="replace")


class RouterTEACHERBackend:
    """Mode B — llama.cpp OpenAI-compatible router, top_logprobs per position.

    llama-server returns the real per-position top-K over the tokens *it*
    generates. For OPD we query the teacher at prefix = prompt +
    student_response[:j] with max_tokens=1 and top_logprobs=K: the returned
    top-K is the teacher's next-token distribution over continuation[j].
    """

    def __init__(self, base_url: str = "http://127.0.0.1:11434/v1",
                 model: Optional[str] = None, top_k: int = 20, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.top_k = top_k
        self.timeout = timeout

    def _chat(self, prompt: str) -> Tuple[str, List[Tuple[str, float]]]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,          # REQUIRED: llama.cpp 400s on top_logprobs alone
            "top_logprobs": self.top_k,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        try:
            content = obj["choices"][0]["logprobs"]["content"]
            item = content[0]
            top = [(_token_str(t), float(t["logprob"]))
                   for t in (item.get("top_logprobs") or [])]
            return _token_str(item), top
        except (KeyError, IndexError, TypeError) as e:
            raise TeacherGateError(f"router logprobs parse failed: {e} — {obj}") from e

    def top_logprobs_at(self, prefix_text: str) -> List[Tuple[str, float]]:
        """Teacher top-K next-token distribution after prefix_text."""
        _, top = self._chat(prefix_text)
        return top

    def decode_token(self, token: str) -> bytes:
        # llama.cpp returns BOTH `token` (str) and `bytes` (authoritative byte list).
        # _token_str() packs the byte list as latin-1, so this round-trips losslessly
        # even for partial-UTF-8 tokens. Plain .encode("utf-8") corrupts those, and
        # the whole reverse-KL support is aligned on byte-strings (brief §2).
        try:
            return token.encode("latin-1")
        except UnicodeEncodeError:
            return token.encode("utf-8", errors="replace")


class TeacherClient:
    """Unified facade; construct with backend='hf' or 'router'."""

    def __init__(self, backend: str = "hf",
                 model: Optional[str] = None, device: str = "cpu",
                 dtype: str = "bf16", base_url: str = "http://127.0.0.1:11434/v1",
                 top_k: int = 20, timeout: int = 120, **kwargs: Any):
        self.backend = backend
        self.top_k = top_k
        if backend == "hf":
            if not model:
                raise TeacherGateError("hf backend needs --teacher-model")
            self.inner = HFTEACHERBackend(model, device=device, dtype=dtype)
            self.vocab_size = self.inner.vocab_size
        elif backend == "router":
            self.inner = RouterTEACHERBackend(
                base_url=base_url, model=model, top_k=top_k, timeout=timeout)
            self.vocab_size = None
        else:
            raise TeacherGateError(f"unknown backend '{backend}' (hf|router)")

    # -- distribution access -------------------------------------------------
    def logits_at(self, prompt_text: str, continuation_text: str) -> torch.Tensor:
        """Dense teacher logits for the continuation (Mode A only)."""
        if self.backend != "hf":
            raise TeacherGateError("logits_at only for backend=hf")
        return self.inner.logits_at(prompt_text, continuation_text)

    def top_logprobs_at(self, prefix_text: str) -> List[Tuple[str, float]]:
        """Top-K next-token distribution after prefix (Mode B)."""
        if self.backend != "router":
            raise TeacherGateError("top_logprobs_at only for backend=router")
        return self.inner.top_logprobs_at(prefix_text)

    def decode_teacher_token(self, token) -> bytes:
        return self.inner.decode_token(token)

    def run_gate_a(self, prompt_text: str, continuation_text: str) -> dict:
        """Phase-0.2 Gate A probe; returns measured facts (no KL claims)."""
        if self.backend == "hf":
            t0 = time.time()
            lg = self.logits_at(prompt_text, continuation_text)
            wall = time.time() - t0
            tcont = self.inner.tokenizer(
                continuation_text, add_special_tokens=False)["input_ids"]
            _softmax_sum_check(lg, label="last-row")
            return {
                "backend": "hf", "shape": tuple(lg.shape),
                "T_cont_expected": len(tcont),
                "T_cont_matches": tuple(lg.shape)[0] == len(tcont),
                "wall_s": round(wall, 3),
            }
        t0 = time.time()
        top = self.top_logprobs_at(prompt_text)
        wall = time.time() - t0
        lps = torch.tensor([lp for _, lp in top])
        return {
            "backend": "router", "model": self.inner.model,
            "top_k": len(top), "wall_s": round(wall, 3),
            "teacher_self_coverage": round(float(lps.exp().sum()), 4),
        }


def main() -> None:  # self-test / Gate A probe
    import argparse
    ap = argparse.ArgumentParser(description="Teacher client self-test (Gate A)")
    ap.add_argument("--backend", choices=["hf", "router"], default="hf")
    ap.add_argument("--teacher-model", default=None)
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--prompt", default="User: Hello there.\n\nAssistant:")
    ap.add_argument("--continuation", default="Hi, how can I help?")
    args = ap.parse_args()

    tc = TeacherClient(backend=args.backend, model=args.teacher_model,
                       device=args.device, base_url=args.base_url,
                       top_k=args.top_k)
    r = tc.run_gate_a(args.prompt, args.continuation)
    print("GATE A RESULT:", json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()