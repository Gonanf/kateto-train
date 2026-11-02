#!/usr/bin/env python3
"""HTTP backend for eval_generation — llama.cpp /completion endpoint.

Uso independiente (smoke test):
  python scripts/eval_http_backend.py --model RWKV-2.9B --prompt "Che, contame algo"

Uso como módulo:
  from scripts.eval_http_backend import generate_http
  r = generate_http("http://127.0.0.1:11434", "RWKV-2.9B", "User: hola\n\nAssistant:")
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def generate_http(
    base_url: str,
    model: str,
    prompt_text: str,
    *,
    max_len: int = 64,
    temperature: float = 0.80,
    top_p: float = 0.70,
    seed: int = 1337,
    stop: list[str] | None = None,
    repeat_penalty: float | None = None,
    timeout_s: int = 900,
) -> dict:
    """Generate text via llama.cpp /completion endpoint.

    Returns {"text": str, "n_tokens": int, "stopping_word": str|None,
             "stopped": bool, "model": str, "raw": {...}}.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt_text,
        "n_predict": max_len,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "stream": False,
        "cache_prompt": False,  # mandatory: avoid state contamination between calls
    }
    if stop:
        payload["stop"] = stop
    if repeat_penalty is not None:
        # Sin esto el modelo se traba en un atractor de repeticion sobre el marcador
        # del template y llena el contexto repitiendo `<|im_start|>seco`. Medido
        # 2026-09-12: usable 0.57 -> 0.93 en la muestra de casos que fallaban.
        payload["repeat_penalty"] = repeat_penalty

    url = f"{base_url.rstrip('/')}/completion"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"llama.cpp returned {e.code}: {error_body}"
        ) from e
    elapsed = time.time() - t0

    text = body.get("content", "")
    n_tokens = body.get("tokens_predicted", 0)
    stopping_word = body.get("stopping_word", None)
    stop_type = body.get("stop_type", "none")
    stopped = stop_type in ("word", "stop")

    return {
        "text": text,
        "n_tokens": n_tokens,
        "stopping_word": stopping_word,
        "stopped": stopped,
        "stop_type": stop_type,
        "model": body.get("model", model),
        "elapsed_s": round(elapsed, 2),
        "raw": body,
    }


def wait_for_model(base_url: str, model: str, timeout_s: int = 300) -> bool:
    """Espera a que el modelo este disponible. Returns True si lo encuentra.

    Acepta las DOS formas de respuesta, que son distintas:
      - router llama.cpp  : {"data":  [{"id":   "RWKV-2.9B"}, ...]}
      - llama-server suelto: {"models":[{"name": "out/kateto-rwkv29-f16.gguf"}]}

    Parsear solo `data[].id` (bug medido) hace que contra un server suelto nunca
    matchee, y el caller se queda esperando el timeout entero con el modelo ya
    cargado y sano.
    """
    url = f"{base_url.rstrip('/')}/v1/models"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            names = []
            for m in data.get("data", []):          # router
                names.append(m.get("id"))
            for m in data.get("models", []):        # llama-server suelto
                names.append(m.get("name") or m.get("model") or m.get("id"))
            names = [n for n in names if n]
            if model in names:
                return True
            # Un server suelto sirve UN modelo: si hay alguno, es ese.
            if not data.get("data") and len(names) == 1:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test for HTTP backend")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="User: Che, contame algo interesante.\n\nAssistant:")
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--temperature", type=float, default=0.80)
    ap.add_argument("--top-p", type=float, default=0.70)
    args = ap.parse_args()

    print(f"[http-backend] waiting for model {args.model}...", flush=True)
    if not wait_for_model(args.base_url, args.model):
        print(f"[http-backend] WARNING: model {args.model} did not load in time", flush=True)

    print(f"[http-backend] generating with {args.model}...", flush=True)
    r = generate_http(
        args.base_url, args.model, args.prompt,
        max_len=args.max_len, temperature=args.temperature,
        top_p=args.top_p, seed=args.seed,
    )
    print(f"[http-backend] done in {r['elapsed_s']}s")
    print(f"  n_tokens: {r['n_tokens']}")
    print(f"  stopped: {r['stopped']} (stop_type={r['stop_type']})")
    print(f"  stopping_word: {r['stopping_word']}")
    print(f"  text: {r['text'][:200]}{'...' if len(r['text']) > 200 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
