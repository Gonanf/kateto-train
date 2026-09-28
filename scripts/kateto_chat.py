#!/usr/bin/env python
"""REPL para hablar con Kateto 2.9B desde la terminal.

Usa la configuracion VALIDADA (n=80, par matched contra el base):

    quant            Q8_0      (3.26 GB -- el f16 da igual pero pesa 5.93 GB)
    repeat_penalty   1.30      <- obligatorio: sin esto se traba repitiendo el
                                  marcador del template y se reporta como vacio
                                  (utilizable 60% -> 87.5% medido)
    temperature      0.80
    top_p            0.70
    stop (servidor)  solo fin de turno real (<|im_end|>, <|endoftext|>)

Uso:

    # arranca el server solo y chatea
    python scripts/kateto_chat.py --serve

    # contra un server ya levantado
    python scripts/kateto_chat.py --url http://127.0.0.1:54699

    # un solo prompt, sin REPL (util para probar rapido)
    python scripts/kateto_chat.py --serve --once "Che, que onda el Kun Aguero?"

Comandos dentro del REPL:
    /voice <nombre>   cambia el marcador de voz (seco, doktor, jane, ...)
    /rp <float>       cambia repeat_penalty
    /temp <float>     cambia temperature
    /raw              muestra las metricas de la ultima respuesta
    /salir            sale
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "rwkv_pipeline"))

from chat_template import render_chat

DEFAULT_GGUF = REPO / "out/quants/kateto29-Q8_0.gguf"
DEFAULT_PORT = 54699

# Marcadores de fin de turno: SOLO estos van al `stop` del servidor.
# Ojo: NO agregar `<|im_start|>` aca. Si el modelo cae al eco del template, su
# primer token ES ese marcador; el server cortaria ahi y devolveria texto vacio,
# borrando el diagnostico. Medido 2026-09-12.
SERVER_STOP = ["<|im_end|>", "<|endoftext|>"]
# Marcadores donde el truncado local corta la respuesta.
TRUNCATE_AT = ["<|im_end|>", "<|im_start|>", "\n<|im_user|>", "<|endoftext|>"]


def build_prompt(user_text: str, voice: str) -> str:
    return render_chat([], user_text, voice)


def truncate(text: str) -> tuple[str, str | None]:
    best_i, best_m = len(text), None
    for m in TRUNCATE_AT:
        i = text.find(m)
        if i != -1 and i < best_i:
            best_i, best_m = i, m
    return text[:best_i], best_m


def classify(text: str, prompt: str) -> str:
    """Clasifica la respuesta: "ok", "template_echo" o "prompt_echo".

    Logica duplicada a proposito de `eval_generation.classify_nonanswer` para que
    este REPL no importe torch ni el pipeline de eval: asi corre con el python3 del
    sistema, sin venv.

    Solo es eco de template si NO QUEDA NADA tras sacar los fragmentos. Un umbral
    de largo marcaba `"2"` (respuesta correcta a "2 + 2?") como no-respuesta, que es
    peor que no medir. `[...]` entra como fragmento: es un marcador del dataset.
    """
    t = (text or "").strip()
    rest = re.sub(r"<\|im_(?:start|end|user)\|>\s*\w*|<\|endoftext\|>|\[\.\.\.\]|>seco", "", t)
    for frag in ("<|im_start|>", "<|im_end|>", "<|im_user|>", ">seco",
                 "<|endoftext|>", "[...]"):
        rest = rest.replace(frag, "")
    rest = "".join(c for c in rest if not c.isspace())
    if not rest:
        return "template_echo"
    pw = set((prompt or "").lower().split())
    tw = t.lower().split()
    if tw and len(tw) > 3 and sum(1 for w in tw if w in pw) / len(tw) > 0.70:
        return "prompt_echo"
    return "ok"


def wait_health(url: str, tries: int = 90) -> bool:
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=3) as r:
                if b'"ok"' in r.read():
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_server(gguf: Path, port: int, ctx: int) -> subprocess.Popen:
    cmd = [
        "llama-server", "--host", "127.0.0.1", "--port", str(port),
        "--model", str(gguf), "--ctx-size", str(ctx),
        "--parallel", "1", "--no-warmup", "--log-verbosity", "1",
    ]
    print(f"[kateto] levantando servidor... ({gguf.name}, {ctx} ctx)")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_health(f"http://127.0.0.1:{port}"):
        proc.kill()
        raise SystemExit("[kateto] el servidor no levanto. Corre con --url si ya tenes uno.")
    print(f"[kateto] listo en http://127.0.0.1:{port}")
    return proc


def generate(url: str, prompt: str, args) -> dict:
    payload = {
        "prompt": prompt,
        "n_predict": args.max_len,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repeat_penalty": args.repeat_penalty,
        "seed": args.seed,
        "stream": False,
        "cache_prompt": False,
        "stop": SERVER_STOP,
    }
    req = urllib.request.Request(
        f"{url.rstrip('/')}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.loads(r.read())
    body["_elapsed"] = round(time.time() - t0, 2)
    body["_payload"] = payload
    return body


def main() -> int:
    ap = argparse.ArgumentParser(description="REPL para hablar con Kateto 2.9B")
    ap.add_argument("--gguf", default=str(DEFAULT_GGUF))
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--url", default=None, help="usar un server ya levantado")
    ap.add_argument("--serve", action="store_true", help="levantar el server")
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--voice", default="seco")
    ap.add_argument("--repeat-penalty", type=float, default=1.30)
    ap.add_argument("--temperature", type=float, default=0.80)
    ap.add_argument("--top-p", type=float, default=0.70)
    ap.add_argument("--max-len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--once", default=None, help="un solo prompt y sale")
    args = ap.parse_args()

    proc = None
    url = args.url
    if args.serve or not url:
        gguf = Path(args.gguf)
        if not gguf.exists():
            raise SystemExit(f"[kateto] no existe el GGUF: {gguf}\n"
                             f"         generalo con: llama-quantize out/kateto-rwkv29-f16.gguf "
                             f"{gguf} Q8_0")
        proc = start_server(gguf, args.port, args.ctx)
        url = f"http://127.0.0.1:{args.port}"

    try:
        if args.once:
            ask(url, args.once, args)
            return 0

        print(f"\nkateto 2.9B  |  voz={args.voice}  rp={args.repeat_penalty}  "
              f"temp={args.temperature}  top_p={args.top_p}")
        print("comandos: /voice <n>  /rp <f>  /temp <f>  /raw  /salir\n")
        last = None
        while True:
            try:
                line = input("vos> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            if line in ("/salir", "/exit", "/quit"):
                break
            if line.startswith("/voice "):
                args.voice = line.split(None, 1)[1].strip()
                print(f"  voz -> {args.voice}")
                continue
            if line.startswith("/rp "):
                args.repeat_penalty = float(line.split(None, 1)[1])
                print(f"  repeat_penalty -> {args.repeat_penalty}")
                continue
            if line.startswith("/temp "):
                args.temperature = float(line.split(None, 1)[1])
                print(f"  temperature -> {args.temperature}")
                continue
            if line == "/raw" and last:
                print(json.dumps({k: v for k, v in last.items()
                                  if k not in ("_payload", "_display")},
                                 indent=2, ensure_ascii=False)[:1200])
                continue
            last = ask(url, line, args)

        if last:
            pass
    finally:
        if proc:
            proc.kill()

    return 0


def ask(url: str, user_text: str, args) -> dict:
    prompt = build_prompt(user_text, args.voice)
    body = generate(url, prompt, args)
    raw = body.get("content") or ""
    text, stop_m = truncate(raw)
    text = text.strip()
    kind = classify(text, user_text)

    if kind == "ok":
        out = f"\nkateto> {text}\n"
    else:
        # No lo escondemos: es el modo de fallo que hay que ver para diagnosticar.
        out = (f"\nkateto> [NO RESPONDIO: {kind}]\n"
               f"        crudo: {raw[:200]!r}\n")
    out += (f"        [{body.get('tokens_predicted')} tok, {body['_elapsed']}s, "
            f"stop={body.get('stop_type')}{'/' + body['stopping_word'] if body.get('stopping_word') else ''}]\n")
    print(out)
    body["_display"] = out
    body["_kind"] = kind
    body["_truncated"] = text
    _ = stop_m
    return body


if __name__ == "__main__":
    raise SystemExit(main())
