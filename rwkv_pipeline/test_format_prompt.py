#!/usr/bin/env python3
"""Verifica el formato del prompt y los centinelas de infer_kateto.

Sin GPU y sin modelo: solo tokenizer + formato. Correr con el venv del repo:

    venv/bin/python rwkv_pipeline/test_format_prompt.py
"""
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "rwkv_pipeline"))

import infer_kateto as ik  # noqa: E402  (importa torch, no levanta modelo)

TOK = ik.RWKV_TOKENIZER(str(PROJECT / "RWKV-PEFT/rwkv_vocab_v20230424.txt"))
FMT = ik.KatetoInferenceEngine.format_chat_prompt


def _check_formato():
    # (a1) sin historial: el turno actual = prefijo abierto del formato del dataset
    #      "<|im_user|>{usuario}<|im_end|>\n<|im_start|>seco\n{respuesta}<|im_end|>"
    sin_hist = FMT(None, "Hola, Kateto", "none", None, 10)
    esperado = "<|im_user|>Hola, Kateto<|im_end|>\n<|im_start|>seco\n"
    assert sin_hist == esperado, f"\n got: {sin_hist!r}\nwant: {esperado!r}"
    print("ok (a1) sin historial :", repr(sin_hist))

    # (a2) con historial: cada turno pasado se serializa igual que una muestra del
    #      dataset (cerrada con <|im_end|>), y el turno actual queda abierto.
    con_hist = FMT(None, "Hola", "none", [("u1", "a1"), ("u2", "a2")], 10)
    esperado_h = ("<|im_user|>u1<|im_end|>\n<|im_start|>seco\na1<|im_end|>"
                  "<|im_user|>u2<|im_end|>\n<|im_start|>seco\na2<|im_end|>"
                  "<|im_user|>Hola<|im_end|>\n<|im_start|>seco\n")
    assert con_hist == esperado_h, f"\n got: {con_hist!r}\nwant: {esperado_h!r}"
    print("ok (a2) con historial :", repr(con_hist))

    # (a3) contra una linea real del dataset plus (el formato con <|im_end|> tras
    #      el usuario, el que vio el modelo entrenado). El historial debe renderizar
    #      la muestra byte a byte y el turno actual continuar el mismo formato.
    plus = PROJECT / "data/rwkv_kateto_base_train_plus.jsonl"
    linea = None
    with open(plus, encoding="utf-8") as fh:
        for l in fh:
            t = json.loads(l)["text"]
            if "<|im_end|>\n<|im_start|>seco\n" in t:
                linea = t
                break
    assert linea is not None, "no se encontro una muestra con el formato <|im_end|> en el plus"
    u_full, a_full = linea.split("<|im_end|>\n<|im_start|>seco\n")
    user, answer = u_full[len("<|im_user|>"):], a_full[: -len("<|im_end|>")]
    assert linea == f"<|im_user|>{user}<|im_end|>\n<|im_start|>seco\n{answer}<|im_end|>"
    rebuilt = FMT(None, "siguiente turno", "none", [(user, answer)], 10)
    esperado_r = linea + "<|im_user|>siguiente turno<|im_end|>\n<|im_start|>seco\n"
    assert rebuilt == esperado_r, f"\n got: {rebuilt!r}\nwant: {esperado_r!r}"
    print("ok (a3) linea real     :", repr(linea[:70]) + "...")

    # (a4) el historial se corta a max_history (los mas recientes)
    cortado = FMT(None, "p", "none", [(f"u{i}", f"a{i}") for i in range(3)], 2)
    assert "<|im_user|>u0<|im_end|>" not in cortado and "<|im_user|>u1<|im_end|>" in cortado
    print("ok (a4) max_history    :", repr(cortado))


def _check_centinelas():
    # (b) los centinelas se encodean en varios tokens (7 y 5) — no hay id unico,
    #     asi que el hack viejo de penalizar UN logit no puede apuntarlos.
    nr, wait = TOK.encode("<|no_response|>"), TOK.encode("<|wait|>")
    print("ids <|no_response|> :", nr)
    print("ids <|wait|>        :", wait)
    assert len(nr) > 1 and len(wait) > 1, "el centinela no es un token unico"
    assert TOK.decode(nr) == "<|no_response|>" and TOK.decode(wait) == "<|wait|>"
    # el id que penalizaba el hack viejo (61) es el '<' compartido por TODOS los
    # marcadores, no el centinela: penalizarlo no selecciona <|no_response|>.
    assert TOK.decode([61]) == "<", f"id 61 deberia ser '<', es {TOK.decode([61])!r}"
    play = TOK.encode("<play>")
    print("ids <play>          :", play, "(comparte el 61)")
    print("ok (b): centinela multitoken, sin id unico; el id 61 es '<' (prefijo de todos los marcadores)")


def _check_argparse():
    # (c) el flag --allow_no_response existe en el argparse real del CLI.
    out = subprocess.run(
        [sys.executable, str(PROJECT / "rwkv_pipeline/infer_kateto.py"), "--help"],
        capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert "--allow_no_response" in out.stdout, out.stdout
    print("ok (c): --allow_no_response presente en el CLI")


if __name__ == "__main__":
    _check_formato()
    _check_centinelas()
    _check_argparse()
    print("\nTODO OK")