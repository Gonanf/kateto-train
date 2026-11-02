#!/usr/bin/env python3
"""Eval MULTITURNO — instrumenta la degradacion por arrastre de estado/turno.

QUE MIDE (y NO arregla)
-----------------------
El reporte del usuario: "el 1.5B responde bien con 1 mensaje pero se degrada
rapido si hay mas de uno". Hoy no hay ninguna medicion de eso: la evaluacion de
comportamiento es de un turno. Este script corre N conversaciones de N turnos y
reporta, POR TURNO, la salud de la respuesta y del estado interno.

Los DOS modos que pide el brief, y por que los dos hacen falta:
  * `--state-policy carry` (default): el estado del RNN se arrastra entre turnos.
  * `--state-policy reset`: el estado se resetea a cero antes de cada turno.

En AMBOS modos se pasa el MISMO historial de texto (el unico texto que cambia es
`--ctx-truncate`). La unica variable es el estado. Si con `reset` no se degrada y
con `carry` si, la causa esta en el estado que se arrastra; si degrada en los dos,
la causa es el contexto que se pasa de ctx512.

NOTA sobre la trampa de triton (K=8 / padding a multiplo de 64 / `logits[0, len-1]`):
esa trampa es de la via VECTORIZADA (RWKV-PEFT `RWKV7.forward_normal`, la que usa
`rwkv_pipeline/opd_rollout.py`, con WKV=triton y chunks de K=8). Este script usa
la via de inferencia de `rwkv_pipeline/infer_kateto.py`, cuyo `RWKV_RNN.forward`
es un RNN POR TOKEN (`forward(token, state) -> logits`), sin dimension T y sin
chunks: no hay `logits[0, len(ids)-1]` que leer porque no hay logits
bidimensionales. Por eso no se aplica padding aqui; se documenta igual para que
nadie lo porte por error a la otra via.

Por que NO se reusa `KatetoInferenceEngine.generate()` tal cual: ese metodo
resetea el estado en cada llamada y no lo devuelve, asi que no puede arrastrarlo
entre turnos — que es justamente lo que hay que medir. Aca se reusan las mismas
primitivas que usa (`engine._forward`, `engine.tokenizer`, `sample_logits`,
`ends_with`, `count_consecutive_ngram_repeats`) y solo se agrega el hilo del
estado a lo largo de los turnos.

El A/B de ROSA (`--rosa`): el forward va por `engine._forward` y no por
`engine.model.forward` porque `_forward` es el unico punto donde la rama ROSA se
mezcla (`infer_kateto.py:404`). Apagado, ese metodo devuelve `model.forward` sin
tocarlo, asi que el brazo OFF es la misma linea base bit a bit. El buffer KV de
ROSA (`engine.rosa_memory`) se arrastra con la misma politica que el estado del
RNN. OJO: los pesos de ROSA son RANDOM, sin entrenar — el A/B mide el ruido de
una capa no entrenada, no el recall de ROSA entrenada.

Uso (los procesos de background de este host corren en un cgroup capado a 4 GiB y
un checkpoint cargado ahi muere con exit 137; `systemd-run` con MemoryMax
infinity es OBLIGATORIO, no cosmetico):

  systemd-run --user --scope -p MemoryMax=infinity --unit=kateto-mt-smoke \
    ./venv-unsloth-qwen/bin/python -u scripts/eval_multiturn.py \
    --model out/rwkv_kateto_base_plus_0.4b/hist/rwkv-1-20260928-1739.pth \
    --conversaciones 1 --turns 3 --device cuda --tag smoke
"""
from __future__ import annotations

# ROCm/torch pre-import env — must precede `import torch` (ver infer_kateto.py).
import os
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "rwkv_pipeline"))

from chat_template import render_chat
from rwkv_pipeline.infer_kateto import (
    KatetoInferenceEngine, RWKV_TOKENIZER, sample_logits, SILENCE_SENTINELS,
)
from rwkv_pipeline._stop_helpers import ends_with, count_consecutive_ngram_repeats

# Marcadores del template. Se escriben como literales aca (no se importan de
# chat_template) para que el corte por ids no dependa de como este definido el set.
IM_END = "<|im_end|>"
IM_USER = "<|im_user|>"

DEFAULT_PROMPT_SET = PROJECT / "data/multiturno_set.jsonl"

# Umbrales del brief. Son contrato del instrumento, no parametros de estilo.
REP8_DEGENERADO = 0.5     # > esto = respuesta degenerada en loop
ECO_FRACCION = 0.70      # > esto = eco del prompt


# --------------------------------------------------------------------------- #
# Metricas puras (sin torch) — testeables con python3 pelado
# --------------------------------------------------------------------------- #

_RX_PALABRA = re.compile(r"\w+", re.UNICODE)


def palabras(text: str) -> list[str]:
    """Minusculas, sin acentuar: el eco y las repeticiones no distinguen 'Pizza' de 'pizza'."""
    return [w.lower() for w in _RX_PALABRA.findall(text or "")]


def rep8(text: str) -> float:
    """Fraccion de 8-gramas de PALABRA repetidos dentro de la respuesta.

    1 - (8-gramas unicos / 8-gramas totales). Con menos de 8 palabras no hay
    ningun 8-grama -> 0.0. Con exactamente 8 hay 1 solo, que es unico por
    definicion -> 0.0. Asi que un texto corto nunca se marca como degenerado por
    repeating: la metrica no mide longitud, mide repeticion.
    """
    w = palabras(text)
    if len(w) < 8:
        return 0.0
    grams = [tuple(w[i:i + 8]) for i in range(len(w) - 7)]
    return 1.0 - (len(set(grams)) / len(grams))


def eco_del_prompt(respuesta: str, prompt: str) -> bool:
    """True si >70% de las palabras de la respuesta YA estaban en el prompt.

    Se mide sobre el prompt REAL que se le paso al modelo (ya truncado por
    --ctx-truncate), que es lo unico que el modelo puede ver.
    """
    r = palabras(respuesta)
    if not r:
        return False
    p = set(palabras(prompt))
    return (sum(1 for w in r if w in p) / len(r)) > ECO_FRACCION


def es_vacia(text: str) -> bool:
    """Respuesta sin contenido util: vacia, solo whitespace, o un centinela de silencio."""
    t = (text or "").strip()
    return (not t) or (t in SILENCE_SENTINELS)


def limpiar_think(text: str) -> str:
    """`--thinking off`: deja solo la respuesta visible.

    Si el modelo abrio un bloque `<think>` y nunca lo cerro, se corta en la
    apertura (no hay respuesta visible que medir). Si lo cerro, se queda con lo
    que viene despues del `</think>`.
    """
    t = text or ""
    if "</think>" in t:
        return t.split("</think>", 1)[1]
    if "<think>" in t:
        return t.split("<think>", 1)[0]
    return t


def _selfcheck() -> None:
    """Asserts sobre las metricas puras. Corre solo con --selfcheck.

    Es el chequeo minimo que falla si la logica de repeticion/eco/vacia se rompe:
    no hay fixtures ni framework, solo asserts sobre casos que tienen que dar
    True/False Known.
    """
    # rep8: loop obvious -> ~1.0 ; texto sin repeticion -> 0.0
    # 36 palabras = 29 8-gramas, solo 3 unicos -> 1-3/29 = 0.8966, NO ~1.0.
    loop = " ".join(["el gato come"] * 12)
    assert rep8(loop) > 0.85, rep8(loop)
    assert rep8("el gato come el perro bebe agua corriendo todo el dia") == 0.0
    assert rep8("corto") == 0.0                       # < 8 palabras -> 0.0
    assert 0.0 <= rep8("a b c d e f g h a b c d e f g h i") < 1.0
    # eco
    assert eco_del_prompt("la pizza y el queso", "Che, se me cayo la pizza del microondas y quedo todo el piso cubierto de queso") is True
    assert eco_del_prompt("teoria de ondas y politica internacional", "Che, se me cayo la pizza del microondas") is False
    assert eco_del_prompt("", "lo que sea") is False
    # vacia
    assert es_vacia("") is True and es_vacia("   \n ") is True
    assert es_vacia("<|wait|>") is True
    assert es_vacia("hola") is False
    # think
    assert limpiar_think("<think>razonando</think>la respuesta") == "la respuesta"
    assert limpiar_think("<think>nunca cerro") == ""
    assert limpiar_think("directa") == "directa"
    print("selfcheck OK: rep8, eco, vacia, think")


# --------------------------------------------------------------------------- #
# Estado del RNN
# --------------------------------------------------------------------------- #

def state_fingerprint(state) -> tuple[str, float]:
    """(sha256, norma L2) del estado completo del RNN.

    El estado es una `List[Tensor]` de `n_layer*3` entradas (x, att_state, ffn).
    Se hashea y se mide en float32 sobre CPU: en bf16/GPU los bytes no son
    reproducibles entre corridas, y el hash tiene que servir para comparar turnos
    DENTRO de la misma corrida (y entre corridas de la misma config).

    `state_norm` (la norma L2 de todo el estado junto) es la columna que mas dice:
    si con ctx512 el estado crece turno a turno sin techo, ahi esta la hipotesis
    del bug, y la tabla lo muestra.
    """
    import torch  # local: las metricas puras de arriba corren sin torch

    h = hashlib.sha256()
    sq = 0.0
    for t in state:
        if t is None:
            h.update(b"\x00none")
            continue
        f = t.detach().to(device="cpu", dtype=torch.float32).contiguous()
        b = f.numpy().tobytes()
        h.update(b)
        sq += float((f * f).sum())
    return h.hexdigest(), sq ** 0.5


# --------------------------------------------------------------------------- #
# Generacion de UN turno, reusando las primitivas de infer_kateto
# --------------------------------------------------------------------------- #

STOP_MARKERS_TEXTO = ["<tool_call>", IM_END, IM_USER, "\n\nUser:", "\nUser:",
                      "\n\nAssistant:", "\nAssistant:", "</tool_call>"]


def generar_turno(engine, state, prompt_text: str, user_text: str, max_tokens: int,
                  temperature: float, top_p: float, thinking_off: bool,
                  alpha_presence: float, alpha_frequency: float,
                  ngram_n: int, ngram_repeat_max: int):
    """Forward del prompt + sampling. Devuelve (texto, state, stop_reason, n_tokens).

    `state` entra y sale (se arrastra entre turnos cuando la politica es `carry`).
    """
    import torch

    tok = engine.tokenizer
    im_end_ids = tok.encode(IM_END)
    im_user_ids = tok.encode(IM_USER)

    # `engine._forward`, no `model.forward`: `_forward` es el unico punto donde la
    # rama ROSA se mezcla (infer_kateto.py:404), y con ROSA apagado devuelve
    # `model.forward` intacto (:412-414) -- por eso el brazo OFF es la misma base.
    fwd = engine._forward

    prompt_ids = tok.encode(prompt_text)
    out = None
    for t in prompt_ids:
        out, state = fwd(t, state)

    # OJO: `sample_logits(temperature=0)` NO es greedy. Miremos su cuerpo: con
    # temperature=0 el `if temperature > 0` no divide, pero igual cae en
    # `torch.multinomial(probs)` -> eso es muestreo ALEATORIO sobre la
    # distribucion completa. Para temp 0 hay que pedir argmax a mano, que es lo
    # que hace esta rama. Sin esto, "determinista" seria mentira.
    greedy = temperature <= 0.0

    # Penalty de ocurrencia inicializado con el prompt del usuario: es lo que ya
    # hace infer_kateto.generate() al servir (desincentiva el eco). Se mantiene
    # para que el eval mida la via de servicio real, no una variante. OJO: se
    # inicializa con el mensaje del USUARIO, no con el prompt formateado completo
    # -- es lo que hace generate(); con el historial entero el penalty seria otra
    # politica y las respuestas no serian comparables con las del servicio.
    occurrence = {t: 1 for t in tok.encode(user_text.strip()) if t not in (0, 11, 61)}

    gen: list[int] = []
    stop_reason = "max_tokens"
    for _ in range(max_tokens):
        if greedy:
            t = int(torch.argmax(out))
        else:
            t = sample_logits(out.clone(), temperature=temperature, top_p=top_p,
                              occurrence=occurrence,
                              alpha_presence=alpha_presence,
                              alpha_frequency=alpha_frequency)
        if t == 0:
            stop_reason = "eos"
            break
        gen.append(t)
        occurrence[t] = occurrence.get(t, 0) + 1
        # Corte por ids (mismo criterio que infer_kateto.generate).
        if ends_with(gen, im_end_ids) or ends_with(gen, im_user_ids):
            stop_reason = "id_end"
            break
        # Guarda de repeticion: el ultimo n-grama se repitio ngram_repeat_max
        # veces seguidas -> degenero en loop.
        if count_consecutive_ngram_repeats(gen, ngram_n) + 1 >= ngram_repeat_max:
            stop_reason = "repetition"
            break
        texto = tok.decode(gen)
        if any(m in texto for m in ("\nUser:", "\n\nUser:", "\nAssistant:",
                                    "\n\nAssistant:", "<tool_call>", "</tool_call>")):
            stop_reason = "marker"
            break
        out, state = fwd(t, state)

    texto = tok.decode(gen)
    for m in STOP_MARKERS_TEXTO:
        if m in texto:
            texto = texto.split(m)[0]
    texto = texto.strip()
    if thinking_off:
        texto = limpiar_think(texto).strip()
    # NO se sustituye la respuesta vacia por el fallback de infer_kateto: el
    # instrumento tiene que poder distinguir "no dijo nada" de "dijo algo".
    return texto, state, stop_reason, len(gen)


# --------------------------------------------------------------------------- #
# Prompt set
# --------------------------------------------------------------------------- #

def cargar_prompt_set(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"ERROR: no existe el prompt set: {path}")
    convs = []
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "turnos" not in d or not isinstance(d["turnos"], list) or not d["turnos"]:
                raise SystemExit(f"ERROR: {path}:{ln} no tiene 'turnos' con contenido")
            convs.append({"id": d.get("id", f"linea{ln}"), "turnos": list(d["turnos"])})
    if not convs:
        raise SystemExit(f"ERROR: {path} no tiene conversaciones")
    return convs


# --------------------------------------------------------------------------- #
# Corrida
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Eval multiturno: degradacion por arrastre de estado vs contexto",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", required=False, default=None,
                    help="Ruta al checkpoint .pth (merged base+fine-tune)")
    ap.add_argument("--turns", type=int, default=8, help="Turnos por conversacion")
    ap.add_argument("--conversaciones", type=int, default=None,
                    help="Cuantas conversaciones del set usar (default: todas)")
    ap.add_argument("--prompt-set", default=str(DEFAULT_PROMPT_SET),
                    help="JSONL de conversaciones")
    ap.add_argument("--out-dir", default=str(PROJECT / "out/multiturno"),
                    help="Directorio de salida")
    ap.add_argument("--tag", default="mt", help="Tag de los archivos de salida")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--thinking", default="off", choices=["off", "on"],
                    help="off = se descarta el bloque <think> y se mide la respuesta visible")
    ap.add_argument("--temperature", type=float, default=0.80,
                    help="Sampling. Default 0.80 = el contrato que ya usan "
                         "eval_generation.py y probe_format_generation()")
    ap.add_argument("--top-p", type=float, default=0.70,
                    help="Solo se usa si --temperature > 0")
    ap.add_argument("--seed", type=int, default=1337,
                    help="Seed de torch (PROBE_SEED del repo). Con sampling esto es lo que "
                         "hace la corrida REPRODUCIBLE; greedy (--temperature 0) es "
                         "determinista sin seed pero esta MEDIDO como patologico en el 0.4B: "
                         "argmax elige '<' (logit 6.6, primer byte de todos los marcadores) y "
                         "arranca '<|no me cajahajaja'. Ver docs/eval-multiturno.md")
    ap.add_argument("--state-policy", default="carry", choices=["carry", "reset"],
                    help="carry = el estado se arrastra entre turnos; reset = a cero cada turno")
    ap.add_argument("--rosa", action="store_true",
                    help="Encender la rama ROSA del head para el brazo ON del A/B. Apagado = "
                         "identico bit a bit a la linea base: con rosa=None, engine._forward() "
                         "devuelve engine.model.forward() sin mas. SIN --rosa-head los pesos "
                         "de ROSA son RANDOM sin entrenar, asi que ese A/B mide el ruido que "
                         "mete una capa no entrenada, NO el recall de ROSA con pesos buenos.")
    ap.add_argument("--rosa-head", default="",
                    help="Head ROSA entrenado (rosa_train_corto: out/rosa_head_corto/"
                         "rosa_head.pth). Sin esto, --rosa corre con pesos al azar. El motor "
                         "aborta si el head fue entrenado con otra arch o vocab.")
    ap.add_argument("--ctx-truncate", type=int, default=512,
                    help="Cuantos tokens de historial se le pasan al modelo")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="Tope de tokens generados por turno")
    ap.add_argument("--ngram-n", type=int, default=3, help="n de la guarda de repeticion")
    ap.add_argument("--ngram-repeat-max", type=int, default=12)
    ap.add_argument("--alpha-presence", type=float, default=0.6)
    ap.add_argument("--alpha-frequency", type=float, default=0.5)
    ap.add_argument("--voice", default="seco", help="Nombre de voz (solo informativo: el "
                                                   "estado no se carga, el .pth ya esta merged)")
    ap.add_argument("--selfcheck", action="store_true",
                    help="Corre los asserts de las metricas puras y sale")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return 0

    if not args.model:
        ap.error("--model es obligatorio (ruta al .pth)")
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"ERROR: no existe el checkpoint: {model_path}")

    convs = cargar_prompt_set(Path(args.prompt_set))
    if args.conversaciones is not None:
        if args.conversaciones < 1:
            raise SystemExit("ERROR: --conversaciones debe ser >= 1")
        convs = convs[:args.conversaciones]

    if args.turns < 1:
        raise SystemExit("ERROR: --turns debe ser >= 1")

    import torch
    if args.temperature > 0:
        torch.manual_seed(args.seed)

    vocab = PROJECT / "RWKV-PEFT/rwkv_vocab_v20230424.txt"
    if not vocab.exists():
        vocab = PROJECT / "RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt"
    if not vocab.exists():
        raise SystemExit(f"ERROR: no se encontro el vocab de RWKV (probes: {vocab})")

    print(f"[mt] modelo      : {model_path}")
    print(f"[mt] prompt set  : {args.prompt_set} ({len(convs)} conversaciones)")
    print(f"[mt] turnos      : {args.turns}   state-policy={args.state_policy}  "
          f"ctx-truncate={args.ctx_truncate}")
    print(f"[mt] sampling    : {'greedy (argmax)' if args.temperature <= 0 else f'temp={args.temperature} top_p={args.top_p} seed={args.seed}'}")
    print(f"[mt] thinking    : {args.thinking}   max-tokens={args.max_tokens}")

    t_load = time.time()
    engine = KatetoInferenceEngine(str(model_path), str(vocab), device=args.device,
                                   rosa=args.rosa, rosa_head=args.rosa_head)
    print(f"[mt] device efectivo: {engine.device}  (pedido: {args.device})")
    print(f"[mt] carga en {time.time() - t_load:.1f}s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jl_path = out_dir / f"{args.tag}.jsonl"
    js_path = out_dir / f"{args.tag}.json"

    filas: list[dict] = []
    t0 = time.time()
    for ci, conv in enumerate(convs, 1):
        print(f"\n[mt] --- conversacion {ci}/{len(convs)}: {conv['id']} ---")
        # En `carry` el estado arranca en cero y se ARRASTRA; en `reset` se
        # reconstruye en cero antes de cada turno. Es la unica diferencia entre
        # los dos modos: el texto que se pasa es el mismo.
        state = engine.build_initial_state(args.voice)
        # El buffer KV de ROSA se arrastra con la MISMA politica que el estado del
        # RNN: `carry` lo acumula entre turnos (que es justo lo que ROSA propone),
        # `reset` lo tira cada turno. Sin esto el A/B compararia memoria acumulada
        # contra memoria vacia y no diria nada.
        engine.rosa_memory = None
        historial: list[tuple[str, str]] = []
        state_hash_previo = None

        for ti in range(1, args.turns + 1):
            user = conv["turnos"][ti - 1]
            if args.state_policy == "reset":
                state = engine.build_initial_state(args.voice)
                engine.rosa_memory = None

            # Mismo texto en los dos modos: historial cerrado + turno abierto.
            texto_prompt = render_chat(historial, user, args.voice)
            ids_prompt = engine.tokenizer.encode(texto_prompt)
            if len(ids_prompt) > args.ctx_truncate:
                texto_prompt = engine.tokenizer.decode(ids_prompt[-args.ctx_truncate:])
                ids_prompt = ids_prompt[-args.ctx_truncate:]

            t_turn = time.time()
            respuesta, state, stop_reason, n_gen = generar_turno(
                engine, state, texto_prompt, user, args.max_tokens,
                args.temperature, args.top_p, args.thinking == "off",
                args.alpha_presence, args.alpha_frequency,
                args.ngram_n, args.ngram_repeat_max)
            dt = time.time() - t_turn

            h, norm = state_fingerprint(state)
            r8 = rep8(respuesta)
            vacia = es_vacia(respuesta)
            eco = eco_del_prompt(respuesta, texto_prompt)
            fila = {
                "conversacion": conv["id"],
                "turno": ti,
                "len_chars": len(respuesta),
                "n_tokens_gen": n_gen,
                "prompt_tokens": len(ids_prompt),
                "rep8": round(r8, 4),
                "vacia": vacia,
                "eco_del_prompt": eco,
                "degen": bool(r8 > REP8_DEGENERADO or vacia or eco),
                "state_hash": h,
                "state_cambio": (h != state_hash_previo) if state_hash_previo is not None else False,
                "state_norm": round(norm, 4),
                "stop_reason": stop_reason,
                "policy": args.state_policy,
                "rosa": bool(args.rosa),
                "rosa_head": bool(args.rosa_head),
                "ctx_truncate": args.ctx_truncate,
                "t_s": round(dt, 1),
                "texto": respuesta[:400],
            }
            filas.append(fila)
            state_hash_previo = h

            print(f"[mt] t{ti} len={fila['len_chars']:4d} tok={n_gen:3d} "
                  f"rep8={r8:.3f} vacia={int(vacia)} eco={int(eco)} "
                  f"|state|={norm:9.1f} stop={stop_reason} ({dt:.1f}s)")

            # La respuesta del modelo pasa a ser historial del proximo turno.
            if not vacia:
                historial.append((user, respuesta))

    with jl_path.open("w", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")

    resumen = construir_resumen(filas, args, model_path, time.time() - t0)
    with js_path.open("w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print(f"\n[mt] wrote {jl_path} ({len(filas)} lineas)")
    print(f"[mt] wrote {js_path}")
    print("\n[mt] === RESUMEN ===")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))
    return 0


def construir_resumen(filas, args, model_path, wall_s) -> dict:
    """Resumen por modelo.

    Denominadores: `tasa_utilizable` y `tasa_rep8_alto` van SOBRE UTILIZABLES (la
    rep8 alta es lo que se mide *dentro* de lo sano). `tasa_vacia` y `tasa_eco` van
    sobre el TOTAL: `vacia` y `eco` fuerzan `degen`, asi que sus filas quedan
    FUERA de `utilizables` — dividirlas por `len(utilizables)` las contaba dos
    veces y daba tasas >1 (medido: 1.2381 y 2.3333).
    """
    turnos = sorted({f["turno"] for f in filas})
    total = len(filas)
    utilizables = [f for f in filas if not f["degen"]]

    rep8_por_turno = {}
    n_util_por_turno = {}
    for t in turnos:
        del_t = [f for f in filas if f["turno"] == t]
        util_t = [f for f in del_t if not f["degen"]]
        n_util_por_turno[t] = len(util_t)
        rep8_por_turno[t] = round(statistics.fmean([f["rep8"] for f in util_t]), 4) if util_t else None

    norm_por_turno = {
        t: round(statistics.fmean([f["state_norm"] for f in filas if f["turno"] == t]), 2)
        for t in turnos
    }

    # Primer turno donde la conversacion se rompe. None = nunca se rompio.
    por_conv = {}
    for f in filas:
        if f["degen"] and por_conv.get(f["conversacion"]) is None:
            por_conv[f["conversacion"]] = f["turno"]

    rotos = [t for t in por_conv.values() if t is not None]

    lens = [f["len_chars"] for f in utilizables]
    if lens:
        q1, q3 = statistics.quantiles(lens, n=4, method="inclusive")[0], \
                 statistics.quantiles(lens, n=4, method="inclusive")[2]
        len_median = round(statistics.median(lens), 1)
        len_iqr = round(q3 - q1, 1)
    else:
        len_median, len_iqr = None, None

    return {
        "model": str(model_path),
        "tag": args.tag,
        "policy": args.state_policy,
        "rosa": bool(args.rosa),
        "rosa_head": args.rosa_head or "",
        "ctx_truncate": args.ctx_truncate,
        "turnos": args.turns,
        "conversaciones": len({f["conversacion"] for f in filas}),
        "sampling": "greedy (argmax)" if args.temperature <= 0 else
                    f"temp={args.temperature} top_p={args.top_p} seed={args.seed}",
        "thinking": args.thinking,
        # Primer turno en el que ALGUNA conversacion se degrada (el mas temprano).
        "turno_max_sano": min(rotos) if rotos else None,
        "turno_max_sano_por_conversacion": por_conv,
        "conversaciones_sanas": sum(1 for v in por_conv.values() if v is None),
        "rep8_por_turno": rep8_por_turno,
        "utilizables_por_turno": n_util_por_turno,
        "state_norm_por_turno": norm_por_turno,
        "len_median": len_median,
        "len_iqr": len_iqr,
        "utilizables": len(utilizables),
        "total": total,
        "tasa_utilizable": round(len(utilizables) / total, 4) if total else None,
        "tasa_vacia": round(sum(1 for f in filas if f["vacia"]) / total, 4) if total else None,
        "tasa_eco": round(sum(1 for f in filas if f["eco_del_prompt"]) / total, 4) if total else None,
        "tasa_rep8_alto": round(sum(1 for f in utilizables if f["rep8"] > REP8_DEGENERADO) / len(utilizables), 4) if utilizables else None,
        "wall_s": round(wall_s, 1),
        "salida_jsonl": str(Path(args.out_dir) / f"{args.tag}.jsonl"),
    }


if __name__ == "__main__":
    raise SystemExit(main())