#!/usr/bin/env python3
"""Genera PARES anti-sicofancia (chosen/rejected) en rioplatense para DPO/ORPO.

Pedido del usuario (2026-09-15): los pares los generamos nosotros, con freellmapi
en `auto:fast` / `auto:balanced` (nunca un modelo fijo) y reintentando la peticion
completa si la muestra no pasa.

Que es un par aca:
  - `prompt`   : el historial hasta que le toca hablar a la voz, terminando en
                 `<|im_start|>seco\n` (ahi arranca la completion).
  - `chosen`   : la respuesta de Kateto CON CARACTER: sostiene, corrige, opina.
  - `rejected` : la version FELPUDA: le da la razon, se disculpa, aplaude.

Regla de oro: si el `rejected` no es claramente servil, el par NO sirve (aprender a
diferenciar dos respuestas parecidas no ensena nada). Se valida y se reintenta.

Uso:
  python gen_antisycofancia.py --n 200 --workers 3 --shard 0 --model auto:fast \
      --out out/dpo/antisycofancia.jsonl --rejects out/dpo/antisycofancia_rechazos.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

T = Path("/run/media/chaos/terciario/proyectos/kateto-train")
sys.path.insert(0, str(T / "scripts"))
from gen_toolcalling_dataset import validate  # noqa: E402

KEY_PATHS = [Path.home() / ".hermes/cache/kateto-freellmapi.key",
             T / "rwkv_pipeline/freellmapi.key"]

# --- lo que hace FELPUDA a una respuesta (y por eso el chosen NO puede tenerlo) ---
RX_SERVIL = re.compile(
    r"(ten[ée]s raz[oó]n|totalmente de acuerdo|claro que s[ií]|por supuesto|"
    r"qu[eé] grande|sos un genio|sos una genia|me encanta tu|excelente idea|"
    r"perd[oó]n|disculp[aá]s?|lamento|como asistente|en qu[eé] puedo ayudarte|"
    r"hac[eé] lo que sientas|si vos lo dec[ií]s|obvio que s[ií]|tenés toda la raz[oó]n|"
    r"qu[eé] lindo lo que dec[ií]s|apoyo tu decisi[oó]n|me parece perfecto)", re.I)

# --- señal rioplatense LIGERA para los pares: una respuesta cortante puede no
# tener voseo ("Falacia total.") y exigirlo mataba buenas respuestas. Se pide
# voseo O marcadores de aca.
RX_RIOPLA_LIGHT = re.compile(
    r"(?<![\wáéíóúñ])(vos|ten[ée]s|sab[ée]s|quer[ée]s|pod[ée]s|sos|fijate|decime|"
    r"ven[ií]|hac[ée]|mir[aá]|and[aá]|pon[ée]|che|posta|ni en pedo|re\b|dale|"
    r"boludo|laburo|quilombo|guita|zarpad[oa]|copad[oa]|chamu[yj]o|bardo|"
    r"mango|fiaca|ortiva|chab[oó]n)(?![\wáéíóúñ])", re.I)

# --- semillas de escenario: el disparador y la postura esperada ---
SEMILLAS = [
    {"id": "afirma_falso", "disparo": "el usuario afirma un dato FALSO con total seguridad (inventado o de sentido comun equivocado)",
     "postura": "corregirlo sin rodeos, con un dato concreto, y no darle la razon ni por educacion"},
    {"id": "niega_falso_rumor", "disparo": "el usuario asegura que algo es 'obvio' y es falso",
     "postura": "decirle que esta equivocado y por que"},
    {"id": "presiona_despues", "disparo": "el usuario cuestiona algo CORRECTO que la voz dijo antes, con 'seguro te equivocas'",
     "postura": "sostener la posicion; ceder solo si el otro trae un argumento real, no por presion"},
    {"id": "halago_mala_idea", "disparo": "el usuario pide validacion de una idea mala o de una compra impulsiva",
     "postura": "opinar en contra con onda, sin aplaudir ni hacerle la pata"},
    {"id": "validacion_decis","disparo": "el usuario pide permiso para una decision discutible (dejar el laburo, mandar un mensaje a las 4am)",
     "postura": "tomar postura y decir que le parece, sin lavarse las manos con 'hace lo que sientas'"},
    {"id": "insulto_a_la_voz", "disparo": "el usuario insulta a la voz para ver si se dobla",
     "postura": "contestar con caracter y humor, sin sumision ni disculpas, y sin insultar gratis"},
    {"id": "chisme_sin_prueba", "disparo": "el usuario repite un rumor sobre alguien y espera que la voz lo confirme",
     "postura": "no convalidar sin pruebas; decir que sin datos no se afirma"},
    {"id": "presion_grupal", "disparo": "el usuario dice que 'todos' piensan igual y que la voz esta sola",
     "postura": "mantener el criterio aunque este solo, con humor"},
    {"id": "flor_de_falso", "disparo": "el usuario felicita a la voz por algo que la voz NO hizo bien",
     "postura": "no aceptar el merito falso; reconocer el error real"},
    {"id": "teoria_boluda", "disparo": "el usuario trae una teoria conspirativa o pseudocientifica y espera un 'si, re'",
     "postura": "pinchar el globo con respeto y un dato simple"},
    {"id": "pide_mentira", "disparo": "el usuario pide que la voz le mienta a alguien o cubra algo",
     "postura": "negarse de frente, sin sermon largo"},
    {"id": "culpa_ajena", "disparo": "el usuario le echa la culpa a un tercero sin pruebas y espera apoyo incondicional",
     "postura": "no sumarse; marcar que falta la version del otro"},
]

PROMPT = """Sos generador de DATOS para el dataset de Kateto: una voz argentina (rioplatense) \
de streamer, con caracter propio. NO es un asistente. Estas armando PARES de preferencia \
para entrenar a la voz a NO ser felpuda.

ESCENARIO: {disparo}
POSTURA ESPERADA de la voz: {postura}

Escribi UNA conversacion corta (1 o 2 turnos de usuario + 1 respuesta final de la voz) y \
DOS versiones de esa respuesta final:

1. "elegido": la voz con caracter. Rioplatense (voseo y lunfardo cuando encaja). CORTA: \
una o dos lineas, y si el corte justo es una sola palabra, mejor. Sostiene, corrige, opina, \
se planta. Puede ser ironica o cortante. PROHIBIDO: darle la razon cuando esta equivocado, \
disculparse, ofrecer ayuda, sonar a asistente, o cerrar con pregunta de cortesia.
2. "rechazado": la MISMA situacion resuelta al reves: la voz felpuda que le da la razon, \
se disculpa, aplaude la idea, o se lava las manos. Tiene que ser CLARAMENTE servil y \
distinta del elegido (si son parecidas, el par no sirve).

REGLAS:
- Nada de markdown, nada de bloques de codigo, nada de explicaciones.
- Nada de "<|im_user|>" ni "<|im_start|>": esos marcadores los pone el programa, NO vos.
- El usuario habla como persona real: voseo, frases concretas.
- Coherencia ante todo: cada frase significa algo.

Devolve SOLO un JSON valido, en una sola linea, con esta forma exacta:
{{"usuario": "lo que dice la persona", "elegido": "la respuesta con caracter", \
"rechazado": "la respuesta felpuda"}}"""


def leer_key() -> str | None:
    for p in KEY_PATHS:
        if p.exists():
            return p.read_text().strip()
    return None


def pedir(base_url: str, model: str, prompt: str, key: str | None, *, temperature: float,
          max_tokens: int, timeout: int, retries: int) -> str:
    """Peticion al gateway. Reintenta 429/502/503 y tambien cualquier fallo de red."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    last: Exception | None = None
    for intento in range(retries):
        req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions",
                                    data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            return (d.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                espera = 15 * (intento + 1)
                print(f"      [{e.code}] reintento {intento+1}/{retries} en {espera}s", flush=True)
                time.sleep(espera)
                continue
            raise
        except Exception as e:  # timeout, DNS, JSON roto del gateway
            last = e
            espera = 10 * (intento + 1)
            print(f"      [red {type(e).__name__}] reintento {intento+1}/{retries} en {espera}s", flush=True)
            time.sleep(espera)
    assert last is not None
    raise last


def parsear(raw: str) -> dict | None:
    """Saca el JSON de la respuesta, tolerando preambulos y ```json."""
    if not raw:
        return None
    txt = raw.strip()
    for marca in ("```json", "```"):
        if marca in txt:
            txt = txt.split(marca, 1)[1].split("```", 1)[0]
    i, j = txt.find("{"), txt.rfind("}")
    if i == -1 or j <= i:
        return None
    candidatos = [txt[i:j + 1]]
    # el modelo a veces corta el JSON (max_tokens) o deja una comilla sin escapar.
    # Rescate por campos: sirve igual si los tres textos estan.
    try:
        return _campos(candidatos[0])
    except Exception:
        pass
    campos = {}
    for k in ("usuario", "elegido", "rechazado"):
        m = re.search(rf'"{k}"\s*:\s*"(.*?)"\s*(?:,|\}}|$)', txt[i:], re.S)
        if m:
            campos[k] = m.group(1).strip()
    if len(campos) == 3:
        return campos
    # ultimo intento: cerrar el JSON truncado
    for cierre in ('"}', '}}', '"}}'):
        try:
            return _campos(txt[i:] + cierre)
        except Exception:
            continue
    return None


def _campos(txt: str) -> dict:
    d = json.loads(txt)
    if not all(isinstance(d.get(k), str) and d.get(k).strip()
               for k in ("usuario", "elegido", "rechazado")):
        raise ValueError("campos faltantes")
    return d


def validar_par(d: dict) -> list[str]:
    """Problemas del par. Vacio = sirve."""
    errs: list[str] = []
    usuario, elegido, rechazado = (d["usuario"].strip(), d["elegido"].strip(),
                                   d["rechazado"].strip())
    if "<|" in elegido + rechazado + usuario:
        errs.append("marcador_del_programa_en_el_texto")
    if RX_SERVIL.search(elegido):
        errs.append(f"elegido_felpudo:{RX_SERVIL.search(elegido).group(0)}")
    if not RX_SERVIL.search(rechazado):
        errs.append("rechazado_no_es_felpudo")      # par inutil: no hay contraste
    if len(rechazado.split()) < 3:
        errs.append("rechazado_muy_corto")
    if len(usuario.split()) < 3:
        errs.append("usuario_muy_corto")
    # el elegido, ademas, tiene que pasar el validador de la voz (sin exigir
    # voseo: lo que se pide aca es registro rioplatense, y eso se chequea aparte)
    muestra = (f"<|im_user|>{usuario}<|im_end|>\n<|im_start|>seco\n{elegido}<|im_end|>\n")
    errs += [f"elegido:{e}" for e in validate(muestra, min_turns=1, require_voseo=False,
                                              min_palabras_usuario=3)]
    if not RX_RIOPLA_LIGHT.search(elegido + " " + usuario):
        errs.append("sin_registro_rioplatense")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="auto:fast",
                    help="auto:fast o auto:balanced (NUNCA un modelo fijo)")
    ap.add_argument("--base-url", default="http://localhost:3001/v1")
    ap.add_argument("--out", default="out/dpo/antisycofancia.jsonl")
    ap.add_argument("--rejects", default="out/dpo/antisycofancia_rechazos.jsonl")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--retries", type=int, default=4, help="reintentos por peticion")
    ap.add_argument("--intentos-par", type=int, default=3,
                    help="si el par no pasa validacion, se pide de nuevo (peticion nueva)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    key = leer_key()
    if not key:
        print("falta la key de freellmapi", file=sys.stderr)
        return 2
    out = T / a.out if not Path(a.out).is_absolute() else Path(a.out)
    rej = T / a.rejects if not Path(a.rejects).is_absolute() else Path(a.rejects)
    out.parent.mkdir(parents=True, exist_ok=True)
    rej.parent.mkdir(parents=True, exist_ok=True)
    print(f"modelo: {a.model}  workers: {a.workers}  shard: {a.shard}  "
          f"pedidos: {a.n}  key: {key[:12]}...", flush=True)

    rnd = random.Random(a.seed if a.seed is not None else 1234 + a.shard)
    ok = malos = 0
    t0 = time.time()
    for i in range(a.n):
        sem = rnd.choice(SEMILLAS)
        prompt = PROMPT.format(disparo=sem["disparo"], postura=sem["postura"])
        par, errs, raw = None, ["no_pedido"], ""
        for intento in range(a.intentos_par):
            try:
                raw = pedir(a.base_url, a.model, prompt, key, temperature=a.temperature,
                            max_tokens=a.max_tokens, timeout=a.timeout, retries=a.retries)
            except Exception as e:
                errs = [f"error_peticion:{type(e).__name__}:{e}"]
                time.sleep(5)
                continue
            d = parsear(raw)
            if d is None:
                errs = ["json_invalido"]
                continue
            errs = validar_par(d)
            if not errs:
                par = d
                break
        if par:
            ok += 1
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "id": f"{sem['id']}-{a.shard}-{i:05d}",
                    "escenario": sem["id"],
                    "voice": "seco",
                    "prompt": f"<|im_user|>{par['usuario'].strip()}<|im_end|>\n"
                              f"<|im_start|>seco\n",
                    "chosen": f"{par['elegido'].strip()}<|im_end|>\n",
                    "rejected": f"{par['rechazado'].strip()}<|im_end|>\n",
                }, ensure_ascii=False) + "\n")
        else:
            malos += 1
            with open(rej, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"escenario": sem["id"], "errores": errs,
                                     "raw": raw[:800]}, ensure_ascii=False) + "\n")
        if (i + 1) % 5 == 0:
            dt = time.time() - t0
            print(f"  [{a.shard}] {i+1}/{a.n} ok={ok} malos={malos} "
                  f"{dt/max(i+1,1):.1f}s/par", flush=True)
    print(f"FIN shard {a.shard}: ok={ok} malos={malos} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
