#!/usr/bin/env python
"""Conversaciones LARGAS de Kateto desde el corpus real de transcripts.

Una muestra = UNA conversacion continua larga (meta >=3000 tokens, 12-60 turnos)
donde seco dialoga sobre el contenido real de un transcript. Es lo contrario de
gen_toolcalling_dataset.py: aca el largo es el requisito, no un accidente.

Camino que funciona (medido en este repo): turnwise, un turno por llamada con el
historial como contexto, transcript armado en Python. El teacher no arma la
conversacion (devuelve un turno y nunca el ).

El largo lo decide un CRITERIO, no un numero fijo (ver decidir_corte): se corta
al agotarse el material, cuando el chat tira <FIN>, cuando el juez dice SI, o al
llegar a --max-ex; nunca antes de --min-ex.

Uso:
    python scripts/gen_conversaciones_largas.py --dry-run
    python scripts/gen_conversaciones_largas.py --n 4 --teacher http
    python scripts/gen_conversaciones_largas.py --check-data out/dataset/largas/largas_v1.jsonl

El validador base se REUSA de gen_toolcalling_dataset (formato, voseo, tuteo,
spanglish, alternancia). Lo propio de este caso son 4 reglas, cada una con el
defecto que la motivo (ver validate_larga).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_toolcalling_dataset as G  # noqa: E402  (validador y teachers del repo)

TRANSCRIPTS = Path("/home/chaos/tmp/kateto-corpus/transcripts")
VOICE = "seco"

# Material de clientes (workmatch/sherut/podcast: 0/149 hoy) y canales de terceros
# identificables por nombre: goncy, rayos419, rebord69 (14 archivos, leidos: son de
# otra gente). NO alcanza para atribuir el corpus: 17/149 con ID de video tambien son
# Rayos419. Ver CONVERSACIONES-LARGAS.md.
RX_EXCLUIR = re.compile(r"workmatch|sherut|podcast|goncy|rayos419|rebord69", re.I)

# Conteo de tokens aproximado, calibrado con el dato del brief: 4,9 MB ~= 1,46 M
# tokens del corpus ASR -> ~3,4 car/token. Alcanza para el gate min_tokens; el
# costo real por muestra sale del `usage` del teacher, no de esto.
CHARS_POR_TOKEN = 3.4


def count_tokens(txt: str) -> int:
    return max(1, int(len(txt) / CHARS_POR_TOKEN))


def listar_transcripts() -> list[Path]:
    return sorted(p for p in TRANSCRIPTS.glob("*.txt")
                  if not RX_EXCLUIR.search(p.name))


def partir_lineas(lineas: list[str], n: int) -> list[str]:
    """`n` trozos contiguos por lineas (balanceados por chars)."""
    total = sum(len(l) for l in lineas)
    trozos, acc, corte = [], [], 0
    for l in lineas:
        acc.append(l)
        corte += len(l)
        if len(trozos) < n - 1 and corte >= total * (len(trozos) + 1) / n:
            trozos.append(" ".join(acc))
            acc = []
    trozos.append(" ".join(acc))
    return trozos


class Trozos:
    """Trozos del material bajo demanda. El largo ya no se conoce de antemano
    (lo decide el criterio de corte), asi que se PIDE el k-esimo y devuelve None
    cuando el material se termino. Antes el generador repetia el texto entero
    para llenar n_ex: eso era el largo fijo.

    El corte queda en `min(n_max, lineas)`: con material corto la charla se
    agota antes, que es el motivo `agotado` del criterio de corte."""

    def __init__(self, texto: str, n_max: int) -> None:
        self.lineas: list[str] = [l.strip() for l in texto.splitlines() if l.strip()]
        self.n: int = max(1, min(n_max, len(self.lineas)))
        self._t: list[str] | None = None

    def _partir(self) -> list[str]:
        return partir_lineas(self.lineas, self.n) if self.lineas else []

    def get(self, k: int) -> str | None:
        """Trozo k, o None si el material ya se consumio entero."""
        if self._t is None:
            self._t = self._partir()
        return self._t[k] if 0 <= k < len(self._t) else None

    def para_prompt(self, k: int) -> str:
        """Trozo k para el prompt. Si el material se agoto antes de --min-ex se
        repite el ultimo: la charla sigue (min_ex manda), la fuente ya no."""
        return self.get(k) or (self._t[-1] if self._t else "")


def limpiar_linea(raw: str, etiqueta_rx: str) -> str:
    """Misma limpieza que el turnwise del generador corto: el teacher antepone
    preambulos y etiquetas; se extrae la primera linea util."""
    cand = G.clean_teacher_output(raw)
    m = G.RX_ANCLA_CONTRATO.search(cand)
    if m:
        cand = cand[m.start():]
    cand = cand.strip().strip('"').strip()
    cand = re.sub(rf"^({etiqueta_rx})\s*:\s*", "", cand,
                  flags=re.I).strip().strip('"').strip()
    return cand.split("\n")[0].strip()


def prompt_usuario(material: str, historial: list[tuple[str, str]], i: int) -> str:
    conv = "\n".join(f"{'CHAT' if q == 'user' else 'KATETO'}: {t}"
                     for q, t in historial) or "(todavia no hablo nadie)"
    return f"""Estas escribiendo UNA linea de chat de un viewer argentino en el stream de Kateto.
En el stream estan hablando de esto ahora:
\"\"\"{material[:1500]}\"\"\"

Lo que se dijo hasta ahora:
{conv}

Escribi el proximo mensaje del CHAT (intercambio {i + 1}): UNA sola linea,
reaccionando a lo que se habla AHORA (pregunta algo concreto, acota, discute, se rie).
Rioplatense natural de chat (che, posta, jajaja, boludo cuando encaja). Corto como
un mensaje de chat real. Sin marcadores, sin "CHAT:" al principio. Solo lo que dice.
Si te quedaste sin nada que decir sobre esto, termina tu mensaje con `<FIN>`."""


def prompt_voz(material: str, historial: list[tuple[str, str]], i: int) -> str:
    conv = "\n".join(f"{'CHAT' if q == 'user' else 'VOS'}: {t}"
                     for q, t in historial) or "(todavia no hablo nadie)"
    return f"""Sos Kateto (seco): streamer argentino, voz propia, caracter seco. Estas en vivo
y en el stream estan hablando de esto ahora:
\"\"\"{material[:1500]}\"\"\"

Lo que se dijo hasta ahora:
{conv}

Escribi tu proximo mensaje (intercambio {i + 1}): un parrafo corto de dialogo,
de 3 a 5 frases. Esto es una charla larga, no un ping-pong: responde con sustancia, no
con una linea de chat.
REGLAS:
- CONVERSA sobre el material, NO lo repitas: prohibido copiar frases textuales de lo
  que se habla en el stream. El material manda el TEMA, no la forma.
- AVANZA: deci algo nuevo sobre lo de ahora, no repitas lo que ya dijiste antes.
- RIOPLATENSE: voseo (tenes, sabes, sos, fijate, mirá), lunfardo cuando encaja
  (boludo, quilombo, pibe, laburo, posta, che). PROHIBIDO el tuteo.
- PROHIBIDO sonar a asistente: nada de "como asistente", "en que puedo ayudarte",
  "lamento", "soy un modelo".
- Opina y tomate postura, no le des la razon por compromiso.
Sin marcadores, sin "VOS:" al principio. Solo lo que decis."""


def prompt_juez(material: str, historial: list[tuple[str, str]]) -> str:
    """Pregunta de corte al teacher: SI = el tema del material se agoto. max_tokens
    bajo y temperatura 0 justamente para que sea un clasificador y no una opinion."""
    conv = "\n".join(f"{'CHAT' if q == 'user' else 'VOS'}: {t}"
                     for q, t in historial[-4:]) or "(todavia no hablo nadie)"
    return f"""¿Ya se agotó el tema de este material, o queda algo nuevo para decir?
Material: \"\"\"{material[:600]}\"\"\"
Conversación hasta ahora: {conv}
Responde SOLO: SI o NO"""


def teacher_http(base_url: str, model: str, key: str | None, prompt: str,
                 max_tokens: int, timeout: int = 120,
                 retries: int = 4, temperature: float = 1.0) -> tuple[str, dict]:
    """Igual que G.generate pero devuelve (texto, usage) para la tabla de costos."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens,
            "reasoning_effort": "none"}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    uso: dict = {}
    last: Exception | None = None
    for intento in range(retries):
        req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            uso = (d.get("usage") or {})
            return ((d.get("choices") or [{}])[0].get("message", {}).get("content") or ""), uso
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 502, 503):
                espera = 20 * (intento + 1)
                print(f"      [{e.code}] reintento {intento + 1}/{retries} en {espera}s")
                time.sleep(espera)
                continue
            raise
    assert last is not None
    raise last


RX_FIN = re.compile(r"<\s*FIN\s*>", re.I)


def _sin_fin(cand: str) -> str:
    """Saca el marcador <FIN> del texto (nunca se guarda en el dato)."""
    return RX_FIN.sub("", cand).strip()


def higiene_voz(cand: str) -> str | None:
    """Turn-level: lo que el validador cazaría a nivel muestra se reintenta a nivel
    turno (turnwise: se pierde la linea, no la conversacion). El teacher chino
    filtra CJK suelto y el tuteo sin acento ('mira'): ambas son lineas, no muestras."""
    if G.RX_NO_LATINO.search(cand):
        return "no_latino"
    w = G._tuteo_en(cand)
    if w:
        return f"tuteo:{w}"
    if G.RX_VICIO.search(cand):
        return "vicio"
    return None


def higiene_dup(hist: list[tuple[str, str]]) -> callable:
    """Un turno repetido tira la conversacion entera: con 22 intercambios, un solo
    turno calcado deja la muestra pegada en repeticion_de_turnos (medido: 2/2
    aceptadas-intensity con 'pero si la vaca lloro...' textual dos veces). Se
    descarta en el turno y se reintenta, no al final. Solo con >=4 palabras de
    contenido: en chat real dos lineas cortas que comparten una palabra dan
    Jaccard alto sin ser repeticion."""
    def check(cand: str) -> str | None:
        cs = _contenido(cand)
        if len(cs) < 4:
            return None
        return "dup" if any(_jaccard(cs, _contenido(t)) >= 0.85 for _, t in hist) else None
    return check


def pedir_turno(maestro, prompt: str, etiqueta_rx: str,
                min_len: int, max_len: int, max_retries: int = 3,
                higiene=None) -> tuple[str, dict]:
    """Pide un turno con reintentos; devuelve (linea, uso). Linea vacia = fallo."""
    uso_total: dict = {}
    for _ in range(max_retries):
        raw, uso = maestro(prompt)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            uso_total[k] = uso_total.get(k, 0) + (uso.get(k) or 0)
        cand = limpiar_linea(raw, etiqueta_rx)
        if not (min_len <= len(cand) <= max_len):
            continue
        if higiene and higiene(cand):
            continue
        return cand, uso_total
    return "", uso_total


def decidir_corte(historial: list[tuple[str, str]], trozos_restantes: str | None,
                 resp_juez: str | None, *, hay_fin: bool = False,
                 min_ex: int = 0, max_ex: int = 60) -> str | None:
    """Motivo de corte del intercambio, o None para seguir. PURA: no cuenta
    llamadas ni lee el material, solo decide con lo que le pasan.

    Orden: agotado > cierre > juez > max_ex, y el min_ex pisa a todos (antes del
    minimo no se corta, aunque el material ya se haya terminado).
    `trozos_restantes` = el trozo k+1 (None = el material se consumio entero).
    `resp_juez` arranca con SI = se agoto el tema; cualquier otra cosa (o error,
    o None) es "segui".
    """
    n = sum(1 for q, _ in historial if q == "user")
    if n < min_ex:
        return None
    if trozos_restantes is None:
        return "agotado"
    if hay_fin:
        return "cierre"
    if (resp_juez or "").strip().upper().startswith("SI"):
        return "juez"
    return "max_ex" if n >= max_ex else None


def generate_larga(trozos: Trozos, maestro, *, min_ex: int = 12, max_ex: int = 60,
                   juez=None, juez_cada: int = 3) -> tuple[str, dict]:
    """Arma la conversacion turno por turno y la termina el CRITERIO, no un
    numero fijo: se corta al agotarse el material, cuando el chat tira <FIN>,
    cuando el juez dice SI, o al llegar a max_ex (nunca antes de min_ex).

    Devuelve (texto, stats); stats trae corte + intercambios para trazar."""
    historial: list[tuple[str, str]] = []
    dup = higiene_dup(historial)
    t0 = time.time()
    uso = Counter()
    corte = None
    i = -1
    for i in range(max_ex):
        mat = trozos.para_prompt(i)
        u, uu = pedir_turno(maestro, prompt_usuario(mat, historial, i),
                            r"CHAT|PERSONA|USUARIO|USER", 2, 400,
                            higiene=lambda c: ("no_latino" if G.RX_NO_LATINO.search(c) else
                                              ("fin_solo" if not _sin_fin(c) else dup(c))))
        uso.update({k: v for k, v in uu.items()})
        if not u:
            return "", {"fallo_en": f"user:{i}"}
        fin = bool(RX_FIN.search(u))
        historial.append(("user", _sin_fin(u)))
        v, vu = pedir_turno(maestro, prompt_voz(mat, historial, i),
                            r"VOS|KATETO|SECO|ASSISTANT", 60, 900,
                            max_retries=4,
                            higiene=lambda c: higiene_voz(c) or dup(c))
        uso.update({k: v_ for k, v_ in vu.items()})
        if not v:
            return "", {"fallo_en": f"voz:{i}"}
        historial.append((VOICE, v))
        # El <FIN> ya dio su turno de voz de cierre: se corta recien aca.
        resp_juez = (juez(prompt_juez(mat, historial))
                     if juez and juez_cada and (i + 1) % juez_cada == 0 else None)
        corte = decidir_corte(historial, trozos.get(i + 1), resp_juez, hay_fin=fin,
                              min_ex=min_ex, max_ex=max_ex)
        if corte:
            break
    partes = [f"<|im_user|>{t}<|im_end|>" if q == "user"
              else f"<|im_start|>{VOICE}\n{t}<|im_end|>"
              for q, t in historial]
    return "\n".join(partes), {"segundos": round(time.time() - t0, 1),
                               "uso": dict(uso),
                               "corte": corte or "max_ex",
                               "intercambios": i + 1}


# --- validador propio: 4 reglas por escenario (ademas del base de G) ---

_RX_PAL = re.compile(r"[a-záéíóúüñ]+", re.I)
_STOP = {"que", "para", "como", "esta", "este", "esto", "todo", "nada", "sobre",
         "nunca", "porque", "pero", "todos", "todas", "luego", "mientras",
         "tiene", "tienen", "hace", "hacer", "decir", "dicho", "cuando",
         "donde", "cual", "sino", "aunque", "entonces", "despues", "antes",
         "mismo", "misma", "estan", "estaba", "habia", "algo", "veces"}


def _norm(w: str) -> str:
    return (w.lower().replace("á", "a").replace("é", "e").replace("í", "i")
            .replace("ó", "o").replace("ú", "u").replace("ü", "u"))


def _contenido(txt: str) -> set[str]:
    return {_norm(w) for w in _RX_PAL.findall(txt)
            if len(w) >= 5 and _norm(w) not in _STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def _shingles(txt: str, k: int = 8) -> set[tuple[str, ...]]:
    toks = [_norm(w) for w in _RX_PAL.findall(txt.lower())]
    return {tuple(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def turnos_de(txt: str, quien: str) -> list[str]:
    if quien == "user":
        return [m.group(1).strip()
                for m in re.finditer(r"<\|im_user\|>(.*?)<\|im_end\|>", txt, re.S)]
    return [m.group(1).strip()
            for m in re.finditer(rf"<\|im_start\|>{re.escape(VOICE)}\s*(.*?)<\|im_end\|>",
                                 txt, re.S)]


def validate_larga(sample: str, material: str, *,
                   min_tokens: int = 3000) -> list[str]:
    """Base de G + 4 reglas propias. Cada regla nueva existe por un defecto que
    el base no caza en muestras largas:

    - corta: el punto entero del escenario es el largo (>=3000 tok). Sin este
      gate, una charla linda de 10 turnos pasa y no entrena contexto largo.
    - calco_del_material: el teacher, con el transcript a la vista, copia frases
      textuales en vez de conversar. Se mide con 8-gramas compartidos: conversar
      sobre un tema repite vocabulario suelto, copiar repite secuencias.
    - repeticion_de_turnos: para "llenar" el largo, el teacher recicla el mismo
      turno con retoques minimos. En muestras cortas no aparece; en 40+ turnos
      es el modo de fallo dominante.
    - avance_de_tema: la charla se queda clavada en el primer tema (o divaga
      fuera del material). Tercios inicial/final parecidos = no avanzo; tercio
      sin vocabulario de su trozo = se despegó del stream.
    """
    errs = G.validate(sample, voice=VOICE, min_turns=10,
                      require_voseo=True, min_palabras_usuario=1,
                      prohibir_tool=True, permitir_preguntas=True)
    tok = count_tokens(sample)
    if tok < min_tokens:
        errs.append(f"corta({tok}<{min_tokens})")
    sh_mat = _shingles(material)
    sh_mu = _shingles(sample)
    if sh_mu:
        frac = len(sh_mu & sh_mat) / len(sh_mu)
        if frac > 0.20:
            errs.append(f"calco_del_material({frac:.2f}>0.20)")
    vistos: dict[str, int] = {}
    voces, usuarios = turnos_de(sample, VOICE), turnos_de(sample, "user")
    for tanda in (voces, usuarios):
        for t in tanda:
            c = re.sub(r"\s+", " ", t.lower()).strip()
            if c in vistos:
                errs.append(f"repeticion_de_turnos:{c[:40]}")
                break
            vistos[c] = 1
    else:
        # Casi-iguales solo en turnos con sustancia (>=4 palabras de contenido):
        # en chat real "como viene X" vs "como viene Y" comparten la unica palabra
        # larga y el Jaccard da 1.0 sin ser repeticion (medido en dry-run).
        for tanda in (voces, usuarios):
            sets = [_contenido(t) for t in tanda]
            for i in range(len(sets)):
                for j in range(i + 1, len(sets)):
                    if (len(sets[i]) >= 4 and len(sets[j]) >= 4
                            and _jaccard(sets[i], sets[j]) >= 0.85):
                        errs.append("repeticion_de_turnos(casi-iguales)")
                        break
                else:
                    continue
                break
    if len(voces) >= 9:
        t = len(voces) // 3
        tercios = [_contenido(" ".join(voces[:t])),
                   _contenido(" ".join(voces[t:2 * t])),
                   _contenido(" ".join(voces[2 * t:]))]
        if _jaccard(tercios[0], tercios[2]) >= 0.55:
            errs.append(f"avance_de_tema(tercios_iguales:{_jaccard(tercios[0], tercios[2]):.2f})")
        mt = len(material) // 3 or 1
        trozos_mat = [_contenido(material[:mt]), _contenido(material[mt:2 * mt]),
                      _contenido(material[2 * mt:])]
        for k in range(3):
            if len(tercios[k] & trozos_mat[k]) < 4:
                errs.append(f"avance_de_tema(tercio{k + 1}_despegado)")
                break
    return errs


# --- chequeo del camino de datos (CPU, sin GPU) ---

def check_data(path: str, base: str | None = None) -> int:
    """Verifica que el JSONL se concatena con el dataset y que --loss_mask qa lo
    digiere: replica dataset.py+mask.py en Python puro. dataset.py usa
    <|im_user|>/<|im_start|> como t1/t2 cuando el ctx los trae; create_mask pone
    en loss los tokens despues de <|im_start|> (voz) y -100 al resto. Aca se
    chequea lo mismo a nivel de spans: cada muestra tiene que tener >=1 span de
    voz no vacio (si no, la mascara deja todo en -100 y la muestra no entrena)."""
    n_ok = n_mal = 0
    toks: list[int] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                print(f"  linea {i}: JSON invalido");
                n_mal += 1
                continue
            spans = re.findall(rf"<\|im_start\|>{VOICE}\s*(.*?)<\|im_end\|>", t, re.S)
            ok = (t.startswith("<|im_user|>") and t.rstrip().endswith("<|im_end|>")
                  and "<|im_user|>" in t and any(s.strip() for s in spans))
            toks.append(count_tokens(t))
            if ok:
                n_ok += 1
            else:
                n_mal += 1
                print(f"  linea {i}: formato/mascara mal")
    print(f"{path}: {n_ok} ok, {n_mal} mal"
          + (f", tokens/mediana/min/max: {sum(toks)}/{sorted(toks)[len(toks)//2]}/{min(toks)}/{max(toks)}"
             if toks else ""))
    if base:
        nb = sum(1 for _ in open(base, encoding="utf-8"))
        print(f"concatena con {base}: {nb} + {n_ok} = {nb + n_ok} muestras, "
              f"mismo esquema {{'text'}}")
    print("loss_mask qa: OK en seco (spans de voz no vacios -> la mascara tiene "
          "que aprender; sin spans seria todo -100)")
    return 0 if n_mal == 0 and n_ok > 0 else 1


# --- dry-run: validador sin gastar llamadas ---
# Fixtures ARMADAS con 3 topicos disjuntos (T1/T2/T3): el material es T1+T2+T3 y
# la voz del tercio k usa palabras de Tk. Asi la BUENA ancla cada tercio a su
# trozo y los tercios difieren; las MALAS rompen exactamente una cosa.

T1 = "carrito compra to-do lista productos oferta cliente tienda pedido entrega".split()
T2 = "teclado switches ruido oficina mate termo torta lluvia ventana campera".split()
T3 = "guitarra cuerdas ensayo banda escenario luces sonido consola publico".split()
_PLANT = [("che, {a} y {b} hoy, posta que {c} cerro todo, boludo", "che, que onda {a}"),
          ("mirá, te digo la posta: {a} con {b} rinde, {c} ni hablar", "posta lo de {a}?"),
          ("fijate que {a} viene encaminado, {b} de a poco, {c} tranqui", "como viene {a}"),
          ("sos un genio pero {a} no sale solo, hay que laburar {b} y {c}", "{a} sale hoy?")]


def _armar(turnos_v: list[str], turnos_u: list[str]) -> str:
    partes = []
    for u, v in zip(turnos_u, turnos_v):
        partes.append(f"<|im_user|>{u}<|im_end|>")
        partes.append(f"<|im_start|>{VOICE}\n{v}<|im_end|>")
    return "\n".join(partes)


def _voz_topico(topico: list[str], k: int) -> str:
    a, b, c = topico[k % len(topico)], topico[(k + 3) % len(topico)], topico[(k + 5) % len(topico)]
    p, _ = _PLANT[k % len(_PLANT)]
    return p.format(a=a, b=b, c=c)


def dry_run() -> int:
    print("=== DRY RUN: validador de largas, sin teacher ===\n")
    mat = " ".join(T1) + " XXXX " + " ".join(T2) + " XXXX " + " ".join(T3)
    casos = []
    tv, tu = [], []
    for i in range(24):
        tk = (T1, T2, T3)[i // 8]
        tv.append(_voz_topico(tk, i))
        _, pu = _PLANT[i % len(_PLANT)]
        a = tk[i % len(tk)]
        tu.append(pu.format(a=a) + f" ({i})")
    casos.append(("BUENA", _armar(tv, tu), mat, 600, []))
    casos.append(("CORTA", _armar(tv[:1], tu[:1]), mat, 3000, ["corta("]))
    casos.append(("CALCADA", _armar([mat] * 6, tu[:6]), mat * 4, 3000,
                  ["calco_del_material"]))
    casos.append(("REPETIDA", _armar([tv[0]] * 12, tu[:12] + tu[:4]), mat, 1,
                  ["repeticion_de_turnos"]))
    tv3 = [_voz_topico(T1, i) + f" (variante {i})" for i in range(12)]
    tu3 = [f"che, que onda la fruta numero {i}" for i in range(12)]
    casos.append(("SIN_AVANCE", _armar(tv3, tu3), mat, 1, ["avance_de_tema"]))
    rc = 0
    for nombre, txt, m, mt, esperados in casos:
        errs = validate_larga(txt, m, min_tokens=mt)
        if not esperados:
            ok = not errs
        else:
            ok = all(any(e.startswith(p) for e in errs) for p in esperados)
        print(f"{nombre}: {'OK' if ok else 'FALLO'} -> {errs}")
        rc += 0 if ok else 1
    print("\nBUENA tiene que pasar; CORTA/CALCADA/REPETIDA/SIN_AVANCE tienen que "
          "fallar por su regla.")
    return rc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Conversaciones largas de Kateto desde transcripts")
    ap.add_argument("--base-url", default="http://127.0.0.1:3001/v1")
    ap.add_argument("--teacher", choices=["http", "cmd"], default="http")
    ap.add_argument("--model", default=None,
                    help="default http->auto:fast, cmd->xiaomi/mimo-v2.5-pro")
    ap.add_argument("--key", default=None)
    ap.add_argument("--n", type=int, default=4, help="cuantas conversaciones")
    ap.add_argument("--min-ex", type=int, default=12,
                    help="intercambios MINIMOS: no se corta antes (gana el criterio)")
    ap.add_argument("--max-ex", type=int, default=60,
                    help="intercambios MAXIMOS: al llegar, corte=max_ex")
    ap.add_argument("--n-ex", type=int, default=None,
                    help="alias viejo de --max-ex (invocaciones previas)")
    ap.add_argument("--juez-cada", type=int, default=3,
                    help="preguntarle al teacher si el tema se agoto cada N intercambios (0=apago)")
    ap.add_argument("--min-tokens", type=int, default=3000)
    ap.add_argument("--transcript", default=None, help="archivo puntual (default: al azar)")
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--out", default="out/dataset/largas/largas_v1.jsonl")
    ap.add_argument("--rejects", default="out/dataset/largas/largas_v1_rejects.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--check-data", default=None, help="verifica un JSONL en seco (CPU)")
    ap.add_argument("--base", default="data/rwkv_kateto_base_train.jsonl")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        return dry_run()
    if args.check_data:
        return check_data(args.check_data, args.base)

    todo = listar_transcripts()
    if not todo:
        print("no hay transcripts (todo excluido o dir vacio)")
        return 1
    print(f"transcripts utiles: {len(todo)} (excluidos por {RX_EXCLUIR.pattern}: "
          f"{len(list(TRANSCRIPTS.glob('*.txt'))) - len(todo)})")

    uso_tot = Counter()
    if args.teacher == "http":
        model = args.model or "auto:fast"
        key = args.key or G.get_freellm_key()
        base = lambda p, mt: teacher_http(args.base_url, model, key, p, mt)

        def juez(prompt: str) -> str:
            txt, u = teacher_http(args.base_url, model, key, prompt, 4, temperature=0.0)
            uso_tot.update({k: v for k, v in u.items() if k in
                            ("prompt_tokens", "completion_tokens", "total_tokens")})
            return txt
    else:
        model = args.model or "xiaomi/mimo-v2.5-pro"
        base = lambda p, mt: (G.generate_cmd(model, p), {})
        juez = lambda prompt: G.generate_cmd(model, prompt)

    def maestro(prompt: str) -> tuple[str, dict]:
        mt = 150 if prompt.startswith("Estas escribiendo UNA linea de chat") else 400
        txt, uso = base(prompt, mt)
        uso_tot.update({k: v for k, v in uso.items() if k in
                        ("prompt_tokens", "completion_tokens", "total_tokens")})
        return txt, uso

    max_ex = args.max_ex if args.n_ex is None else args.n_ex
    rng = random.Random(args.seed)
    # Solo transcripts con material suficiente para max_ex trozos distintos; con
    # menos lineas que eso, Trozos rinde menos trozos y la charla se agota antes.
    candidatas = [p for p in todo
                  if sum(1 for _ in open(p, encoding="utf-8", errors="ignore")) >= max(100, max_ex)]
    if not candidatas:
        candidatas = todo
    out_p, rej_p = REPO / args.out, REPO / args.rejects
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ok = rej = 0
    razones: Counter[str] = Counter()
    cortes: Counter[str] = Counter()
    with out_p.open("a", encoding="utf-8") as fo, rej_p.open("a", encoding="utf-8") as fr:
        for i in range(args.n):
            tp = TRANSCRIPTS / args.transcript if args.transcript else rng.choice(candidatas)
            mat = tp.read_text(encoding="utf-8", errors="ignore")
            t0 = time.time()
            try:
                txt, stats = generate_larga(Trozos(mat, max_ex), maestro,
                                           min_ex=args.min_ex, max_ex=max_ex,
                                           juez=juez, juez_cada=args.juez_cada)
                if not txt:
                    raise RuntimeError(f"turnwise fallo ({stats.get('fallo_en')})")
                errs = validate_larga(txt, mat, min_tokens=args.min_tokens)
            except Exception as e:
                rej += 1
                razones[f"exc_{type(e).__name__}"] += 1
                fr.write(json.dumps({"fuente": tp.name, "errors": [str(e)],
                                     "text": ""}, ensure_ascii=False) + "\n")
                print(f"[{i + 1}/{args.n}] {tp.name}: EXC {type(e).__name__}: {e}")
                continue
            seg = round(time.time() - t0, 1)
            ntok = count_tokens(txt)
            ntur = len(G.RX_TURN.findall(txt)) + txt.count("<|im_user|>")
            corte, nex = stats["corte"], stats["intercambios"]
            cortes[corte] += 1
            if errs:
                rej += 1
                for e in errs:
                    razones[e.split(":")[0].split("(")[0]] += 1
                fr.write(json.dumps({"fuente": tp.name, "errors": errs,
                                     "tokens_est": ntok, "corte": corte,
                                     "intercambios": nex, "text": txt},
                                    ensure_ascii=False) + "\n")
                print(f"[{i + 1}/{args.n}] {tp.name}: RECHAZADA {ntok}tok/{ntur}t/"
                      f"{seg}s {nex}ex corte={corte} -> {errs}")
            else:
                ok += 1
                fo.write(json.dumps({"text": txt, "fuente": tp.name,
                                     "tokens_est": ntok, "turnos": ntur,
                                     "segundos": seg, "uso": stats.get("uso", {}),
                                     "corte": corte, "intercambios": nex},
                                    ensure_ascii=False) + "\n")
                print(f"[{i + 1}/{args.n}] {tp.name}: OK {ntok}tok/{ntur}t/{seg}s "
                      f"{nex}ex corte={corte}")
    print(f"\n=== resumen ===\n  aceptadas: {ok}\n  rechazadas: {rej}")
    for r, c in razones.most_common():
        print(f"    {r:30s} {c}")
    print("  cortes: " + ", ".join(f"{k} {v}" for k, v in cortes.most_common()))
    u = uso_tot
    if u.get("total_tokens"):
        print(f"  teacher: prompt {u.get('prompt_tokens',0)} + compl "
              f"{u.get('completion_tokens',0)} = {u.get('total_tokens',0)} tokens "
              f"en {(u.get('total_tokens',0) / max(1, ok + rej)):.0f} tok/muestra")
    print(f"\n  dataset: {out_p}\n  rechazos: {rej_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
