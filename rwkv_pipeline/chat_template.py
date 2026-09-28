#!/usr/bin/env python3
"""FUENTE UNICA DE VERDAD del template de chat de Kateto (ChatML RWKV).

PROBLEMA QUE RESUELVE
---------------------
Hoy cada script arma el template con su propio f-string y **se contradicen**. En
el set vivo hay 817 firmas estructurales distintas; las divergencias son TRES
ejes independientes, y la inferencia (`infer_kateto.format_chat_prompt`) solo
coincide con una combinacion:

  1. `cierra_usuario`  — ¿hay `<|im_end|>` despues del turno del usuario?
                         42,7% del set NO lo tiene (dialecto B).
  2. `salto_antes_cierre` — ¿hay `\\n` entre la respuesta y su `<|im_end|>`?
                         22.352 muestras del set lo tienen.
  3. `salto_entre_turnos` — ¿hay `\\n` entre el `<|im_end|>` de un turno y el
                         `<|im_user|>` del siguiente? 20.181 lo tienen; la
                         inferencia NO lo emite.

REGLA
-----
Los marcadores los pone SOLO este modulo. Un modelo (teacher o la voz) **nunca**
escribe un marcador: rellena el slot `answer` (o `user`) y nada mas. Si el texto
que llega a un slot trae un delimitador, `slot_guard()` levanta
(`SlotContaminado`) en vez de dejar que el marcador entre al dataset.

USO
---
    from chat_template import render_turn, render_historial, render_chat, slot_guard

    slot_guard(pregunta, "user"); slot_guard(respuesta, "answer")
    text = render_turn(pregunta, respuesta, "seco")     # turno cerrado, canonico
    text = render_chat(historial, pregunta, "seco")      # historial + turno abierto

`render_chat` reproduce **byte a byte** lo que emite
`KatetoInferenceEngine.format_chat_prompt` (lo verifica
`kateto-medicion/test-chat-template.py`). Si cambia el template, se cambia aca y
el test avisa si la inferencia quedo desalineada.

El parser (`parse_traza`) modela los TRES ejes por turno y exige round-trip
byte-exacto, asi que puede reescribir el set viejo sin adivinar: lo que no
entiende, lo deja intacto y lo reporta.
"""
from __future__ import annotations

import re

import jinja2

# ── dialectos ────────────────────────────────────────────────────────────────
DIALECTO_A = "A"   # turno de usuario CERRADO con <|im_end|>
DIALECTO_B = "B"   # turno de usuario SIN cerrar

MARCADORES = ("<|im_user|>", "<|im_start|>", "<|im_end|>", "<|endoftext|>")
RX_MARCADOR = re.compile("|".join(re.escape(m) for m in MARCADORES))

#: la combinacion que sirve la inferencia: el estandar a partir de ahora
CANONICO = {"cierra_usuario": True, "salto_antes_cierre": False,
            "salto_entre_turnos": False}


class SlotContaminado(ValueError):
    """El texto que llega a un slot trae un marcador del template adentro."""


class TemplateRoto(ValueError):
    """El texto no se puede parsear como una conversacion de este template."""


# ── templates (la unica definicion) ──────────────────────────────────────────
_ENV = jinja2.Environment(
    undefined=jinja2.StrictUndefined,
    keep_trailing_newline=True,
    autoescape=False,
    trim_blocks=False,
    lstrip_blocks=False,
)

#: turno cerrado (usuario cerrado, respuesta pegada al cierre, sin salto final)
TPL_TURNO = _ENV.from_string(
    "<|im_user|>{{ user }}<|im_end|>\n<|im_start|>{{ voice }}\n{{ answer }}<|im_end|>"
)
#: turno del asistente abierto, esperando respuesta (fin del prompt)
TPL_ABIERTO = _ENV.from_string(
    "<|im_user|>{{ user }}<|im_end|>\n<|im_start|>{{ voice }}\n"
)


def slot_guard(texto: str, campo: str) -> str:
    """Rechaza texto con marcadores del template. Devuelve el texto stripeado."""
    if texto is None:
        raise SlotContaminado(f"slot '{campo}' es None")
    m = RX_MARCADOR.search(texto)
    if m:
        raise SlotContaminado(
            f"slot '{campo}' trae el marcador {m.group(0)!r} en la posicion {m.start()}: "
            f"los marcadores los pone el template, no el texto. "
            f"Fragmento: {texto[max(0, m.start()-40):m.start()+40]!r}"
        )
    return texto.strip()


def render_turn(user: str, answer: str, voice: str, *,
                cierra_usuario: bool = True, salto_antes_cierre: bool = False,
                salto_final: str = "") -> str:
    """Un turno cerrado. Por defecto, la forma canonica (la que sirve el runtime)."""
    u, a = user.strip(), answer.strip()
    cabeza = f"<|im_user|>{u}<|im_end|>\n" if cierra_usuario else f"<|im_user|>{u}\n"
    medio = "\n" if salto_antes_cierre else ""
    return f"{cabeza}<|im_start|>{voice}\n{a}{medio}<|im_end|>{salto_final}"


def render_historial(turnos, voice: str, *, salto_entre_turnos: bool = False) -> str:
    """Turnos pasados, todos cerrados. `turnos` = [(user, answer), ...] o trazas."""
    turnos = list(turnos)
    out = []
    for i, t in enumerate(turnos):
        if isinstance(t, dict):
            out.append(render_turn(t["user"], t["answer"], t.get("voice", voice),
                                   cierra_usuario=t.get("cierra_usuario", True),
                                   salto_antes_cierre=t.get("salto_antes_cierre", False),
                                   salto_final=t.get("salto_final", "")))
        else:
            u, a = t
            ultimo = i == len(turnos) - 1
            out.append(render_turn(u, a, voice,
                                   salto_final="" if ultimo
                                   else ("\n" if salto_entre_turnos else "")))
    return "".join(out)


def render_traza(turnos) -> str:
    """Re-renderiza respetando los flags por turno (para round-trip del parser)."""
    return "".join(render_turn(t["user"], t["answer"], t["voice"],
                               cierra_usuario=t["cierra_usuario"],
                               salto_antes_cierre=t["salto_antes_cierre"],
                               salto_final=t["salto_final"]) for t in turnos)


def render_chat(turnos, pregunta: str, voice: str, max_history: int | None = None) -> str:
    """Historial cerrado + turno actual abierto. Replica `format_chat_prompt`.

    OJO con `max_history=0`: el inferidor hace `(history or [])[-max_history:]`, y
    `[-0:]` en Python es `[0:]` = la lista ENTERA, o sea "sin corte", no "sin
    historial". Se replica tal cual para que entrenamiento y servicio no se
    desalineen en silencio; el default del CLI es 10 y el de `generate` es 4.
    """
    if max_history is not None:
        turnos = list(turnos)[-max_history:]
    return (render_historial(turnos or [], voice)
            + TPL_ABIERTO.render(user=pregunta.strip(), voice=voice))


# ── parser: modela los tres ejes y exige round-trip ──────────────────────────
RX_EVENTO = re.compile(r"(<\|im_user\|>|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>)")


def _eventos(texto: str):
    return [(m.group(1), m.start(), m.end()) for m in RX_EVENTO.finditer(texto)]


def parse_traza(texto: str):
    """Devuelve (turnos, separadores) o levanta `TemplateRoto`.

    `turnos` = lista de dicts con user/voice/answer/cierra_usuario/
    salto_antes_cierre/salto_final, en el orden en que estan.
    """
    ev = _eventos(texto)
    if not ev:
        raise TemplateRoto("sin marcadores: no es una conversacion de este template")
    turnos = []
    i = 0
    while i < len(ev):
        marc, ini, fin = ev[i]
        if marc != "<|im_user|>":
            raise TemplateRoto(f"se esperaba <|im_user|> en {ini}, hay {marc!r}")
        # texto del usuario: hasta el <|im_end|> o hasta el <|im_start|>
        if i + 1 >= len(ev):
            raise TemplateRoto("turno de usuario sin cierre ni <|im_start|>")
        sig = ev[i + 1]
        if sig[0] == "<|im_end|>":
            cierra = True
            user = texto[fin:sig[1]]
            if not user.endswith("\n"):
                raise TemplateRoto("falta el \\n antes de <|im_start|>")
            user = user[:-1]
            j = i + 2
        elif sig[0] == "<|im_start|>":
            cierra = False
            user = texto[fin:sig[1]]
            if not user.endswith("\n"):
                raise TemplateRoto("falta el \\n antes de <|im_start|>")
            user = user[:-1]
            j = i + 1
        else:
            raise TemplateRoto(f"tras el usuario viene {sig[0]!r}")
        if j >= len(ev) or ev[j][0] != "<|im_start|>":
            raise TemplateRoto("falta <|im_start|>")
        _, s_ini, s_fin = ev[j]
        voz_gap = texto[s_fin:ev[j + 1][1]] if j + 1 < len(ev) else texto[s_fin:]
        if "\n" not in voz_gap:
            raise TemplateRoto("falta el \\n despues del nombre de voz")
        voice, resto = voz_gap.split("\n", 1)
        if ev[j + 1][0] == "<|im_end|>":
            # respuesta vacia
            answer, salto, sgte = resto, False, ev[j + 1][2]
            turnos.append({"user": user, "voice": voice, "answer": answer,
                           "cierra_usuario": cierra, "salto_antes_cierre": salto,
                           "salto_final": ""})
            i = j + 1
            if i < len(ev):
                raise TemplateRoto("se esperaba fin de texto tras la respuesta")
            continue
        raise TemplateRoto("respuesta no implementada para este patron")
    return turnos


def _parse_con_regex(texto: str):
    """Parser alternativo por regex, para los patrones con texto de voz variable."""
    rx_a = re.compile(
        r"<\|im_user\|>(?P<user>.*?)<\|im_end\|>\n<\|im_start\|>(?P<voice>[^\n]*)\n"
        r"(?P<answer>.*?)(?P<salto>\n?)<\|im_end\|>(?P<resto>.*)", re.S)
    rx_b = re.compile(
        r"<\|im_user\|>(?P<user>.*?)\n<\|im_start\|>(?P<voice>[^\n]*)\n"
        r"(?P<answer>.*?)(?P<salto>\n?)<\|im_end\|>(?P<resto>.*)", re.S)
    for rx, cierra in ((rx_a, True), (rx_b, False)):
        turnos = []
        pos = 0
        while pos < len(texto):
            m = rx.match(texto, pos)
            if not m:
                turnos = None
                break
            turnos.append({"user": m.group("user"), "voice": m.group("voice"),
                           "answer": m.group("answer"),
                           "cierra_usuario": cierra,
                           "salto_antes_cierre": bool(m.group("salto")),
                           "salto_final": ""})
            pos = m.end("resto") - len(m.group("resto")) + len(m.group("resto"))
            # el "resto" es lo que sigue al <|im_end|>: si arranca con <|im_user|>
            # o con \n<|im_user|>, se consume el separador
            resto = m.group("resto")
            if resto == "":
                pos = len(texto)
                break
            if resto.startswith("\n<|im_user|>"):
                turnos[-1]["salto_final"] = "\n"
                pos = m.end() - len(resto) + 1
                continue
            if resto.startswith("<|im_user|>"):
                turnos[-1]["salto_final"] = ""
                pos = m.end() - len(resto)
                continue
            turnos = None
            break
        if turnos:
            return turnos, cierra
    return None, None


def parse_turnos(texto: str):
    """Parsea y VERIFICA round-trip. Devuelve (turnos, dialecto)."""
    turnos, cierra = _parse_con_regex(texto)
    if turnos:
        dialecto = DIALECTO_A if cierra else DIALECTO_B
        rearmado = render_traza(turnos)
        if rearmado == texto:
            return turnos, dialecto
    raise TemplateRoto(f"no parsea con round-trip exacto ({len(texto)} chars): {texto[:120]!r}")


def detectar_dialecto(texto: str) -> str:
    return parse_turnos(texto)[1]


def canonicalizar(texto: str):
    """Devuelve (texto_canonico, cambio, motivo). Idempotente."""
    turnos, dialecto = parse_turnos(texto)
    voces = {t["voice"] for t in turnos}
    if len(voces) != 1:
        raise TemplateRoto(f"la muestra mezcla voces: {sorted(voces)}")
    # canonico SIEMPRE: se arma con los defaults de render_turn, no con la traza
    nuevo = "".join(render_turn(t["user"], t["answer"], t["voice"]) for t in turnos)
    if nuevo == texto:
        return nuevo, False, "ya_canonico"
    ejes = []
    if not turnos[0]["cierra_usuario"]:
        ejes.append("usuario_sin_im_end")
    if any(t["salto_antes_cierre"] for t in turnos):
        ejes.append("salto_antes_cierre")
    if any(t["salto_final"] for t in turnos):
        ejes.append("salto_entre_turnos")
    return nuevo, True, "+".join(ejes) or "otro"


def es_canonico(texto: str) -> bool:
    try:
        _, cambio, _ = canonicalizar(texto)
    except TemplateRoto:
        return False
    return not cambio


if __name__ == "__main__":
    u, a, v = "Che, quien sos?", "Kateto. No me encasillo en ningun rol.", "seco"
    print("canonico :", repr(render_turn(u, a, v)))
    print("dialectoB:", repr(render_turn(u, a, v, cierra_usuario=False,
                                          salto_antes_cierre=True)))
    print("chat     :", repr(render_chat([(u, a)], "y vos?", v)))
    for kw in ({}, {"cierra_usuario": False}, {"salto_antes_cierre": True},
               {"cierra_usuario": False, "salto_antes_cierre": True}):
        t = render_turn(u, a, v, **kw)
        print(f"  {str(kw):60s} -> canonico? {es_canonico(t)}  canon={canonicalizar(t)[1:]}")
    try:
        slot_guard("respuesta con <|im_end|> adentro", "answer")
    except SlotContaminado as e:
        print("slot_guard OK:", str(e)[:70], "...")
