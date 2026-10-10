"""Tests de `scripts/k5_sistema.py`. Stdlib puro + pytest, sin red.

Cubren lo que NO es obvvio por lectura: que cada regla de feedback matchee su
ejemplo y NO matchee un turno limpio, que el guard de turno mudo reintente, y
que el schema del juez sea lo que dice ser.

Lo que necesita servidor (`salud`, `comedia`, `juez` de verdad, TTS) NO va aca:
eso se verifica corriendo el CLI, ver docs/k5-sistema.md.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

_AQUI = Path(__file__).resolve().parent
# El test importa `k5_sistema` y sus tres dependencias (`scripts.eval_http_backend`,
# `rwkv_pipeline.piper_tts`, `kateto.safety.content_filter`), que solo existen en el
# repo. Localizado por variable de entorno o por busqueda; si no esta, los tests
# que no necesitan esos modulos igual corren.
REPO = Path(os.environ.get("KATETO_REPO", "")).resolve() or next(
    (p for p in _AQUI.parents if (p / "scripts" / "eval_http_backend.py").exists()),
    _AQUI.parent,
)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(_AQUI))

import k5_sistema as k  # noqa: E402


def _regla(rid: str) -> k.Regla:
    return next(r for r in k.REGLAS if r.id == rid)


# --------------------------------------------------------------------------- #
# Feedback deterministico
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rid,texto", [
    ("R4/M6", "haha (risa)"),
    ("R4/M6", "che [pausa]"),
    ("R4/M6", "mira *esto*"),
    ("R8", "hola!"),
    ("R22", "el chiste es que no dormi"),
    ("R21", "¿querés que te lo explique?"),
    ("R21", "¿algo más?"),
    ("R21", "¿necesitás algo?"),
    ("M9/R19", "¡Hola, cómo andás?"),
    ("think_leak", "<think>el usuario pregunta"),
    ("M6", "che 😀"),
    ("R19", "tu madre no entiende nada"),
])
def test_regla_matchea_su_ejemplo(rid, texto):
    assert _regla(rid).patron.search(texto), f"{rid} no matchea {texto!r}"


def test_r21_acepta_los_dos_ordenes():
    # MEDIDO: "en que puedo ayudarte" (sin "te") es lo que devuelve neohorse y
    # la version sin `te` se escapaba. Las dos formas tienen que entrar.
    assert _regla("R21").patron.search("¿En que puedo ayudarte hoy?")
    assert _regla("R21").patron.search("en qué te puedo ayudar")


@pytest.mark.parametrize("texto", [
    "el error era en mi configuracion",
    "no puedo hacer nada por ahora",
    "me trabé con el deploy tres horas",
    "por culpa de lo lento y torpe",
])
def test_turno_limpio_no_trae_violaciones(texto):
    # Guard contra falsos positivos: el filtro no puede ser tan ruidoso que
    # marque todo, o nadie lo mira.
    r = k.feedback_deterministico(texto)
    assert r["violaciones"] == [], r["violaciones"]
    assert r["veredicto"] == "ok"
    assert r["puntajes"] == {"limites": 3, "limpieza": 3}


def test_veredicto_ajuste_si_hay_grave():
    r = k.feedback_deterministico("Hola! ¿En que puedo ayudarte? (risa)")
    assert r["veredicto"] == "ajuste"
    assert {v["regla"] for v in r["violaciones"]} >= {"R4/M6", "R8", "R21"}


def test_solo_avisos_no_tiran_a_ajuste():
    # Sin "hola" al inicio a proposito: caeria en M9/R19 y no aislaria R8.
    r = k.feedback_deterministico("el deploy salio bien!")
    assert r["veredicto"] == "ok"
    assert [v["regla"] for v in r["violaciones"]] == ["R8"]
    assert r["puntajes"]["limpieza"] == 2
    assert r["puntajes"]["limites"] == 3


def test_cada_violacion_trae_regla_y_cita():
    r = k.feedback_deterministico("<think>x</think>")
    v = r["violaciones"][0]
    assert v["regla"] and v["que"] and v["cita"] == "<think>"
    assert v["gravedad"] in {"grave", "aviso"}


def test_puntajes_no_negativos():
    r = k.feedback_deterministico("<think>a</think> 😱 ¡b! ¿querés que sea?")
    assert r["puntajes"]["limites"] >= 0
    assert r["puntajes"]["limpieza"] >= 0


def test_texto_vacio_no_revienta():
    r = k.feedback_deterministico("")
    assert r["violaciones"] == []
    assert r["veredicto"] == "ok"


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def test_prompt_turno_arranca_con_el_registro():
    # El registro va VERBATIM del archivo: si esto falla, el prompt no inyecta.
    p = k.prompt_turno([], "hola")
    assert p.startswith(k.registro())
    assert p.endswith("hola\n")
    assert "Kateto" in p


def test_prompt_turno_intercala_el_historial():
    p = k.prompt_turno([("che", "que tal"), ("bien", "yo tambien")], "hola")
    assert "che\nque tal" in p
    assert "bien\nyo tambien" in p
    assert p.count(k.registro()) == 1


def test_system_block_inyecta_la_voz_al_final():
    for voz in k.VOCES:
        b = k.system_block(voz)
        assert b.startswith(k.registro())
        assert b.rstrip().endswith(k.VOCES[voz]["desc"])
        assert f"voz es {voz}" in b


def test_system_block_rechaza_voz_inventada():
    with pytest.raises(KeyError):
        k.system_block(" Standup Clown")


def test_voz_desconocida_falla_en_hablar():
    with pytest.raises(KeyError):
        k.hablar("hola", voz=" Standup Clown")


# --------------------------------------------------------------------------- #
# Guard de turno mudo (el bug medido 2026-10-07)
# --------------------------------------------------------------------------- #
def test_hablar_reintenta_hasta_un_turno_con_voz(monkeypatch):
    # Dos mudos y despues texto: tiene que reintentar 2 veces y Devolver voz.
    llamadas = []

    def fake_generate(url, model, prompt, **kw):
        llamadas.append(kw["seed"])
        txt = "" if len(llamadas) < 3 else "por culpa de todo"
        return {"text": txt, "n_tokens": 12, "elapsed_s": 0.1}

    monkeypatch.setattr(k, "generate_http", fake_generate)
    r = k.hablar("hola")
    assert r["texto"] == "por culpa de todo"
    assert r["reintentos"] == 2
    # Cada reintento cambia la seed: repetir la misma llamada daria el mismo
    # mudo, porque el muestreo no es determinista con la misma seed.
    assert len(set(llamadas)) == len(llamadas) == 3


def test_hablar_acepta_un_mudo_al_ultimo_intento(monkeypatch):
    # Si el modelo se sigue callando, devuelve "" (con el reintento contado) en
    # vez de reintentar para siempre: un stream no puede colgarse.
    n = []

    def fake_generate(url, model, prompt, **kw):
        n.append(1)
        return {"text": "", "n_tokens": 1, "elapsed_s": 0.1}

    monkeypatch.setattr(k, "generate_http", fake_generate)
    r = k.hablar("hola")
    assert r["texto"] == ""
    assert r["reintentos"] == k.TURNOS_MUDOS_REINTENTOS - 1
    assert len(n) == k.TURNOS_MUDOS_REINTENTOS


def test_hablar_no_reintenta_un_turno_normal(monkeypatch):
    n = []

    def fake_generate(url, model, prompt, **kw):
        n.append(1)
        return {"text": "todo bien", "n_tokens": 4, "elapsed_s": 0.1}

    monkeypatch.setattr(k, "generate_http", fake_generate)
    r = k.hablar("hola")
    assert r["texto"] == "todo bien"
    assert r["reintentos"] == 0
    assert len(n) == 1


def test_hablar_recorta_el_prefijo_repetido(monkeypatch):
    # El modelo a veces repite el estimulo antes de responder.
    monkeypatch.setattr(k, "generate_http", lambda url, model, prompt, **kw: {
        "text": "hola\nque tal", "n_tokens": 5, "elapsed_s": 0.1})
    assert k.hablar("hola")["texto"] == "que tal"


def test_hablar_no_reporta_el_raw_anidado(monkeypatch):
    monkeypatch.setattr(k, "generate_http", lambda url, model, prompt, **kw: {
        "text": "hola", "n_tokens": 2, "elapsed_s": 0.1, "raw": {"x": 1}})
    r = k.hablar("hola")
    assert "raw" not in r["raw"]
    assert r["raw"]["n_tokens"] == 2


# --------------------------------------------------------------------------- #
# Juez
# --------------------------------------------------------------------------- #
def test_schema_del_juez_es_json_serializable():
    import json
    # Se manda entero por HTTP: si no serializa, el juez se cae en runtime.
    json.dumps(k.JUEZ_SCHEMA)


def test_schema_del_juez_acerca_los_puntajes():
    props = k.JUEZ_SCHEMA["properties"]
    assert set(k.JUEZ_SCHEMA["required"]) == {"ritmo", "anclaje", "seco", "notas"}
    for eje in ("ritmo", "anclaje", "seco"):
        assert props[eje]["type"] == "integer"
        assert props[eje]["minimum"] == 0 and props[eje]["maximum"] == 3
    # Sin `maxLength` el modelo encadenaba espacios hasta truncar el JSON.
    assert props["notas"]["items"]["maxLength"] <= 120


def test_prompt_del_juez_no_pide_copiar_el_json():
    # La gramatica ya fuerza el shape; pedirlo en el prompt hacia al modelo a
    # narrar y gastar el presupuesto en <think>.
    assert "JSON" not in k.JUEZ_PROMPT
    assert "{pregunta}" in k.JUEZ_PROMPT and "{turno}" in k.JUEZ_PROMPT


def test_juez_caido_no_es_veredicto_de_paso(monkeypatch):
    # Un juez que no responde tiene que quedar marcado, nunca Approved.
    monkeypatch.setattr(k.urllib.request, "urlopen",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("down")))
    r = k.feedback_juez("hola", "que tal", k.SERVERS[0])
    assert r["ok"] is False and "error" in r


def test_verificar_con_juez_ignora_un_juez_caido(monkeypatch):
    monkeypatch.setattr(k, "feedback_juez",
                        lambda *a, **kw: {"ok": False, "error": "down"})
    r = k.verificar("hola", "todo bien", con_juez=True)
    # El veredicto sale del deterministico, no del juez roto.
    assert r["veredicto"] == "ok"
    assert r["juez"]["ok"] is False


# --------------------------------------------------------------------------- #
# Estructura
# --------------------------------------------------------------------------- #
def test_los_dos_servidores_del_brief_estan_declarados():
    # El brief dice 11662 = neohorse-1.4B y 11661 = e2b. Si se reordenan, el
    # default de --servidor cambia de modelo sin avisar.
    por_url = {s.url.rsplit(":", 1)[1]: s.nombre for s in k.SERVERS}
    assert por_url == {"11662": "neohorse", "11661": "e2b"}


def test_toda_regla_tiene_patron_compilado():
    for r in k.REGLAS:
        assert isinstance(r.patron, re.Pattern), r.id
        assert r.que and r.gravedad in {"grave", "aviso"}