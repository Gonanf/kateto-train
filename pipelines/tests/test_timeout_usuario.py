#!/usr/bin/env python3
"""R21: el rol usuario tiene su propio presupuesto, y un timeout se reintenta.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python -m pytest pipelines/tests -q

Que cierra esto, con la instrumentacion de R19:

    motivo=excepcion_teacher  etapa={'intento': 0, 'rol': 'usuario', 'turno': 1}
    detalle: TimeoutError: timed out

El primer turno del usuario se comia la muestra entera porque `ref.generate`
tiene `timeout=240` y `max_tokens=2500` FIJOS para los dos roles: a 2,8 tok/s
(12B en CPU) 2500 tokens son 890 s. Dos cosas que cambiar, y una que no:

  1. `max_tokens` POR ROL. El usuario escribe mensajes de chat; darle el tope
     de la voz es lo que lo hace salir largo y narrativo.
  2. `timeout` POR ROL. El default sigue en 240 s para no mover el pool.
  3. Un timeout es INFRA, no un rechazo de contenido: cuenta contra
     `MAX_INTENTOS` y solo al agotarse deja la muestra en `descartadas.jsonl`.

Stub de servidor HTTP de verdad, no un callable: la pregunta es si el tope llega
al REQUEST. Un test que solo mira que el flag exista no la contesta. Por eso el
handler graba cada payload y el assert mira `max_tokens` en el JSON enviado.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PIPELINES = Path(__file__).resolve().parent.parent
for _p in (PIPELINES, PIPELINES.parent / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kateto_gen as g  # noqa: E402

VOZ = "seco"
SC = {"id": "solo_charla", "cat": "B", "tools": 0, "txt": "charla de stream", "voseo": True}

# Turnos sanos de cada rol, con voseo (el gate de `solo_charla` lo exige). Dos
# variantes por rol porque el gate rechaza `usuario_repetido`: un stub que
# devuelve siempre la misma linea midiria otra cosa.
TURNO_USUARIO = "che arranquemos que el stream esta muerto"
TURNO_USUARIO_2 = "dale metele que el stream sigue muerto"
TURNO_VOZ = "Che, arranqué el stream y al final no vino nadie"
TURNO_VOZ_2 = "Listo, yo te aviso cuando se llene de gente"


# --------------------------------------------------------------------------- #
# el stub de servidor: graba cada payload y sabe cuando tardarse
# --------------------------------------------------------------------------- #
class Servidor:
    """`/v1/chat/completions` de mentira que registra lo que le llega.

    `tardar` es una lista de segundos, uno por pedido: `None`/0 responde al
    toque. Asi el "tarda una sola vez y despues contesta bien" es un caso y no una
    mockería: el segundo request lo ve el mismo handler.

    El estado vive en el closure (no en `self` del handler): `BaseHTTPRequestHandler`
    no expone `.parent`, asi que el handler captura estas dos listas.
    """

    def __init__(self, tardar: list[float | None] | None = None) -> None:
        payloads: list[dict] = []
        esperas = tardar if tardar is not None else []
        n = [0]
        lock = threading.Lock()

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a) -> None:  # silencio: la suite imprime
                pass

            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                payload = json.loads(body)
                with lock:
                    idx = n[0]
                    n[0] += 1
                    payloads.append(payload)
                if idx < len(esperas) and esperas[idx]:
                    time.sleep(esperas[idx])
                # el rol se lee del prompt y el turno del indice: `idx//2` reparte
                # usuario/voz. Cada turno sale distinto, asi que ni el eco (R10)
                # ni el `usuario_repetido` del gate disparan.
                voz = "le toca a la voz" in payload["messages"][0]["content"]
                par = (TURNO_VOZ, TURNO_VOZ_2) if voz else (TURNO_USUARIO, TURNO_USUARIO_2)
                crudo = json.dumps({"choices": [{"message": {
                    "content": par[(idx // 2) % 2]}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(crudo)))
                self.end_headers()
                self.wfile.write(crudo)

        self.payloads = payloads
        self.tardar = esperas
        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self._srv.server_address[1]}/v1"
        self._hilo = threading.Thread(target=self._srv.serve_forever, daemon=True)

    def __enter__(self) -> "Servidor":
        self._hilo.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._srv.shutdown()
        self._srv.server_close()

    def users(self) -> list[dict]:
        """Los payloads de los pedidos del rol USUARIO (los que prompt no es
        el de la voz)."""
        return [p for p in self.payloads if "le toca a la voz" not in p["messages"][0]["content"]]


def teacher_por_rol(base_url: str, mt_u: int, mt_v: int, to_u: int, to_v: int):
    """Los DOS teachers que arma `GenerarMuestras.process`, con los knobs de
    los flags. Se construye aca y no via el Step para que el test lea igual el
    default que el default."""
    return (g.TeacherHTTP(base_url, "stub", None, "none", max_tokens=mt_u, timeout=to_u),
            g.TeacherHTTP(base_url, "stub", None, "none", max_tokens=mt_v, timeout=to_v))


def fila(teacher, teacher_voz=None, *, n_ex: int = 2) -> dict:
    """Una fila como la que arma `GenerarMuestras`: el loop + el gate."""
    sc = dict(SC)
    st: dict = {}
    txt, _ = g.generar_muestra(sc, VOZ, teacher, n_ex, stats=st,
                               teacher_voz=teacher_voz)
    return {
        "text": txt,
        "scenario": sc["id"],
        "cat": sc["cat"],
        "voice": VOZ,
        "fallo": "voz_eco" if st["descarte_voz_eco"] else ("" if txt else "no_se_pudo_armar"),
        "ecos": st["ecos"], "motivos_eco": st["motivos_eco"],
        "voz_eco": st["voz_eco"], "motivos_eco_voz": st["motivos_eco_voz"],
        "descartes": st["descartes"],
    }


def por_sink(filas: list[dict], tmp_path: Path):
    """Las filas por el gate y el sink reales."""
    gate = g.AnotarValidacion(name="anotar_validacion")
    out, rej, desc = (tmp_path / "ds.jsonl", tmp_path / "ds_r.jsonl", tmp_path / "ds_d.jsonl")
    sink = g.EscribirDataset(name="escribir_dataset", out_path=str(out),
                             rejects_path=str(rej), descarta_path=str(desc))
    sink._logger = logging.getLogger("test_timeout_usuario")
    for paso in gate.process(filas):
        for _ in sink.process(paso):
            pass
    sink.unload()
    leer = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]  # noqa: E731
    return (json.loads(Path(str(out).replace(".jsonl", ".stats.json")).read_text()),
            leer(out), leer(rej), leer(desc))


# --------------------------------------------------------------------------- #
# 1. el tope POR ROL llega al request
# --------------------------------------------------------------------------- #
def test_el_tope_por_rol_llega_al_request() -> None:
    """El punto del brief: no basta con que el flag exista.

    El servidor graba el JSON de cada pedido; el del rol usuario tiene que
    traer `max_tokens` corto y el de la voz el suyo.
    """
    with Servidor() as srv:
        u, v = teacher_por_rol(srv.url, 220, 2500, 240, 240)
        g.generar_muestra(dict(SC), VOZ, u, 2, teacher_voz=v)

    assert srv.payloads, "no se mando ningun request"
    assert [p["max_tokens"] for p in srv.users()] == [220, 220], "el usuario va corto"
    voz = [p for p in srv.payloads if p not in srv.users()]
    assert [p["max_tokens"] for p in voz] == [2500, 2500], "la voz conserva el suyo"


def test_el_tope_por_rol_viaja_con_cualquier_valor() -> None:
    """Los flags son los que mandan: con `--max-tokens-usuario 90` el request
    lleva 90. Sin esto, el default de 220 seria un numero muerto."""
    with Servidor() as srv:
        u, v = teacher_por_rol(srv.url, 90, 800, 240, 240)
        g.generar_muestra(dict(SC), VOZ, u, 1, teacher_voz=v)

    assert {p["max_tokens"] for p in srv.users()} == {90}
    assert {p["max_tokens"] for p in srv.payloads} - {90} == {800}


def test_sin_flags_nuevos_nada_cambia(tmp_path: Path) -> None:
    """La compatibilidad del brief: sin los flags, 240 s y 2500 tokens para LOS
    DOS roles — el comportamiento del pool no se mueve."""
    assert (g.MAX_TOKENS_USUARIO, g.MAX_TOKENS_VOZ) == (220, 2500)
    assert g.TIMEOUT_S == 240

    # un solo teacher, el de antes: los dos roles salen con su tope
    with Servidor() as srv:
        g.generar_muestra(dict(SC), VOZ,
                          g.TeacherHTTP(srv.url, "stub", None), 1)
    assert {p["max_tokens"] for p in srv.payloads} == {g.MAX_TOKENS_VOZ}

    # y el Step, sin los knobs puestos, arma los defaults del brief
    step = g.GenerarMuestras(name="generar_turnwise")
    assert (step.max_tokens_usuario, step.max_tokens_voz) == (220, 2500)
    assert (step.timeout_usuario, step.timeout_voz) == (240, 240)


def test_los_flags_existen_con_esos_defaults(capsys) -> None:
    """Los cuatro flags, leidos del parser de verdad y no de la constante."""
    with pytest.raises(SystemExit):
        g.main(["--help"])
    salida = capsys.readouterr().out
    for flag in ("--max-tokens-usuario", "--max-tokens-voz",
                 "--timeout-usuario", "--timeout-voz"):
        assert flag in salida, f"el flag {flag} no esta en el parser"


# --------------------------------------------------------------------------- #
# 2. un timeout SIEMPRE lento: se reintenta y recien ahi se descarta
# --------------------------------------------------------------------------- #
def test_timeout_siempre_se_descarta_con_su_motivo(tmp_path: Path) -> None:
    """El server no responde nunca a tiempo. La muestra tiene que terminar en
    `descartadas.jsonl` con `excepcion_teacher`/`timeout`, y el conteo de
    intentos tiene que cerrar."""
    with Servidor(tardar=[3.0] * 12) as srv:
        u, v = teacher_por_rol(srv.url, 220, 2500, 0.3, 0.3)
        stats, dataset, rechazos, descartadas = por_sink([fila(u, v)], tmp_path)

    assert len(descartadas) == 1, f"un timeout agotado deja una linea: {descartadas}"
    d = descartadas[0]
    assert d["motivo"] == "excepcion_teacher"
    assert d["detalle"] == "timeout", f"el detalle tiene que decir `timeout`: {d}"
    assert d["etapa"]["rol"] == "usuario", "murio en el primer rol que se pide"
    assert d["etapa"]["turno"] == 1
    assert stats["motivos_descarte"] == {"excepcion_teacher": 1}
    assert stats["intentos"] == stats["aceptadas"] + stats["rechazadas"] + stats["descartadas"]
    assert dataset == [] and rechazos == []


def test_el_timeout_gasta_los_tres_intentos(tmp_path: Path) -> None:
    """Cuenta contra `MAX_INTENTOS`, no contra un reintento suelto: con el server
    muerto el teacher tiene que ser llamado exactamente 3 veces."""
    with Servidor(tardar=[3.0] * 12) as srv:
        pedidos: list[float] = []
        base = srv.url

        def cuenta(prompt: str) -> str:
            pedidos.append(time.monotonic())
            return g.TeacherHTTP(base, "stub", None, "none", max_tokens=220, timeout=0.3)(prompt)

        por_sink([fila(cuenta)], tmp_path)
    assert len(pedidos) == g.MAX_INTENTOS == 3, f"intentos: {len(pedidos)}"


# --------------------------------------------------------------------------- #
# 3. un timeout UNA vez: la muestra se salva (el caso que hoy se perdia)
# --------------------------------------------------------------------------- #
def test_timeout_una_vez_no_tira_la_muestra(tmp_path: Path) -> None:
    """El caso que el brief dice que hoy se pierde: el primer pedido del usuario
    se pasa de tiempo y el segundo sale bien. Antes se descartaba la muestra
    entera; ahora sale."""
    with Servidor(tardar=[3.0]) as srv:
        u, v = teacher_por_rol(srv.url, 220, 2500, 0.3, 0.3)
        stats, dataset, _, descartadas = por_sink([fila(u, v)], tmp_path)

    assert descartadas == [], f"un timeout transitorio no descarta: {descartadas}"
    assert stats["descartadas"] == 0
    assert len(dataset) == 1, f"la muestra se guarda: {dataset}"
    assert stats["intentos"] == stats["aceptadas"] + stats["rechazadas"]


def test_timeout_una_vez_no_ensucia_el_motivo(tmp_path: Path) -> None:
    """Si se salvo, no puede quedar rastro de `excepcion_teacher` en el
    `descartes` de la fila: el motivo del descarte es el del ULTIMO intento."""
    with Servidor(tardar=[3.0]) as srv:
        u, v = teacher_por_rol(srv.url, 220, 2500, 0.3, 0.3)
        f = fila(u, v)
    assert f["descartes"] == [], f"la fila sana no lleva descartes: {f['descartes']}"


# --------------------------------------------------------------------------- #
# 4. una excepcion que NO es timeout sigue sin reintentarse (R19 no se rompe)
# --------------------------------------------------------------------------- #
def test_una_excepcion_comun_no_se_reintenta() -> None:
    """Lo de R19 se mantiene: `RuntimeError` no es transitorio y se corta en el
    acto. Lo que cambia en R21 es solo el timeout."""
    llamadas: list[int] = []

    def revienta(prompt: str) -> str:
        llamadas.append(1)
        raise RuntimeError("el teacher cayo")

    g.generar_muestra(dict(SC), VOZ, revienta, 2)
    assert len(llamadas) == 1, f"una excepcion no se reintenta: {llamadas}"


def test_un_urlerror_de_timeout_tambien_es_timeout() -> None:
    """`urlopen` a veces envuelve el `TimeoutError` en un `URLError`. Si eso no
    se pesca, el reintento no dispara en el camino real."""
    import urllib.error

    e = urllib.error.URLError(TimeoutError("timed out"))
    assert g._es_timeout(e)
    assert g._es_timeout(TimeoutError("timed out"))
    assert not g._es_timeout(RuntimeError("el teacher cayo"))