#!/usr/bin/env python3
"""R18: un teacher por rol. El usuario puede ser otro modelo en otro server.

    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/secundario/venvs/distilabel/bin/python -m pytest pipelines/tests -q

Los roles del loop son SECUENCIALES (usuario -> voz -> usuario), asi que la
latencia de un turno hoy es la SUMA de los dos roles y con dos devices es el
MAXIMO. Estos tests prueban las dos mitades de ese reparto: que cada rol pegue
a SU server (y no al del otro), y que el reparto quede MEDIDO en la muestra y en
las stats — sin eso no hay forma de saber si sirvio.

El teacher es un server HTTP de verdad (no un callable): si el pedido del rol
usuario se fuera por el otro camino no habria forma de Notarlo, que es
justamente el defecto que el reparto viene a tapar.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

PIPELINES = Path(__file__).resolve().parent.parent
for _p in (PIPELINES, PIPELINES.parent / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kateto_gen as g  # noqa: E402

SC = {"id": "solo_charla", "cat": "B", "tools": 0, "txt": "charla de stream", "voseo": True}
VOZ = "seco"
# El rol se lee del prompt, no del numero de pedido: es lo unico que ve el
# teacher del otro lado. `build_turn_prompt` arma "que le toca a {quien_txt}" y
# para la voz `quien_txt` es "la voz `seco` (Kateto)".
ES_VOZ = "le toca a la voz"


class Stub:
    """Un `llama-server` de mentira: cuenta los pedidos y contesta del rol.

    Los turnos que devuelve llevan la marca del stub adelante, asi que un turno
    de un stub nunca puede ser eco de un turno del otro (el guard de eco mira
    tokens, y "stubA-V1" vs "stubB-U1" no comparte ni el arranque de 6).
    """

    def __init__(self, modelo: str) -> None:
        self.modelo = modelo
        self.prompts: list[str] = []
        self.lock = threading.Lock()
        srv = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._srv = srv
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        host, port = self._srv.server_address[:2]
        return f"http://{host}:{port}/v1"

    def pedir(self) -> "g.TeacherHTTP":
        return g.TeacherHTTP(self.base_url, self.modelo, "k")

    def es_de_voz(self, i: int) -> bool:
        return ES_VOZ in self.prompts[i]

    def _handler(self):  # noqa: ANN202
        stub = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # silencia el access log a stderr
                pass

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                prompt = body["messages"][0]["content"]
                with stub.lock:
                    i = len(stub.prompts)
                    stub.prompts.append(prompt)
                    texto = (
                        f"stub{stub.modelo}V{i} mientras tanto yo aca cuelgo un remix y se van"
                        if ES_VOZ in prompt else
                        f"stub{stub.modelo}U{i} che en el stream hay dos pibes peleandose por el micro"
                    )
                out = json.dumps({"choices": [{"message": {"content": texto}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        return H


def correr(base_url: str, model: str, **usuario) -> list[dict]:
    """Una muestra por `GenerarMuestras.process`, que es el camino real."""
    step = g.GenerarMuestras(voice=VOZ, base_url=base_url, model=model,
                             teacher_key="k", **usuario)
    (filas,) = list(step.process([{"escenario": SC}]))
    return filas


def test_el_rol_usuario_pega_a_su_server_y_el_de_la_voz_al_suyo() -> None:
    a, b = Stub("usuario"), Stub("voz")
    filas = correr(b.base_url, b.modelo, usuario_base_url=a.base_url,
                   usuario_model=a.modelo)

    assert len(a.prompts) == 3 and len(b.prompts) == 3
    assert all(b.es_de_voz(i) for i in range(len(b.prompts))), \
        "al stub de la voz llego un pedido del rol usuario"
    assert not any(a.es_de_voz(i) for i in range(len(a.prompts))), \
        "al stub del usuario le llego un pedido del rol voz"
    assert filas[0]["text"], "la muestra no se armo"


def test_sin_flags_de_usuario_todo_pega_al_unico_server() -> None:
    a = Stub("unico")
    filas = correr(a.base_url, a.modelo)

    # 3 intercambios x 2 roles = 6 pedidos al mismo server, mitad y mitad.
    assert len(a.prompts) == 6
    assert sum(a.es_de_voz(i) for i in range(len(a.prompts))) == 3
    assert filas[0]["rol_modelo"] == {"usuario": a.modelo, "voz": a.modelo}


def test_el_texto_no_cambia_si_los_dos_roles_apuntan_al_mismo_server() -> None:
    """La compat de verdad: partir los roles no puede mover ni un caracter.

    Con `usuario_*` apuntando al MISMO server que la voz, el texto tiene que ser
    identico al de la corrida sin flags. Si aparece una diferencia, el reparto
    esta tocando algo del loop.
    """
    sin_flags, con_flags = Stub("mismo"), Stub("mismo")
    a = correr(sin_flags.base_url, sin_flags.modelo)
    b = correr(con_flags.base_url, con_flags.modelo,
               usuario_base_url=con_flags.base_url, usuario_model=con_flags.modelo,
               usuario_key="k", usuario_reasoning="none")

    assert b[0]["text"] == a[0]["text"]


def test_la_muestra_trae_el_modelo_y_los_segundos_de_cada_rol() -> None:
    a, b = Stub("usuario"), Stub("voz")
    fila = correr(b.base_url, b.modelo, usuario_base_url=a.base_url,
                  usuario_model=a.modelo)[0]

    assert fila["rol_modelo"] == {"usuario": "usuario", "voz": "voz"}
    assert set(fila["rol_segundos"]) == set(g.ROLES)
    assert all(s > 0 for s in fila["rol_segundos"].values())


def test_las_stats_reportan_promedio_y_porcentaje_de_pared_por_rol(tmp_path: Path) -> None:
    a, b = Stub("usuario"), Stub("voz")
    step = g.GenerarMuestras(voice=VOZ, base_url=b.base_url, model=b.modelo,
                             teacher_key="k", usuario_base_url=a.base_url,
                             usuario_model=a.modelo)
    (filas,) = list(step.process([{"escenario": SC}, {"escenario": SC}]))
    sink = g.EscribirDataset(out_path=str(tmp_path / "muestras.jsonl"),
                             rejects_path=str(tmp_path / "rechazos.jsonl"))
    # `_errores` lo pone `AnotarValidacion`, que aca no corre: estos tests son del
    # reparto, no del gate. Una de las dos se rechaza, y tambien tiene que quedar
    # con su trazabilidad: una muestra rechazada igual se costo, y en el smoke es
    # lo unico que se puede auditar cuando la corrida entera cae.
    filas[0]["_errores"] = ["tuteo:mira"]
    filas[1]["_errores"] = []
    sink.load()  # el ciclo de vida de verdad de un Step: `load` arma el logger
    list(sink.process(filas))
    sink.unload()

    aceptada = json.loads((tmp_path / "muestras.jsonl").read_text())
    rechazada = json.loads((tmp_path / "rechazos.jsonl").read_text())
    for fila in (aceptada, rechazada):
        assert fila["rol_modelo"] == {"usuario": "usuario", "voz": "voz"}
        assert all(fila["rol_segundos"][r] > 0 for r in g.ROLES)

    roles = json.loads((tmp_path / "muestras.stats.json").read_text())["roles"]
    assert set(roles) == {"segundos", "promedio", "pared_pct"}
    assert roles["segundos"]["usuario"] > 0 and roles["segundos"]["voz"] > 0
    # Con tolerancia y no con igualdad: `resumen_roles` redondea a 3 decimales
    # los dos campos, asi que recomputar el promedio desde el valor YA
    # redondeado puede caer en otro redondeo. El contrato es la division.
    for r in g.ROLES:
        assert roles["promedio"][r] == pytest.approx(roles["segundos"][r] / 2, abs=1e-3)
    assert sum(roles["pared_pct"].values()) == 100.0