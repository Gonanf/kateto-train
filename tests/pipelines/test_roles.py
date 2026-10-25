#!/usr/bin/env python3
"""ROLES: division de roles (cada uno con su modelo) + scheduler sin barrera.

    env -u PYTHONPATH -u VIRTUAL_ENV \\
      /run/media/chaos/secundario/venvs/distilabel/bin/python -m pytest tests/pipelines -q

Stubs, sin red: los motores son callables que devuelven una linea con su nombre y
el rol, y tardan lo que el test diga. La suite NO llama a freellmapi ni a
llama-server.

Lo que estos tests CIERRAN, que es lo que el pedido pide:

  1. los TRES roles, y cada turno lo pide el motor de SU rol (no uno cualquiera);
  2. el rol no-humano (`paralelo`) se serializa con `<|im_dev|>`, la marca que da de
     alta `rwkv_pipeline/chat_template.py` — distinto de `<|im_user|>` (la persona);
  3. el motor que termina agarra el item siguiente SIN esperar al mas lento
     (solapamiento medido en `eventos.jsonl`, no una promesa);
  4. cada item se escribe a su shard AL TERMINAR, con flush: matando la corrida,
     lo cerrado esta en disco;
  5. un timeout se REINTENTA a otro motor (R21), no se descarta.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

PIPELINES = Path(__file__).resolve().parents[2] / "scripts" / "pipelines"
for _p in (PIPELINES, PIPELINES.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import kateto_roles as r  # noqa: E402

VOZ = "seco"


def sc_persona(**kw) -> dict:
    base = {"id": "solo_charla", "cat": "B", "tools": 0, "txt": "charla de stream",
            "voseo": True, "n_ex": 1}
    return {**base, **kw}


def sc_agente(**kw) -> dict:
    base = {"id": "fuente_otra_voz", "cat": "C", "tools": 0, "contraparte": "agente",
            "txt": "otra voz abre la charla", "voseo": True, "n_ex": 1}
    return {**base, **kw}


# --------------------------------------------------------------------------- #
# stubs
# --------------------------------------------------------------------------- #
def rol_del_prompt(prompt: str) -> str:
    """El rol del turno, leido del prompt (los stubs no ven la cola)."""
    if "la PERSONA que mira el stream" in prompt:
        return r.ROL_USUARIO
    if "OTRA VOZ / AGENTE que comenta" in prompt:
        return r.ROL_PARALELO
    return r.ROL_PRINCIPAL


def linea(motor: str, rol: str, i: int) -> str:
    """Una linea sana y distinta por (motor, rol, vuelta): el gate exige largo."""
    return f"{motor} dice desde el rol {rol} la vuelta {i}, que el stream sigue vivo"


class Stub:
    """Teacher de mentira: graba `(motor, rol)` de cada llamada y tarda lo pedido."""

    def __init__(self, motor: str, *, delays: dict[str, float] | None = None,
                 fallar: dict[str, str] | None = None) -> None:
        self.motor = motor
        self.delays = delays or {}
        self.fallar = fallar or {}
        self.llamadas: list[tuple[str, str]] = []
        self._n = 0

    def __call__(self, prompt: str) -> str:
        rol = rol_del_prompt(prompt)
        self.llamadas.append((self.motor, rol))
        self._n += 1
        if rol in self.fallar:
            kind = self.fallar[rol]
            if kind == "timeout":
                raise TimeoutError("timed out")
            raise RuntimeError(kind)
        time.sleep(self.delays.get(rol, 0.0))
        return linea(self.motor, rol, self._n)


def motores(*defs: tuple[str, frozenset, float]) -> list[r.Motor]:
    """`(nombre, roles, max_tokens)` -> motores; el device/timeout no importan aca."""
    return [r.Motor(nombre=n, base_url="http://stub", model=n, roles=roles)
            for n, roles, _ in defs]


def factory(stubs: dict[str, Stub]):
    return lambda m: stubs[m.nombre]


def correr(motores_: list[r.Motor], escenarios: list[dict], out: Path,
           stubs: dict[str, Stub] | None = None, usar_registro: bool = False) -> dict:
    muestras = r.preparar_muestras(escenarios, VOZ)
    fac = factory(stubs) if stubs else None
    prog = r.Programador(motores_, muestras, VOZ, out, usar_registro=usar_registro,
                         teacher_factory=fac)
    return prog.correr()


# --------------------------------------------------------------------------- #
# 1. los tres roles
# --------------------------------------------------------------------------- #
def test_secuencia_de_persona_junta_los_tres_roles() -> None:
    assert r.secuencia(sc_persona(), 2) == [r.ROL_USUARIO, r.ROL_PRINCIPAL, r.ROL_PARALELO,
                                            r.ROL_USUARIO, r.ROL_PRINCIPAL, r.ROL_PARALELO]


def test_secuencia_con_contraparte_agente_la_abre_el_paralelo() -> None:
    assert r.secuencia(sc_agente(), 2) == [r.ROL_PARALELO, r.ROL_PRINCIPAL,
                                           r.ROL_PARALELO, r.ROL_PRINCIPAL]


def test_cada_rol_lo_pide_su_motor(tmp_path: Path) -> None:
    """El turno de cada rol sale del motor de ESE rol, no de uno cualquiera."""
    motores_ = [
        r.Motor("e2b", "http://stub", "e2b", frozenset({r.ROL_USUARIO})),
        r.Motor("neo", "http://stub", "neo", frozenset({r.ROL_PARALELO})),
        r.Motor("qat", "http://stub", "qat", frozenset({r.ROL_PRINCIPAL})),
    ]
    stubs = {m.nombre: Stub(m.nombre) for m in motores_}
    correr(motores_, [sc_persona()], tmp_path, stubs)
    pedidos = {(m, rol) for s in stubs.values() for m, rol in s.llamadas}
    assert pedidos == {("e2b", r.ROL_USUARIO), ("neo", r.ROL_PARALELO),
                       ("qat", r.ROL_PRINCIPAL)}


def test_motor_generico_sirve_cualquier_rol(tmp_path: Path) -> None:
    """El cuarto motor (`auto`): sin `roles`, agarra cualquier turno de la cola."""
    motores_ = [r.Motor("auto", "http://stub", "auto", frozenset())]
    stubs = {"auto": Stub("auto")}
    stats = correr(motores_, [sc_persona()], tmp_path, stubs)
    roles = {rol for _, rol in stubs["auto"].llamadas}
    assert roles == {r.ROL_USUARIO, r.ROL_PRINCIPAL, r.ROL_PARALELO}
    assert stats["aceptadas"] == 1


def test_el_generico_no_le_roba_el_turno_al_especialista_libre(tmp_path: Path) -> None:
    """Con el especialista libre, el turno de su rol va al ESPECIALISTA (su modelo).

    El generico solo lo agarra si el especialista esta ocupado: asi 'cada rol con
    su propio modelo' se sostiene, y el cuarto motor igual agrega items en paralelo.
    """
    motores_ = [
        r.Motor("e2b", "http://stub", "e2b", frozenset({r.ROL_USUARIO})),
        r.Motor("neo", "http://stub", "neo", frozenset({r.ROL_PARALELO})),
        r.Motor("qat", "http://stub", "qat", frozenset({r.ROL_PRINCIPAL})),
        r.Motor("auto", "http://stub", "auto", frozenset()),
    ]
    stubs = {m.nombre: Stub(m.nombre) for m in motores_}
    correr(motores_, [sc_persona()], tmp_path, stubs)
    assert ("qat", r.ROL_PRINCIPAL) in stubs["qat"].llamadas
    assert ("auto", r.ROL_PRINCIPAL) not in stubs["auto"].llamadas


# --------------------------------------------------------------------------- #
# 2. el formato: cada rol con SU marcador
# --------------------------------------------------------------------------- #
def test_el_marcador_del_rol_no_humano_es_el_del_template() -> None:
    """`<|im_dev|>` sale de la fuente unica: no se inventa un string aca."""
    from rwkv_pipeline.chat_template import MARCAS_ROL
    assert r.IM_USER == MARCAS_ROL["user"]
    assert r.IM_DEV == MARCAS_ROL["developer"]
    assert r.IM_DEV != r.IM_USER


def test_el_paralelo_se_serializa_con_im_dev_y_no_como_una_persona() -> None:
    hist = [(r.ROL_USUARIO, "che, que onda el stream"),
            (r.ROL_PRINCIPAL, "aca andamos, tranqui"),
            (r.ROL_PARALELO, "yo miro desde el chat")]
    txt = r.armar(hist, VOZ)
    assert txt.count(r.IM_USER) == 1
    assert txt.count(r.IM_DEV) == 1
    assert txt.count(r.MARCADOR_VOZ.format(voz=VOZ)) == 1
    cuerpos_user = [c for cab, c in r.desarmar(txt) if cab == r.IM_USER]
    cuerpos_dev = [c for cab, c in r.desarmar(txt) if cab == r.IM_DEV]
    assert cuerpos_user == ["che, que onda el stream"]
    assert cuerpos_dev == ["yo miro desde el chat"]


def test_roles_de_relee_los_tres_roles_del_texto() -> None:
    """El formato distingue quien habla: el rol se reconstruye, no se adivina."""
    hist = [(r.ROL_USUARIO, "che, que onda el stream"),
            (r.ROL_PARALELO, "yo miro desde el chat"),
            (r.ROL_PRINCIPAL, "aca andamos, tranqui")]
    assert r.roles_de(r.armar(hist, VOZ), VOZ) == hist


def test_armar_desarmar_rearma_exacto() -> None:
    hist = [(r.ROL_USUARIO, "hola que tal"), (r.ROL_PRINCIPAL, "todo bien che"),
            (r.ROL_PARALELO, "y bueno")]
    txt = r.armar(hist, VOZ)
    assert txt.endswith(r.IM_END + "\n")
    assert "\n".join(f"{cab}{cuerpo}{r.IM_END}" for cab, cuerpo in r.desarmar(txt)) + "\n" == txt


def test_motivos_de_rechaza_el_formato_roto() -> None:
    assert r.motivos_de("", VOZ) == ["vacia"]
    roto = f"{r.IM_USER}hola que tal el stream de hoy"
    assert "no_termina_en_im_end" in r.motivos_de(roto, VOZ)
    sano = r.armar([(r.ROL_USUARIO, "che, que onda el stream"),
                    (r.ROL_PRINCIPAL, "aca andamos tranqui con el stream")], VOZ)
    assert r.motivos_de(sano, VOZ) == []


# --------------------------------------------------------------------------- #
# 3. el scheduler no espera al mas lento (el pedido central)
# --------------------------------------------------------------------------- #
def test_un_motor_termina_y_toma_el_siguiente_mientras_otro_sigue_ocupado(tmp_path: Path) -> None:
    # `lento1`/`lento2` (principal, 0.35 s/turno) vs `rapido` (usuario+paralelo, 0 s).
    motores_ = [
        r.Motor("rapido", "http://stub", "rapido",
                frozenset({r.ROL_USUARIO, r.ROL_PARALELO})),
        r.Motor("lento1", "http://stub", "lento1", frozenset({r.ROL_PRINCIPAL})),
        r.Motor("lento2", "http://stub", "lento2", frozenset({r.ROL_PRINCIPAL})),
    ]
    stubs = {"rapido": Stub("rapido"),
             "lento1": Stub("lento1", delays={r.ROL_PRINCIPAL: 0.35}),
             "lento2": Stub("lento2", delays={r.ROL_PRINCIPAL: 0.35})}
    escenarios = [sc_persona(n_ex=2) for _ in range(3)]

    t0 = time.time()
    stats = correr(motores_, escenarios, tmp_path, stubs)
    dt = time.time() - t0
    assert stats["aceptadas"] == 3

    ev = [json.loads(l) for l in (tmp_path / "eventos.jsonl").read_text().splitlines()]
    # Un "fin" del motor rapido cae ESTRICTAMENTE dentro de una ventana del lento:
    # el rapido siguio sacando items mientras los lentos estaban ocupados.
    ventanas_lentas = [(e["t"], e["t"] + e["dt"]) for e in ev
                       if e["motor"].startswith("lento") and e["evento"] == "fin"]
    fins_rapidos = [e["t"] for e in ev if e["motor"] == "rapido" and e["evento"] == "fin"]
    assert any(a < f < b for a, b in ventanas_lentas for f in fins_rapidos), \
        "no hay solapamiento: los motores se turnaron en bloques separados"

    # y el total es menor que la suma serial de los turnos.
    serial = sum(e["dt"] for e in ev if e["evento"] == "fin")
    assert dt < serial


def test_el_lento_no_bloquea_a_los_demas(tmp_path: Path) -> None:
    """Mientras el `lento` esta en su PRIMER turno, el `rapido` ya cerro un item."""
    motores_ = [
        r.Motor("rapido", "http://stub", "rapido",
                frozenset({r.ROL_USUARIO, r.ROL_PARALELO})),
        r.Motor("lento", "http://stub", "lento", frozenset({r.ROL_PRINCIPAL})),
    ]
    stubs = {"rapido": Stub("rapido"), "lento": Stub("lento", delays={r.ROL_PRINCIPAL: 0.6})}
    # muestra A: principal FORZADO -> la cierra el rapido al toque, sin modelo.
    a = sc_agente(forzar_respuesta=["<|no_response|>"])
    b = sc_persona()
    muestras = r.preparar_muestras([a, b], VOZ)
    prog = r.Programador(motores_, muestras, VOZ, tmp_path, usar_registro=False,
                         teacher_factory=factory(stubs))
    visto: dict[str, bool] = {}
    hilo = threading.Thread(target=prog.correr, daemon=True)
    hilo.start()
    shard_rapido = tmp_path / "shards" / "rapido.jsonl"
    limite = time.time() + 3
    while time.time() < limite and hilo.is_alive():
        if shard_rapido.exists() and shard_rapido.read_text().strip():
            visto["antes_de_terminar"] = True
            break
        time.sleep(0.02)
    hilo.join(timeout=5)
    assert visto.get("antes_de_terminar"), \
        "el shard no tenia la muestra cerrada mientras el otro motor seguia ocupado"


# --------------------------------------------------------------------------- #
# 5. timeout -> reintento a otro motor (R21)
# --------------------------------------------------------------------------- #
def test_un_timeout_se_reintenta_a_otro_motor(tmp_path: Path) -> None:
    """El item que un motor tiempo-out NO se descarta: lo toma otro (R21)."""
    motores_ = [
        r.Motor("falla", "http://stub", "falla", frozenset({r.ROL_PRINCIPAL})),
        r.Motor("auto", "http://stub", "auto", frozenset()),
        r.Motor("charla", "http://stub", "charla",
                frozenset({r.ROL_USUARIO, r.ROL_PARALELO})),
    ]
    stubs = {"falla": Stub("falla", fallar={r.ROL_PRINCIPAL: "timeout"}),
             "auto": Stub("auto"), "charla": Stub("charla")}
    stats = correr(motores_, [sc_agente()], tmp_path, stubs)
    assert stats["aceptadas"] == 1
    assert stats["descartadas"] == 0
    ev = [json.loads(l) for l in (tmp_path / "eventos.jsonl").read_text().splitlines()]
    assert any(e["evento"] == "retry" and e["motor"] == "falla" for e in ev)
    assert any(e["evento"] == "fin" and e["motor"] == "auto" for e in ev)


def test_un_motor_caido_tambien_se_reintenta(tmp_path: Path) -> None:
    """Un server muerto (ConnectionError/URLError) es INFRA: reintenta, no descarta."""
    motores_ = [
        r.Motor("caido", "http://stub", "caido", frozenset({r.ROL_PRINCIPAL})),
        r.Motor("auto", "http://stub", "auto", frozenset()),
        r.Motor("charla", "http://stub", "charla",
                frozenset({r.ROL_USUARIO, r.ROL_PARALELO})),
    ]
    stubs = {"caido": Stub("caido", fallar={r.ROL_PRINCIPAL: "URLError"}),
             "auto": Stub("auto"), "charla": Stub("charla")}
    stats = correr(motores_, [sc_agente()], tmp_path, stubs)
    assert stats["aceptadas"] == 1 and stats["descartadas"] == 0
    ev = [json.loads(l) for l in (tmp_path / "eventos.jsonl").read_text().splitlines()]
    assert any(e["evento"] == "retry" and e["motor"] == "caido" for e in ev)
    assert any(e["evento"] == "fin" and e["motor"] == "auto" for e in ev)


# --------------------------------------------------------------------------- #
# 4. flush por item y cobertura de roles
# --------------------------------------------------------------------------- #
def test_rol_sin_motor_no_arranca(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        correr([r.Motor("solo_usuario", "http://stub", "x",
                        frozenset({r.ROL_USUARIO}))], [sc_persona()], tmp_path)


def test_los_stats_cierran_la_cuenta(tmp_path: Path) -> None:
    motores_ = [r.Motor("auto", "http://stub", "auto", frozenset())]
    stubs = {"auto": Stub("auto")}
    stats = correr(motores_, [sc_persona(), sc_agente()], tmp_path, stubs)
    assert stats["aceptadas"] + stats["rechazadas"] + stats["descartadas"] == 2
