#!/usr/bin/env python3
"""ROLES — generacion con DIVISION DE ROLES + scheduler que no espera al mas lento.

Tres roles, cada uno con su PROPIO modelo:

  1. usuario   — la persona que habla (mismo prompt que el simulador de `kateto_gen`).
  2. paralelo  — otra voz/agente que interviene o comenta EN PARALELO (o el bloque
                 de un evento del sistema).
  3. principal — la voz de Kateto, la que responde.

Y CUATRO motores corriendo a la vez, sin esperarse:

    e2b       (Vulkan1, Intel Iris Xe DG1)  -> sirve `usuario`
    neohorse  (Vulkan0, RX 6500 XT)         -> sirve `paralelo`
    qat12b    (CPU)                         -> sirve `principal`
    auto      (freellmapi :3001, `auto`)    -> consumidor GENERICO: sirve cualquier
                                               rol, o sea agrega muestras en paralelo
                                               cuando un motor dedicado esta ocupado

## La regla de scheduling (el punto del pedido)

NO es "un lote por motor y barrera". Es **una cola de items compartida + N
consumidores**: el motor que termina su item agarra el siguiente EN EL ACTO, y el
item que un motor lento tiene tomado NO bloquea a los demas (siguen sacando items
de la cola, incluso de la MISMA muestra, porque acá el item es un TURNO, no una
muestra entera). Ver la clase `Programador`.

Cada item se escribe a su shard AL TERMINAR, con `flush` (no al final de la
corrida: un instrumento que escribe al final pierde TODO si lo matan a mitad).
Timeout por item con REINTENTO a otro motor (R21), no descarte.

## El FORMATO: el rol no-humano usa `<|im_dev|>` (decision del dueno, cerrada)

El formato del dato tiene CUATRO marcadores literales: `<|im_user|>` (rol `user`),
`<|im_dev|>` (rol `developer`, el rol NO-humano), `<|im_start|>{voz}` y
`<|im_end|>`. La marca de `developer` la da de alta `rwkv_pipeline/chat_template.py`
(`MARCAS_ROL`), la FUENTE UNICA del template; aca NO se inventa un string:

    usuario  -> <|im_user|>texto<|im_end|>
    paralelo -> <|im_dev|>texto<|im_end|>    <-- la OTRA voz / agente / evento
    principal-> <|im_start|>seco\\n<respuesta><|im_end|>

`<|im_dev|>` NO agrega un token al vocabulario: es un string multi-token, la misma
clase que `<|im_user|>` (verificado con el tokenizer del repo). Con la marca puesta,
el modelo SI puede aprender a distinguir quien habla, y `roles_de` reconstruye los
tres roles del texto (antes `usuario` y `paralelo` eran indistinguibles).

    env -u PYTHONPATH -u VIRTUAL_ENV \\
      /run/media/chaos/secundario/venvs/distilabel/bin/python pipelines/kateto_roles.py \\
      --config /home/chaos/harness-run/roles/motores.json \\
      --scenario solo_charla --scenario events_lookup --seed 42 \\
      --out out/dataset/roles
"""

import argparse
import json
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PIPELINES = Path(__file__).resolve().parent
REPO = PIPELINES.parent
for _p in (PIPELINES, REPO, REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gen_toolcalling_dataset as ref  # noqa: E402  la referencia, sin tocar
import kateto_gen as gen  # noqa: E402  el generador de R2, sin tocar
import kateto_purge as purge  # noqa: E402  reglas de R1
from rwkv_pipeline.chat_template import MARCAS_ROL  # noqa: E402  FUENTE UNICA del template

# --------------------------------------------------------------------------- #
# Roles y formato
# --------------------------------------------------------------------------- #
ROL_USUARIO = "usuario"
ROL_PARALELO = "paralelo"
ROL_PRINCIPAL = "principal"
ROLES = (ROL_USUARIO, ROL_PARALELO, ROL_PRINCIPAL)

# Contrapartes que NO son una persona: el turno que abre la conversacion lo pone
# el rol `paralelo` (otra voz / agente / evento del sistema).
CONTRAPARTE_PARALELA = ("agente", "otra_voz", "evento", "sistema")

IM_USER = MARCAS_ROL["user"]
IM_DEV = MARCAS_ROL["developer"]        # rol no-humano: otra voz / agente / evento
IM_START = gen.IM_START
IM_END = gen.IM_END
MARCADOR_VOZ = gen.MARCADOR_VOZ

# La marca del rol no-humano tiene que ser la MISMA que usa el generador vivo y el
# template; si divergen, el dataset sale con dos dialectos.
assert IM_USER == gen.IM_USER, "kateto_gen y chat_template divergen en <|im_user|>"

#: grupo 1 = el marcador SOLO (incluida la cabecera `<|im_start|>{voz}\n`), grupo 2 =
#: el cuerpo. Extension de `gen.RX_BLOQUE` que ademas reconoce `<|im_dev|>`.
RX_BLOQUE_ROLES = re.compile(
    f"({re.escape(IM_USER)}|{re.escape(IM_DEV)}|{re.escape(IM_START)}[^\\n]*\\n)"
    f"(.*?){re.escape(IM_END)}", re.S)

# Presupuesto de tokens POR ROL (R21). El usuario y el agente paralelo escriben
# mensajes de chat; la voz conserva su presupuesto.
MAX_TOKENS_PARALELO = gen.MAX_TOKENS_USUARIO
MAX_TOKENS_PRINCIPAL = gen.MAX_TOKENS_VOZ
MIN_LINEA_CHARLA = gen.MIN_LINEA_VOZ
MAX_INTENTOS_ITEM = 3


def es_paralela(sc: dict) -> bool:
    """La contraparte de este escenario no es una persona."""
    return sc.get("contraparte", "persona") in CONTRAPARTE_PARALELA


def secuencia(sc: dict, n_ex: int) -> list[str]:
    """Los roles de la conversacion, en orden.

    Escenario de persona: por cada intercambio `[usuario, principal, paralelo]`
    — la otra voz/agente COMENTA EN PARALELO despues de que la voz responde. Es
    el caso que junta los tres roles en una misma muestra.

    Escenario con contraparte agente/evento: `[paralelo, principal]` — el otro
    agente abre y Kateto responde.
    """
    par = es_paralela(sc)
    out: list[str] = []
    for _ in range(n_ex):
        out.append(ROL_PARALELO if par else ROL_USUARIO)
        out.append(ROL_PRINCIPAL)
        if not par:
            out.append(ROL_PARALELO)
    return out


def n_ex_de(sc: dict) -> int:
    """Cuantos intercambios, igual que `kateto_gen`."""
    return sc.get("n_ex", 1 if sc.get("min_turns", 2) == 1 else 3)


def transcript(sc: dict, historial: list[tuple[str, str]]) -> str:
    """Lo dicho hasta ahora, con la etiqueta del contrato del escenario.

    Las tres etiquetas salen del contrato para que el prompt lea igual que el
    resto del repo: PERSONA / AGENTE / VOS.
    """
    etiqueta = ref.get_contrato_escenario(sc)["etiqueta"]
    nombres = {ROL_USUARIO: etiqueta, ROL_PARALELO: "AGENTE", ROL_PRINCIPAL: "VOS"}
    return "\n".join(f"{nombres.get(r, r)}: {t}" for r, t in historial) or \
        "(todavia no hablo nadie)"


# --------------------------------------------------------------------------- #
# Prompts. Uno por rol; los tres piden UNA sola linea sin marcadores.
# --------------------------------------------------------------------------- #
REGLAS_BASE = """- RIOPLATENSE: voseo (tenes, sabes, sos, fijate), lunfardo cuando encaja (boludo,
  quilombo, pibe, laburo, posta, che). Ni espanol neutro ni spanglish.
- PROHIBIDO el tuteo: tienes, puedes, quieres, eres, dime, hazlo, ven, amigo mio.
- COHERENCIA ANTE TODO: cada frase tiene que significar algo. No inventes palabras.
- PROHIBIDO sonar a asistente: nada de "como asistente", "en que puedo ayudarte",
  "lamento", "soy un modelo", ofrecer ayuda ni disculparse servilmente.
- Sin marcadores <|im_user|> ni <|im_end|>, sin etiquetas al principio, sin markdown."""

QUIEN = {
    ROL_USUARIO: "la PERSONA que mira el stream (cuenta algo, pregunta algo)",
    ROL_PARALELO: ("OTRA VOZ / AGENTE que comenta EN PARALELO (intervencion corta, "
                   "como si fuera el chat o un segundo agente)"),
    ROL_PRINCIPAL: "la voz `{voz}` (Kateto), con caracter propio",
}


def turno_prompt(sc: dict, rol: str, voice: str,
                 historial: list[tuple[str, str]], n_ex: int) -> str:
    """El prompt de UN turno del rol dado, con el historial como contexto.

    No reusa `ref.build_turn_prompt`: aquel asume DOS roles (user / voz) y
    etiquetaria al agente paralelo como `VOS`. Con el historial de tres roles
    hay que armar el transcript acá.
    """
    quien = QUIEN[rol].format(voz=voice)
    cierre = "" if rol == ROL_PRINCIPAL else \
        "- UNA sola intervencion corta, una o dos frases. No cierres con pregunta.\n"
    return f"""Estas escribiendo UNA sola linea de dialogo para un dataset de Kateto: un
personaje de voz argentino, streamer, con caracter propio. NO sos un asistente.

Escenario: {sc['txt']}

Lo que se dijo hasta ahora:
{transcript(sc, historial)}

Escribi AHORA el proximo mensaje, que le toca a {quien}. Es el mensaje
{len(historial) + 1} de {n_ex * 2} en total.

REGLAS:
{REGLAS_BASE}
{cierre}
Devolve SOLO eso, nada mas."""


def de_kateto(prompt: str, usar_registro: bool) -> str:
    """El registro de `prompts/prompt-compartido.md` va SOLO donde habla Kateto."""
    return f"{gen.registro()}\n\n{prompt}" if usar_registro else prompt


def armar(historial: list[tuple[str, str]], voice: str) -> str:
    """El texto final, con el formato exacto del dato. Sin modelo adentro.

    Cada rol con SU marcador: `usuario` con `<|im_user|>`, `paralelo` (la otra voz /
    agente / evento) con `<|im_dev|>`, `principal` con `<|im_start|>{voz}`. El
    formato SI distingue quien habla (decision del dueno).
    """
    partes = [
        f"{IM_START}{voice}\n{t}{IM_END}" if rol == ROL_PRINCIPAL
        else f"{IM_DEV}{t}{IM_END}" if rol == ROL_PARALELO
        else f"{IM_USER}{t}{IM_END}"
        for rol, t in historial
    ]
    return "\n".join(partes) + "\n"


def desarmar(txt: str) -> list[tuple[str, str]]:
    """`[(cabecera, cuerpo)]`, inverso exacto de `armar` (regex de tres marcadores)."""
    return [(m.group(1), m.group(2)) for m in RX_BLOQUE_ROLES.finditer(txt)]


def roles_de(txt: str, voice: str) -> list[tuple[str, str]]:
    """`[(rol, cuerpo)]` reconstruido desde el formato.

    Los tres marcadores son unicos, asi que el rol se relee del texto sin ambiguedad:
    `<|im_start|>{voz}` -> principal, `<|im_dev|>` -> paralelo, `<|im_user|>` -> usuario.
    """
    voz_cab = gen.MARCADOR_VOZ.format(voz=voice)
    out: list[tuple[str, str]] = []
    for cab, cuerpo in desarmar(txt):
        if cab == voz_cab:
            out.append((ROL_PRINCIPAL, cuerpo))
        elif cab == IM_DEV:
            out.append((ROL_PARALELO, cuerpo))
        else:
            out.append((ROL_USUARIO, cuerpo))
    return out


def motivos_de(txt: str, voice: str, min_len: int = MIN_LINEA_CHARLA) -> list[str]:
    """El gate: forma del formato + higiene de R1. VACIO = pasa.

    NO reusa `ref.validate`: aquel exige la alternancia usuario/voz de DOS roles
    y rechazaria por diseño una conversacion de tres roles. Las reglas de
    contenido que no dependen de los roles SI se reusan (`purge.evaluar`).
    """
    errs: list[str] = []
    if not txt:
        return ["vacia"]
    if not txt.rstrip().endswith(IM_END):
        errs.append("no_termina_en_im_end")
    bloques = desarmar(txt)
    if len(bloques) < 2:
        errs.append(f"pocos_bloques({len(bloques)})")
    voz_cab = gen.MARCADOR_VOZ.format(voz=voice)
    if not any(cab == voz_cab for cab, _ in bloques):
        errs.append("sin_turno_de_voz")
    if not any(cab in (IM_USER, IM_DEV) for cab, _ in bloques):
        errs.append("sin_turno_de_entrada")
    for _, cuerpo in bloques:
        if len(cuerpo.strip()) < min_len:
            errs.append(f"turno_corto({len(cuerpo.strip())})")
            break
    errs += [f"purga:{r}" for r in purge.evaluar(txt)]
    return errs


# --------------------------------------------------------------------------- #
# Motores
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Motor:
    """Un motor = un endpoint de teacher + los roles que sabe servir.

    `roles=None` (vacio) = consumidor generico: sirve cualquier rol. Es el caso
    del cuarto motor (`auto` de freellmapi), que agrega muestras en paralelo.
    """

    nombre: str
    base_url: str
    model: str
    roles: frozenset[str] = frozenset()
    key: str | None = None
    reasoning: str = "none"
    max_tokens: int = MAX_TOKENS_PRINCIPAL
    timeout: int = gen.TIMEOUT_S
    device: str = ""

    def sirve(self, rol: str) -> bool:
        return not self.roles or rol in self.roles

    def teacher(self) -> Callable[[str], str]:
        """El teacher HTTP de R2, reusado tal cual (reintenta el 401 de `auto`)."""
        return gen.TeacherHTTP(self.base_url, self.model, self.key, self.reasoning,
                               max_tokens=self.max_tokens, timeout=self.timeout)


def cargar_motores(path: str | Path) -> list[Motor]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    motores = []
    for d in data["motores"]:
        roles = frozenset(d.get("roles") or ())
        motores.append(Motor(
            nombre=d["nombre"], base_url=d["base_url"], model=d["model"], roles=roles,
            key=_clave(d.get("key")), reasoning=d.get("reasoning", "none"),
            max_tokens=int(d.get("max_tokens", MAX_TOKENS_PRINCIPAL)),
            timeout=int(d.get("timeout", gen.TIMEOUT_S)), device=d.get("device", ""),
        ))
    if not motores:
        raise SystemExit("config sin motores")
    return motores


def _clave(valor: str | None) -> str | None:
    """`key` de la config: literal, `@file:<ruta>` (para no embeber el secreto en
    un archivo versionado), o `None`."""
    if not valor:
        return None
    if valor.startswith("@file:"):
        return Path(valor[len("@file:"):]).expanduser().read_text(encoding="utf-8").strip()
    return valor


# --------------------------------------------------------------------------- #
# El scheduler: cola compartida + N consumidores, sin barrera
# --------------------------------------------------------------------------- #
# Un motor local que se cae (server muerto, OOM-kill, puerto cerrado) es INFRA, no
# un rechazo de contenido: el item se reintenta en OTRO motor (R21). MEDIDO: al
# qat12b en CPU lo mato el OOM killer y sin esto la muestra entera se descartaba
# en vez de caer al cuarto motor.
_INFRA = ("URLError", "ConnectionError", "ConnectionRefused", "RemoteDisconnected",
          "OSError", "timed out", "timeout")


def _es_infra(detalle: str) -> bool:
    return any(t.lower() in detalle.lower() for t in _INFRA)


@dataclass
class Muestra:
    """Estado de UNA conversacion. El item de la cola es un TURNO de esta muestra."""

    idx: int
    sc: dict
    voice: str
    n_ex: int
    secuencia: list[str] = field(default_factory=list)
    historial: list[tuple[str, str]] = field(default_factory=list)
    paso: int = 0
    texto: str = ""
    fallo: str = ""
    descartes: list[dict[str, Any]] = field(default_factory=list)
    hecha: bool = False


@dataclass
class Tarea:
    """Un turno por generar.

    `intentos` y `excluidos` para el reintento a OTRO motor (R21): un timeout es
    infra, y el motor que lo sufrio no vuelve a agarrar el mismo item.
    """

    muestra: Muestra
    rol: str
    intentos: int = 0
    excluidos: frozenset[str] = frozenset()


class Programador:
    """Cola de items compartida + N consumidores. El que termina agarra el siguiente.
    Los consumidores son un thread por motor. Un motor NO espera a otro: termina
    su turno, resuelve la dependencia de esa muestra (encola el turno siguiente,
    o cierra la muestra) y vuelve a sacar de la cola EN EL ACTO. El turno que un
    motor lento tiene tomado no bloquea a nadie: la cola tiene turnos de OTRAS
    muestras.

    Escribir es por item, con flush: `Shards` abre un archivo por motor y hace
    `flush()` en cada muestra cerrada.
    """

    def __init__(self, motores: list[Motor], muestras: list[Muestra], voice: str,
                 out_dir: Path, usar_registro: bool = True,
                 max_intentos: int = MAX_INTENTOS_ITEM, reloj: float | None = None,
                 teacher_factory: Callable[[Motor], Callable[[str], str]] | None = None) -> None:
        self.motores = motores
        self.muestras = muestras
        self.voice = voice
        self.out_dir = out_dir
        self.usar_registro = usar_registro
        self.max_intentos = max_intentos
        # El teacher sale de acá: en produccion es `Motor.teacher()` (HTTP real);
        # en la suite es un stub. Un solo punto de inyeccion.
        self.teacher_de = teacher_factory or (lambda m: m.teacher())
        self._t0 = reloj if reloj is not None else time.time()
        self._cond = threading.Condition()
        self._cola: list[Tarea] = []
        self._ocupado: set[str] = set()
        self._vivas = len(muestras)
        self._parar = False
        self._shards = Shards(out_dir / "shards")
        self._eventos = out_dir / "eventos.jsonl"
        self._eventos.parent.mkdir(parents=True, exist_ok=True)
        self._log = self._eventos.open("w", encoding="utf-8")
        self.stats: dict[str, Any] = {
            "motores": {m.nombre: {"turnos": 0, "segundos": 0.0,
                                   "aceptadas": 0, "rechazadas": 0,
                                   "timeouts": 0, "errores": 0} for m in motores},
            "aceptadas": 0, "rechazadas": 0, "descartadas": 0,
        }
        self._verificar_cobertura()

    def _verificar_cobertura(self) -> None:
        """Ningun rol puede quedar sin motor: la cola se colgaria esperando."""
        pedidos = {r for mu in self.muestras for r in mu.secuencia}
        sin = [r for r in pedidos if not any(mot.sirve(r) for mot in self.motores)]
        if sin:
            raise SystemExit(f"roles sin motor que los sirva: {sin}")

    # -- log con timestamps: es la prueba del no-espera ---------------- #
    def evento(self, motor: str, tipo: str, muestra: int, rol: str = "",
               segundos: float = 0.0, extra: str = "") -> None:
        fila = {"t": round(time.time() - self._t0, 3), "motor": motor, "evento": tipo,
                "muestra": muestra, "rol": rol, "dt": round(segundos, 3)}
        if extra:
            fila["detalle"] = extra
        self._log.write(json.dumps(fila, ensure_ascii=False) + "\n")
        self._log.flush()

    # -- cola ---------------------------------------------------------- #
    def _encolar(self, muestra: Muestra, motor: str = "cola") -> None:
        """Encola el proximo turno de `muestra`, o la cierra si ya termino."""
        if muestra.paso >= len(muestra.secuencia):
            self._cerrar(muestra, motor)
            return
        rol = muestra.secuencia[muestra.paso]
        # centinela forzado: no se le pide al modelo (igual que R2)
        forzadas = muestra.sc.get("forzar_respuesta", [])
        i_ex = muestra.paso // (2 if es_paralela(muestra.sc) else 3)
        if rol == ROL_PRINCIPAL and i_ex < len(forzadas) and forzadas[i_ex]:
            muestra.historial.append((ROL_PRINCIPAL, forzadas[i_ex]))
            muestra.paso += 1
            self._encolar(muestra, motor)
            return
        self._cola.append(Tarea(muestra, rol))

    def _siguiente(self, motor: Motor, ignorar: bool = False) -> Tarea | None:
        """El primer item de la cola que ESTE motor sabe servir, o `None`.

        Un item que le tiro un timeout a este motor NO vuelve a caerle (salvo
        `ignorar=True`, que se usa solo cuando el item ya no lo puede tomar nadie:
        asi un reintento no queda colgado).

        Y un motor GENERICO (sin rol propio, el cuarto) no se roba el turno de un
        ESPECIALISTA LIBRE: cada rol conserva su modelo. Cuando el especialista
        esta ocupado, el generico si lo agarra — ahi es donde AGREGA un item en
        paralelo en vez de esperar.
        """
        for i, t in enumerate(self._cola):
            if not motor.sirve(t.rol) or (not ignorar and motor.nombre in t.excluidos):
                continue
            if not motor.roles and any(
                esp.roles and esp.sirve(t.rol) and esp.nombre not in self._ocupado
                and esp.nombre not in t.excluidos
                for esp in self.motores
            ):
                continue
            return self._cola.pop(i)
        return None

    def _muerta(self, t: Tarea) -> bool:
        """A `t` no lo puede tomar ya ningun motor: todos los capaces lo excluyen."""
        return not any(m.sirve(t.rol) and m.nombre not in t.excluidos for m in self.motores)

    # -- cierre de item ------------------------------------------------ #
    def _cerrar(self, muestra: Muestra, motor: str = "cola") -> None:
        """Escribe la muestra a su shard, con flush, AL TERMINAR (no al final)."""
        muestra.hecha = True
        muestra.texto = armar(muestra.historial, self.voice)
        errs = motivos_de(muestra.texto, self.voice)
        fila = {
            "text": muestra.texto, "scenario": muestra.sc["id"], "cat": muestra.sc.get("cat", ""),
            "voice": self.voice, "roles": muestra.secuencia, "motor_fin": motor,
            "fallo": "" if not errs else "invalida", "errores": errs,
        }
        if errs:
            muestra.fallo = "invalida"
            self.stats["rechazadas"] += 1
        else:
            self.stats["aceptadas"] += 1
        self._shards.escribir(fila, motor)
        self._vivas -= 1
        if self._vivas <= 0:
            self._parar = True

    def _descartar(self, muestra: Muestra, motivo: str, rol: str, detalle: str,
                   motor: str = "cola") -> None:
        muestra.fallo = motivo
        muestra.descartes.append({
            "motivo": motivo, "rol": rol,
            "turno": muestra.paso + 1, "detalle": detalle,
            "texto": armar(muestra.historial, self.voice),
        })
        muestra.hecha = True
        self.stats["descartadas"] += 1
        self._shards.escribir({
            "text": "", "scenario": muestra.sc["id"], "cat": muestra.sc.get("cat", ""),
            "voice": self.voice, "roles": muestra.secuencia, "motor_fin": motor,
            "fallo": motivo, "errores": [motivo], "descartes": muestra.descartes,
        }, motor)
        self._vivas -= 1
        if self._vivas <= 0:
            self._parar = True

    # -- un turno, fuera del lock ------------------------------------- #
    def _generar(self, motor: Motor, tarea: Tarea) -> tuple[str, str, bool]:
        """Pide el turno al teacher del motor.

        Devuelve `(linea, motivo, reintentable)`. `max_retries=1`: el reintento de
        un timeout lo hace el SCHEDULER (y a otro motor, R21), no `_un_turno`.
        """
        m = tarea.muestra
        prompt = turno_prompt(m.sc, tarea.rol, self.voice, list(m.historial), m.n_ex)
        if tarea.rol == ROL_PRINCIPAL:
            prompt = de_kateto(prompt, self.usar_registro)
        motivos: dict[str, str] = {}
        linea = gen._un_turno(self.teacher_de(motor), prompt, min_len=MIN_LINEA_CHARLA,
                              con_tools=False, max_retries=1, motivos=motivos)
        motivo = motivos.get("motivo", "")
        detalle = motivos.get("detalle", "")
        reintentable = bool(motivos.get("reintentable")) or _es_infra(detalle)
        if detalle == "timeout":
            motivo = "timeout"
        return linea, motivo, reintentable

    # -- consumidor ---------------------------------------------------- #
    def _consumidor(self, motor: Motor) -> None:
        while True:
            with self._cond:
                while not self._parar and not self._cola:
                    self._cond.wait()
                if self._parar:
                    return
                tarea = self._siguiente(motor)
                if tarea is None:
                    # Hay cola, pero nada que ESTE motor sepa servir... salvo que
                    # el item ya no lo pueda tomar nadie (todos sus motores lo
                    # excluyeron): ahi se lo lleva igual, para no colgar la corrida.
                    if any(self._muerta(t) for t in self._cola):
                        tarea = self._siguiente(motor, ignorar=True)
                    if tarea is None:
                        self._cond.wait()
                        continue
                self._ocupado.add(motor.nombre)
                muestra, rol = tarea.muestra, tarea.rol
            t0 = time.time()
            self.evento(motor.nombre, "inicio", muestra.idx, rol)
            error = ""
            try:
                linea, motivo, reintentable = self._generar(motor, tarea)
            except Exception as e:  # noqa: BLE001 — el error no se lleva la corrida
                linea, motivo, reintentable = "", f"excepcion:{type(e).__name__}", False
                error = type(e).__name__
            dt = time.time() - t0
            with self._cond:
                st = self.stats["motores"][motor.nombre]
                st["turnos"] += 1
                st["segundos"] += dt
                if motivo == "timeout":
                    st["timeouts"] += 1
                elif error:
                    st["errores"] += 1

                if linea:
                    muestra.historial.append((rol, linea))
                    muestra.paso += 1
                    self.evento(motor.nombre, "fin", muestra.idx, rol, dt)
                    self._encolar(muestra, motor.nombre)
                elif reintentable and tarea.intentos + 1 < self.max_intentos:
                    # R21: un timeout es infra -> REINTENTO A OTRO MOTOR, no descarte.
                    tarea.intentos += 1
                    tarea.excluidos = tarea.excluidos | {motor.nombre}
                    self._cola.append(tarea)
                    self.evento(motor.nombre, "retry", muestra.idx, rol, dt,
                                extra=f"intento {tarea.intentos}")
                else:
                    self.evento(motor.nombre, "descarte", muestra.idx, rol, dt, extra=motivo)
                    self._descartar(muestra, motivo or "turno_vacio", rol,
                                    f"{dt:.1f}s", motor.nombre)
                self._ocupado.discard(motor.nombre)
                self._cond.notify_all()

    def correr(self) -> dict[str, Any]:
        with self._cond:
            for m in self._muestras_iniciales():
                self._encolar(m)
        hilos = [threading.Thread(target=self._consumidor, args=(m,), daemon=True,
                                  name=f"motor-{m.nombre}") for m in self.motores]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()
        self._log.close()
        self._shards.cerrar()
        self._escribir_stats()
        return self.stats

    def _muestras_iniciales(self) -> list[Muestra]:
        return list(self.muestras)

    def _escribir_stats(self) -> None:
        self.stats["segundos_totales"] = round(time.time() - self._t0, 2)
        (self.out_dir / "roles.stats.json").write_text(
            json.dumps(self.stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Shards:
    """Un archivo por motor, abierto en modo append y con `flush` por item.

    El punto: si matan la corrida a mitad, lo ya terminado esta en disco. Un
    instrumento que escribe al final pierde TODO.
    """

    def __init__(self, dir_: Path) -> None:
        self.dir = dir_
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fh: dict[str, Any] = {}
        self._n: dict[str, int] = {}

    def _de(self, motor: str):
        if motor not in self._fh:
            self._fh[motor] = (self.dir / f"{motor}.jsonl").open("a", encoding="utf-8")
        return self._fh[motor]

    def escribir(self, fila: dict[str, Any], motor: str = "cola") -> None:
        fh = self._de(motor)
        fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
        fh.flush()
        self._n[motor] = self._n.get(motor, 0) + 1

    def cerrar(self) -> None:
        for fh in self._fh.values():
            fh.flush()
            fh.close()
        self._fh.clear()


# --------------------------------------------------------------------------- #
# Corrida
# --------------------------------------------------------------------------- #
def elegir_escenarios(n: int, seed: int | None, solo: list[str] | None) -> list[dict]:
    if solo:
        por_id = {sc["id"]: sc for sc in ref.SCENARIOS}
        faltan = [i for i in solo if i not in por_id]
        if faltan:
            raise SystemExit(f"escenario inexistente: {faltan}")
        return [por_id[i] for i in solo]
    rng = random.Random(seed)
    return [rng.choice(ref.SCENARIOS) for _ in range(n)]


def preparar_muestras(escenarios: list[dict], voice: str) -> list[Muestra]:
    return [Muestra(idx=i, sc=sc, voice=voice, n_ex=n_ex_de(sc),
                    secuencia=secuencia(sc, n_ex_de(sc)))
            for i, sc in enumerate(escenarios)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ROLES: division de roles + scheduler sin barrera")
    ap.add_argument("--config", required=True, help="JSON con la lista de motores")
    ap.add_argument("--scenario", action="append", default=None,
                    help="id de escenario, repetible")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--voice", default="seco")
    ap.add_argument("--out", default="out/dataset/roles")
    ap.add_argument("--sin-registro", action="store_true")
    ap.add_argument("--max-intentos", type=int, default=MAX_INTENTOS_ITEM)
    args = ap.parse_args(argv)

    motores = cargar_motores(args.config)
    escenarios = elegir_escenarios(args.n, args.seed, args.scenario)
    muestras = preparar_muestras(escenarios, args.voice)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"roles: {len(muestras)} muestras; motores: "
          f"{[m.nombre for m in motores]}", flush=True)
    prog = Programador(motores, muestras, args.voice, out,
                       usar_registro=not args.sin_registro, max_intentos=args.max_intentos)
    stats = prog.correr()
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
