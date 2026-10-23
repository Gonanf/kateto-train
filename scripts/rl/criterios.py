#!/usr/bin/env python3
"""Mapa de criterios deterministas para evaluación y reward de RL.

Fuente única de criterios: reusa directamente las reglas y validadores existentes
en producción sin duplicar expresiones regulares ni reimplementar lógica.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path
from typing import Any

# Asegurar acceso a pipelines y scripts del workspace local.
# Los módulos requeridos por las reglas deterministas (kateto_purge, gen_orpo_v8, etc.)
# residen en los subdirectorios pipelines/ y scripts/ del repositorio.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for p in [_ROOT / "pipelines", _ROOT / "scripts"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import kateto_purge as kp
from analyze_eval29_coherence import check_degenerate
import gen_toolcalling_dataset as ref

try:
    import gen_orpo_v8 as v8
except ImportError as e:
    raise ImportError(
        "Falta scripts/gen_orpo_v8.py; el reward no puede correr sin él (criterio servil)."
    ) from e


def wrap_turno(respuesta: str, voice: str = "seco") -> str:
    """Envuelve la respuesta como turno de voz dentro de un diálogo simulado.

    `kateto_purge.evaluar` sólo analiza los turnos pertenecientes a la VOZ
    (por defecto `seco`), ignorando el texto suelto o los turnos de otros
    participantes (ej. `user`). Sin este envoltorio, reglas críticas como
    `charla_de_asistente` nunca disparan.
    """
    return f"<|im_start|>user\nChe, contame.<|im_end|>\n<|im_start|>{voice}\n{respuesta}<|im_end|>\n"


def _check_tuteo(texto: str) -> tuple[bool, str | None]:
    """Detecta tuteo usando la misma lógica de gen_toolcalling_dataset._tuteo_en."""
    t_norm = unicodedata.normalize("NFC", texto or "")
    m = ref.RX_TUTEO_INEQUIVOCO.search(t_norm)
    if m:
        return True, m.group(1).lower()
    for m in ref.RX_TUTEO_HOMOGRAFO.finditer(t_norm):
        if ref.RX_TUTEO_GUARDA.search(t_norm[: m.start()]):
            continue
        return True, m.group(1).lower()
    return False, None


def criterios_deterministas(
    respuesta: str,
    *,
    voice: str = "seco",
    escenario: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Evalúa todos los criterios deterministas sobre una respuesta de modelo.

    Retorna un diccionario mapeando cada criterio a un dict con:
      - 'disparado': bool (True si violó el criterio / disparó penalización)
      - 'disparo': bool (alias idéntico a 'disparado')
      - 'detalle': información adicional sobre la coincidencia
    """
    t = respuesta or ""
    t_strip = t.strip()
    res: dict[str, dict[str, Any]] = {}

    # 1. Reglas de kateto_purge (evaluadas sobre la respuesta envuelta en turno de voz)
    wrapped = wrap_turno(t, voice=voice)
    purga_disparadas = set(kp.evaluar(wrapped))
    for nombre in kp.REGLAS:
        fired = nombre in purga_disparadas
        res[f"purga:{nombre}"] = {
            "disparado": fired,
            "disparo": fired,
            "detalle": nombre if fired else None,
        }
    # En caso de reglas dinámicas o monkeypatched en kp.REGLAS
    for extra in purga_disparadas:
        k = f"purga:{extra}"
        if k not in res:
            res[k] = {"disparado": True, "disparo": True, "detalle": extra}

    # 2. Servilismo (gen_orpo_v8.RX_SERVIL)
    # Se importa gen_orpo_v8 de forma dura: la regla servil se calcula siempre
    # sin degradar silenciosamente a un estado permisivo ante imports faltantes.
    # El patrón RX_SERVIL identifica frases complacientes típicas de chatbot.
    # Esta verificación garantiza que las respuestas serviles o sumisas obtengan
    # de manera determinista un score de 0.0 sin requerir evaluación semántica.
    # Sin este filtro léxico, la separación total del reward cae de 63.6% a 8.4%.
    # El módulo scripts/gen_orpo_v8.py se encuentra versionado desde el commit 08f0ebd.
    # Criterio determinista estricto sin tolerancia a servilismo de asistente.
    m_serv = v8.RX_SERVIL.search(t)
    fired_serv = bool(m_serv)
    res["servil"] = {
        "disparado": fired_serv,
        "disparo": fired_serv,
        "detalle": m_serv.group(0) if m_serv else None,
    }

    # 3. Tuteo (gen_toolcalling_dataset)
    fired_tut, match_tut = _check_tuteo(t)
    res["tuteo"] = {
        "disparado": fired_tut,
        "disparo": fired_tut,
        "detalle": match_tut,
    }

    # 4. Degeneración / bucle (analyze_eval29_coherence.check_degenerate)
    is_deg, rep_count = check_degenerate(t)
    res["degenerado"] = {
        "disparado": is_deg,
        "disparo": is_deg,
        "detalle": f"max_rep={rep_count}" if is_deg else None,
    }

    # 5. Centinela (vacío o prefijo de espera / no-response)
    is_cent = (
        not t_strip
        or t_strip.startswith("<|wait|>")
        or t_strip.startswith("<|no_response|>")
    )
    res["centinela"] = {
        "disparado": is_cent,
        "disparo": is_cent,
        "detalle": (
            "vacio"
            if not t_strip
            else (
                "<|wait|>"
                if t_strip.startswith("<|wait|>")
                else "<|no_response|>" if t_strip.startswith("<|no_response|>") else None
            )
        ),
    }

    # 6. Largo fuera de banda (< 40 o > 900 chars)
    largo = len(t_strip)
    fuera_de_banda = largo < 40 or largo > 900
    res["largo_fuera_de_banda"] = {
        "disparado": fuera_de_banda,
        "disparo": fuera_de_banda,
        "detalle": f"{largo} chars (esperado 40..900)" if fuera_de_banda else None,
    }

    return res
