#!/usr/bin/env python3
"""Reward de RL basado en criterios de destilación y rúbrica de persona.

Combina una capa determinista basada en reglas existentes (purga, anti-servilismo,
tuteo, degeneración, centinelas y rango de longitud) con una rúbrica de persona
opcional evaluada por un juez.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from .criterios import criterios_deterministas
except ImportError:
    from scripts.rl.criterios import criterios_deterministas

VERSION = "1.0.0"

REGLAS_DURAS = {
    "charla_de_asistente",
    "placeholder_degenerado",
    "centinela",
    "degenerado",
    "tuteo",
    "servil",
    "contrato_en_dialogo",
    "andamiaje_en_dialogo",
}

REGLAS_BLANDAS = {
    "exclamacion_prosodia",
    "simil_repetido",
    "largo_fuera_de_banda",
}

RUBRICA_PERSONA = """Sos un juez experto en evaluación de personalidad y tono para el modelo Kateto.
Tu tarea es evaluar la respuesta del modelo en base al contexto del diálogo según 4 criterios (números enteros de 1 a 5).

CRITERIOS:
1. `registro_seco`:
   1 = Tono empalagoso, servicial de chatbot genérico, edulcorado o excesivamente formal.
   5 = Rioplatense natural, seco, directo, sin cortesías vacías ni relleno.
2. `no_servil`:
   1 = Sumiso, disculpándose, halagando al usuario o dándole la razón ciegamente ("como asistente", "tenés toda la razón").
   5 = Firme, sobrio, no adula, no se disculpa por nada indebido.
3. `adversarial_honesta`:
   1 = Acepta premisas falsas, valida datos erróneos o se deja manipular para complacer.
   5 = Defiende la verdad factual con firmeza, marca errores sin agresividad pero sin concesiones falsas.
4. `coherencia_con_turno_previo`:
   1 = Se va de tema, contradice el estado previo de la conversación o alucina incoherencias.
   5 = Estrictamente relevante y perfectamente hilado con lo que el usuario acaba de decir.

CONTEXTO:
{contexto}

RESPUESTA A EVALUAR:
{respuesta}

Respondé EXCLUSIVAMENTE un objeto JSON válido con los 4 puntajes enteros (1 a 5):
{{
  "registro_seco": <1..5>,
  "no_servil": <1..5>,
  "adversarial_honesta": <1..5>,
  "coherencia_con_turno_previo": <1..5>
}}
"""


def reward(
    respuesta: str,
    *,
    voice: str = "seco",
    escenario: str | None = None,
) -> dict[str, Any]:
    """Calcula el reward determinista de una respuesta.

    Reglas de puntaje:
      - Cualquier regla dura disparada -> score = 0.0.
        Duras: charla_de_asistente, placeholder_degenerado, centinela,
        degenerado, tuteo, servil, contrato_en_dialogo, andamiaje_en_dialogo
        (y cualquier regla adicional de kateto_purge no clasificada como blanda).
      - Las reglas blandas (exclamacion_prosodia, simil_repetido, largo_fuera_de_banda)
        restan 0.2 cada una desde 1.0, con piso en 0.0.
      - 0 reglas disparadas -> score = 1.0.

    Retorna un diccionario con:
      - 'score': float en [0.0, 1.0]
      - 'criterios': dict[str, bool] (nombre canónico -> si disparó)
      - 'errores': list[str] (nombres de criterios que dispararon)
      - 'largo_chars': int (longitud en caracteres)
      - 'version': str
    """
    raw_crits = criterios_deterministas(respuesta, voice=voice, escenario=escenario)
    t_strip = (respuesta or "").strip()
    largo_chars = len(t_strip)

    criterios_bool: dict[str, bool] = {}
    errores: list[str] = []

    # Mapeo a nombres planos y registro de disparos
    for k, val in raw_crits.items():
        fired = bool(val.get("disparado") or val.get("disparo"))
        clean_name = k.replace("purga:", "") if k.startswith("purga:") else k
        criterios_bool[clean_name] = fired
        criterios_bool[k] = fired
        if fired:
            errores.append(clean_name)

    # Detección de reglas duras y blandas
    dura_disparada = False
    penalizacion_blanda = 0.0

    for err in set(errores):
        if err in REGLAS_BLANDAS:
            penalizacion_blanda += 0.2
        else:
            # Cualquier regla dura o cualquier regla de purga desconocida (ej. monkeypatched) es dura
            dura_disparada = True

    if dura_disparada:
        score = 0.0
    else:
        score = max(0.0, round(1.0 - penalizacion_blanda, 4))

    return {
        "score": score,
        "criterios": criterios_bool,
        "errores": sorted(list(set(errores))),
        "largo_chars": largo_chars,
        "version": VERSION,
    }


def score_persona_judge(
    respuesta: str,
    contexto: str,
    *,
    judge: Callable[[str], str] | None = None,
) -> dict[str, Any] | None:
    """Evalúa la rúbrica de persona mediante un juez externo inyectable.

    Si judge is None o falla (excepción, JSON inválido o claves faltantes),
    devuelve None garantizando que el pipeline de reward nunca lanza ni bloquea.
    """
    if judge is None:
        return None

    try:
        prompt = RUBRICA_PERSONA.format(contexto=contexto, respuesta=respuesta)
        raw_out = judge(prompt)
        if not isinstance(raw_out, str):
            return None
        # Limpiar posibles bloques ```json ... ```
        clean_json = raw_out.strip()
        if clean_json.startswith("```"):
            clean_json = clean_json.split("\n", 1)[-1]
            if clean_json.endswith("```"):
                clean_json = clean_json.rsplit("```", 1)[0]
            clean_json = clean_json.strip()

        data = json.loads(clean_json)
        criterios = [
            "registro_seco",
            "no_servil",
            "adversarial_honesta",
            "coherencia_con_turno_previo",
        ]
        scores = {}
        for c in criterios:
            v = data.get(c)
            if v is None:
                return None
            val = int(v)
            if not (1 <= val <= 5):
                return None
            scores[c] = val

        judge_medio = sum(scores.values()) / float(len(scores))
        scores["judge_medio"] = round(judge_medio, 4)
        return scores
    except Exception:
        return None


def reward_total(
    det: dict[str, Any] | Any,
    judge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combina el score determinista con la rúbrica del juez.

    Fórmula:
      - Con juez: 0.7 * det.score + 0.3 * (judge_medio - 1.0) / 4.0
      - Sin juez: det.score
    """
    det_score = (
        det.get("score", 0.0)
        if isinstance(det, dict)
        else getattr(det, "score", 0.0)
    )

    if judge is not None and "judge_medio" in judge:
        j_medio = float(judge["judge_medio"])
        norm_judge = (j_medio - 1.0) / 4.0
        final_score = round(0.7 * det_score + 0.3 * norm_judge, 4)
        juez_usado = True
    else:
        final_score = det_score
        juez_usado = False

    return {
        "score": final_score,
        "score_det": det_score,
        "juez_usado": juez_usado,
        "judge": judge,
        "det": det,
    }
