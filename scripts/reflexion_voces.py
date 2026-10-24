#!/usr/bin/env python3
"""
Reflexión de Voces (card t_d21f3f4a)
Job periódico que analiza la memoria real de cada Voz y genera perfiles de personalidad
con citas verificables a la evidencia.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
from types import SimpleNamespace
import urllib.error
import urllib.request


DEFAULT_VOICES = ["conquest", "doktor", "jane", "whisperer"]
NAME_TO_VOICE = {
    "jane": "jane",
    "doktor": "doktor",
    "whisperer": "whisperer",
    "conquest": "conquest",
}

TURN_HEADER_RE = re.compile(
    r"^\s*\*\*(Jane|Doktor|Whisperer|Conquest)\*\*"
    r"(?:\s*\((?:argument|objection|rebuttal|ruling)\))?"
    r":?\s*(.*)$",
    re.IGNORECASE,
)


class CorridaLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, paso: str, mensaje: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        line = f"{ts} [{paso.upper()}] {mensaje}\n"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line)


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def extract_date(text: str) -> str:
    m = re.search(r"\b(20\d\d[-/]\d\d[-/]\d\d)\b", text)
    if m:
        return m.group(1).replace("/", "-")
    m2 = re.search(r"\b(20\d\d\d\d\d\d)\b", text)
    if m2:
        val = m2.group(1)
        return f"{val[:4]}-{val[4:6]}-{val[6:8]}"
    return ""


def compute_item_id(fuente: str, texto: str, tipo: str) -> str:
    tipo_prefix = {"debate": "d", "journal": "j", "sesion": "s"}.get(tipo, "x")
    key = f"{fuente}|{texto}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{tipo_prefix}-{h}"


def extract_and_parse_json(raw: str) -> dict:
    text = raw.strip()
    if "```" in text:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if m:
            text = m.group(1).strip()
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1]
    return json.loads(text)


def load_debate_evidence(config_dir: Path, max_chars_item: int) -> dict[str, list[dict]]:
    evidence: dict[str, list[dict]] = {v: [] for v in DEFAULT_VOICES}
    registry_dir = config_dir / "bate_debate" / "registry"
    if not registry_dir.exists():
        return evidence

    for file_path in sorted(registry_dir.glob("debate-*.md")):
        try:
            rel_path = file_path.relative_to(config_dir).as_posix()
        except ValueError:
            rel_path = file_path.name

        file_date = extract_date(file_path.name)
        lines = file_path.read_text(encoding="utf-8").splitlines()
        if not file_date:
            for l in lines[:10]:
                d = extract_date(l)
                if d:
                    file_date = d
                    break

        current_voice: str | None = None
        current_line: int | None = None
        current_texts: list[str] = []

        def flush():
            nonlocal current_voice, current_line, current_texts
            if current_voice and current_line is not None and current_texts:
                full_text = normalize_whitespace(" ".join(current_texts))[:max_chars_item]
                if full_text:
                    fuente = f"{rel_path}:{current_line}"
                    tipo = "debate"
                    item_id = compute_item_id(fuente, full_text, tipo)
                    evidence[current_voice].append({
                        "id": item_id,
                        "fuente": fuente,
                        "tipo": tipo,
                        "fecha": file_date,
                        "texto": full_text,
                    })
            current_voice = None
            current_line = None
            current_texts = []

        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            m = TURN_HEADER_RE.match(stripped)
            if m:
                flush()
                v_name = m.group(1).lower()
                current_voice = NAME_TO_VOICE.get(v_name)
                current_line = idx
                rest = m.group(2).strip() if m.group(2) else ""
                if rest:
                    current_texts.append(rest)
                continue

            if current_voice:
                if not stripped:
                    flush()
                elif stripped.startswith("#") or stripped.startswith("**Participantes"):
                    flush()
                else:
                    current_texts.append(stripped)

        flush()

    return evidence


def load_journal_evidence(
    config_dir: Path, max_chars_item: int
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    evidence: dict[str, list[dict]] = {v: [] for v in DEFAULT_VOICES}
    ruido: dict[str, int] = {v: 0 for v in DEFAULT_VOICES}
    voices_dir = config_dir / "voices"
    if not voices_dir.exists():
        return evidence, ruido

    for v in DEFAULT_VOICES:
        journal_path = voices_dir / v / "JOURNAL.md"
        if not journal_path.exists():
            continue

        try:
            rel_path = journal_path.relative_to(config_dir).as_posix()
        except ValueError:
            rel_path = journal_path.name

        lines = journal_path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            s = line.strip()
            if not s:
                continue
            if s.startswith("[auto] voice_idle at") or "[auto] voice_idle at" in s:
                ruido[v] += 1
            else:
                fuente = f"{rel_path}:{idx}"
                tipo = "journal"
                texto = normalize_whitespace(s)[:max_chars_item]
                fecha = extract_date(s) or extract_date(journal_path.name)
                item_id = compute_item_id(fuente, texto, tipo)
                evidence[v].append({
                    "id": item_id,
                    "fuente": fuente,
                    "tipo": tipo,
                    "fecha": fecha,
                    "texto": texto,
                })

    return evidence, ruido


def load_session_evidence(config_dir: Path, max_chars_item: int) -> dict[str, list[dict]]:
    evidence: dict[str, list[dict]] = {v: [] for v in DEFAULT_VOICES}
    logs_dir = config_dir / "logs"
    if not logs_dir.exists():
        return evidence

    for file_path in sorted(logs_dir.glob("session-*.jsonl")):
        try:
            rel_path = file_path.relative_to(config_dir).as_posix()
        except ValueError:
            rel_path = file_path.name

        lines = file_path.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                data = json.loads(s)
            except Exception:
                continue

            if data.get("event") == "voice_turn":
                raw_voice = str(data.get("voice", "")).lower().strip()
                v = NAME_TO_VOICE.get(raw_voice)
                raw_text = str(data.get("text", "")).strip()
                if v and raw_text:
                    fuente = f"{rel_path}:{idx}"
                    tipo = "sesion"
                    texto = normalize_whitespace(raw_text)[:max_chars_item]
                    fecha = (
                        extract_date(str(data.get("ts", "")))
                        or extract_date(str(data.get("timestamp", "")))
                        or extract_date(str(data.get("created_at", "")))
                        or extract_date(file_path.name)
                    )
                    item_id = compute_item_id(fuente, texto, tipo)
                    evidence[v].append({
                        "id": item_id,
                        "fuente": fuente,
                        "tipo": tipo,
                        "fecha": fecha,
                        "texto": texto,
                    })

    return evidence


def load_soul(config_dir: Path, voz: str) -> str:
    soul_path = config_dir / "voices" / voz / "SOUL.md"
    if soul_path.exists():
        try:
            return soul_path.read_text(encoding="utf-8").strip()[:800]
        except Exception:
            return ""
    return ""


def build_bundle_for_voice(raw_items: list[dict], max_items: int) -> list[dict]:
    # Deduplicate by item ID
    seen_ids = set()
    deduped = []
    for item in raw_items:
        if item["id"] not in seen_ids:
            seen_ids.add(item["id"])
            deduped.append(item)

    # Sort deterministically by (tipo, fuente)
    deduped.sort(key=lambda x: (x["tipo"], x["fuente"]))

    n = len(deduped)
    step = max(1, n // max_items)
    selected = deduped[::step][:max_items]
    return selected


def call_llm(
    base_url: str,
    model: str,
    messages: list[dict],
    api_key: str = "",
    timeout: float = 600.0,
    seed: int = 7,
    temperature: float = 0.2,
    max_tokens: int = 900,
) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        resp_json = json.loads(body)
        return resp_json["choices"][0]["message"]["content"]


SYSTEM_PROMPT_PERFIL = (
    "Sos un analista de personalidad lingüística. Tu tarea es analizar la memoria y evidencia real de una voz "
    "y sintetizar su perfil con citas verificables a los IDs provistos.\n"
    "Reglas estrictas:\n"
    "1. Devolvé ÚNICAMENTE un objeto JSON válido, sin bloques de código ni texto adicional.\n"
    "2. El perfil debe contener entre 3 y 8 rasgos ('rasgos').\n"
    "3. Cada rasgo debe citar entre 1 y 3 IDs de evidencia existentes en el bundle recibido ('ev': ['id1', ...]).\n"
    "4. Prohibido inventar hechos o citas que no figuren en la evidencia.\n"
    "5. El perfil describe CÓMO HABLA Y QUÉ DEFIENDE, no qué es ontológicamente.\n"
    "6. Prohibido usar frases como 'es una IA' o adjetivos vacíos como 'carismática' o 'única'.\n"
    "7. 'lexico': hasta 8 expresiones textuales características que la voz usa.\n"
    "8. 'evita': hasta 5 cosas que la voz no hace o evita.\n"
    "9. 'resumen': síntesis de hasta 600 caracteres.\n"
    "Formato JSON requerido:\n"
    "{\n"
    '  "interesante": ["...", "..."],\n'
    '  "perfil": {\n'
    '    "resumen": "≤600 chars",\n'
    '    "rasgos": [{"rasgo": "≤200 chars", "ev": ["id"]}],\n'
    '    "lexico": ["..."],\n'
    '    "evita": ["..."]\n'
    "  }\n"
    "}"
)

SYSTEM_PROMPT_NACIMIENTO = (
    "Sos un arquitecto de diseño de agentes y ensambles de debate. "
    "Tu tarea es evaluar las estadísticas de cobertura del corpus y los rasgos de las voces actuales "
    "para decidir si se justifica el nacimiento de una nueva voz que cubra un hueco genuino.\n"
    "Reglas estrictas:\n"
    "1. Devolvé ÚNICAMENTE un objeto JSON válido.\n"
    "2. 'debe_nacer': booleano (true o false).\n"
    "3. 'motivo': justificación concisa de hasta 400 caracteres.\n"
    "4. 'propuesta': objeto con:\n"
    "   - 'voz_id': string en minúsculas sin espacios (ej: 'fiscal', 'artesano').\n"
    "   - 'descripcion': descripción concisa de hasta 400 caracteres.\n"
    "   - 'evidencia_de_hueco': lista de IDs de evidencia del corpus que muestran el vacío temático o metodológico.\n"
    "Formato exacto:\n"
    "{\n"
    '  "debe_nacer": bool,\n'
    '  "motivo": "≤400 chars",\n'
    '  "propuesta": {\n'
    '    "voz_id": "lowercase sin espacios",\n'
    '    "descripcion": "≤400 chars",\n'
    '    "evidencia_de_hueco": ["id"]\n'
    "  }\n"
    "}"
)


def get_voice_profile(
    voz: str,
    soul: str,
    bundle: list[dict],
    llm_fn,
    logger: CorridaLogger,
) -> tuple[dict | None, str | None]:
    bundle_lines = [
        f"[{item['id']}] ({item['tipo']}, {item['fuente']}): {item['texto']}" for item in bundle
    ]
    user_prompt = (
        f"Voz a analizar: {voz}\n\n"
        f"Contexto previo (SOUL, referencia de estilo, NO citar):\n{soul}\n\n"
        f"Bundle de evidencia disponible (usar SOLO estos IDs en 'ev'):\n"
        + "\n".join(bundle_lines)
        + "\n\nGenerá el JSON con el análisis solicitado."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_PERFIL},
        {"role": "user", "content": user_prompt},
    ]

    last_error = None
    for attempt in range(2):
        logger.log("llamada", f"Solicitando perfil LLM para '{voz}' (intento {attempt + 1})")
        try:
            raw = llm_fn(messages)
            data = extract_and_parse_json(raw)
            if isinstance(data, dict):
                return data, None
            last_error = f"Formato no es dict: {type(data)}"
        except Exception as e:
            last_error = str(e)
            logger.log("llamada", f"Error en respuesta LLM para '{voz}' (intento {attempt + 1}): {e}")
            if attempt == 0:
                messages.append({
                    "role": "user",
                    "content": "Tu respuesta anterior no fue un JSON válido. Por favor, respondé ÚNICAMENTE con el objeto JSON solicitado.",
                })

    return None, last_error


def validate_profile(
    raw_data: dict, bundle: list[dict], logger: CorridaLogger, voz: str
) -> dict:
    perfil = raw_data.get("perfil", {}) if isinstance(raw_data, dict) else {}
    interesante = raw_data.get("interesante", []) if isinstance(raw_data, dict) else []
    if not isinstance(interesante, list):
        interesante = []

    resumen = str(perfil.get("resumen", ""))
    if len(resumen) > 600:
        logger.log("validacion", f"Resumen de '{voz}' excedió 600 chars; truncado.")
        resumen = resumen[:600]

    lexico = perfil.get("lexico", [])
    if not isinstance(lexico, list):
        lexico = []
    lexico = [str(x) for x in lexico][:8]

    evita = perfil.get("evita", [])
    if not isinstance(evita, list):
        evita = []
    evita = [str(x) for x in evita][:5]

    bundle_id_set = {item["id"] for item in bundle}
    raw_rasgos = perfil.get("rasgos", [])
    if not isinstance(raw_rasgos, list):
        raw_rasgos = []

    rasgos_validos = []
    descartados = []

    for r in raw_rasgos:
        if not isinstance(r, dict):
            descartados.append({
                "rasgo": str(r)[:200],
                "ev": [],
                "motivo": "sin_evidencia",
            })
            continue

        rasgo_txt = str(r.get("rasgo", "")).strip()[:200]
        ev = r.get("ev", [])
        if not isinstance(ev, list) or len(ev) == 0:
            descartados.append({
                "rasgo": rasgo_txt,
                "ev": ev if isinstance(ev, list) else [],
                "motivo": "sin_evidencia",
            })
            continue

        missing = [eid for eid in ev if eid not in bundle_id_set]
        if missing:
            descartados.append({
                "rasgo": rasgo_txt,
                "ev": ev,
                "motivo": f"id_inexistente: {missing[0]}",
            })
            continue

        rasgos_validos.append({
            "rasgo": rasgo_txt,
            "ev": ev,
        })

    rasgos_totales = len(rasgos_validos) + len(descartados)
    valid_count = len(rasgos_validos)
    tasa = round(valid_count / rasgos_totales, 4) if rasgos_totales > 0 else 0.0

    if not (3 <= valid_count <= 8):
        logger.log("validacion", f"Voz '{voz}': rasgos válidos ({valid_count}) fuera de rango [3, 8]")

    valido = valid_count >= 3

    return {
        "valido": valido,
        "resumen": resumen,
        "rasgos": rasgos_validos,
        "descartados": descartados,
        "lexico": lexico,
        "evita": evita,
        "interesante": interesante,
        "cobertura": {
            "rasgos_totales": rasgos_totales,
            "rasgos_validos": valid_count,
            "tasa": tasa,
        },
    }


def evaluate_nacimiento(
    perfiles_info: dict[str, dict],
    llm_fn,
    logger: CorridaLogger,
) -> dict:
    stats_lines = []
    all_bundle_ids = []
    for v, p in perfiles_info.items():
        stats_lines.append(
            f"- Voz '{v}': {p.get('bundle_items', 0)} items en bundle, "
            f"tasa cobertura: {p.get('cobertura', {}).get('tasa', 0.0)}"
        )
        rasgos = [r.get("rasgo", "") for r in p.get("rasgos", [])[:3]]
        if rasgos:
            stats_lines.append(f"  Rasgos principales: " + "; ".join(rasgos))
        for r in p.get("rasgos", []):
            all_bundle_ids.extend(r.get("ev", []))

    user_prompt = (
        "Estadísticas del corpus actual de voces:\n"
        + "\n".join(stats_lines)
        + "\n\nEvaluá si debe nacer una nueva voz y respondé con el JSON requerido."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_NACIMIENTO},
        {"role": "user", "content": user_prompt},
    ]

    logger.log("llamada", "Solicitando evaluación de nacimiento al LLM")
    try:
        raw = llm_fn(messages)
        data = extract_and_parse_json(raw)
    except Exception as e:
        logger.log("validacion", f"Error en respuesta de nacimiento: {e}")
        return {
            "valido": False,
            "debe_nacer": False,
            "motivo": f"Error parseando JSON: {e}",
            "propuesta": None,
        }

    if not isinstance(data, dict):
        return {
            "valido": False,
            "debe_nacer": False,
            "motivo": "Respuesta no es objeto JSON",
            "propuesta": None,
        }

    raw_debe = data.get("debe_nacer")
    if not isinstance(raw_debe, bool):
        return {
            "valido": False,
            "debe_nacer": False,
            "motivo": "Campo 'debe_nacer' no es booleano",
            "propuesta": None,
        }

    motivo = str(data.get("motivo", ""))[:400]
    propuesta = data.get("propuesta")

    if raw_debe:
        if not isinstance(propuesta, dict):
            return {
                "valido": False,
                "debe_nacer": False,
                "motivo": "Propuesta inválida para nacimiento afirmativo",
                "propuesta": None,
            }
        voz_id = str(propuesta.get("voz_id", "")).strip().lower()
        if not voz_id or " " in voz_id:
            return {
                "valido": False,
                "debe_nacer": False,
                "motivo": "voz_id debe ser lowercase sin espacios",
                "propuesta": None,
            }
        desc = str(propuesta.get("descripcion", ""))[:400]
        ev_hueco = propuesta.get("evidencia_de_hueco", [])
        if not isinstance(ev_hueco, list):
            ev_hueco = []
        return {
            "valido": True,
            "debe_nacer": True,
            "motivo": motivo,
            "propuesta": {
                "voz_id": voz_id,
                "descripcion": desc,
                "evidencia_de_hueco": ev_hueco,
            },
        }

    return {
        "valido": True,
        "debe_nacer": False,
        "motivo": motivo,
        "propuesta": propuesta if isinstance(propuesta, dict) else None,
    }


def write_outputs(
    out_dir: Path,
    perfiles: dict[str, dict],
    nacimiento_res: dict | None,
    cobertura_res: dict,
    logger: CorridaLogger,
) -> None:
    perfiles_dir = out_dir / "perfiles"
    perfiles_dir.mkdir(parents=True, exist_ok=True)

    for v, p in perfiles.items():
        logger.log("escritura", f"Escribiendo perfiles/{v}.json y perfiles/{v}.md")
        json_path = perfiles_dir / f"{v}.json"
        json_path.write_text(json.dumps(p, indent=2, ensure_ascii=False), encoding="utf-8")

        md_lines = [
            f"# Perfil de Personalidad: {v.capitalize()}",
            "",
            "## Resumen",
            p.get("resumen", ""),
            "",
            "## Rasgos Característicos",
        ]
        rasgos = p.get("rasgos", [])
        if rasgos:
            for r in rasgos:
                ev_str = ", ".join(r.get("ev", []))
                md_lines.append(f"- {r.get('rasgo', '')} [ev: {ev_str}]")
        else:
            md_lines.append("*Sin rasgos validados.*")

        md_lines.extend(["", "## Léxico Característico"])
        lexico = p.get("lexico", [])
        if lexico:
            for term in lexico:
                md_lines.append(f"- {term}")
        else:
            md_lines.append("*Sin registro léxico.*")

        md_lines.extend(["", "## Evita"])
        evita = p.get("evita", [])
        if evita:
            for term in evita:
                md_lines.append(f"- {term}")
        else:
            md_lines.append("*Sin restricciones registradas.*")

        md_lines.append("")
        md_path = perfiles_dir / f"{v}.md"
        md_path.write_text("\n".join(md_lines), encoding="utf-8")

    if nacimiento_res is not None:
        logger.log("escritura", "Escribiendo nacimiento.json")
        nac_path = out_dir / "nacimiento.json"
        nac_path.write_text(
            json.dumps(nacimiento_res, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    logger.log("escritura", "Escribiendo cobertura.json")
    cob_path = out_dir / "cobertura.json"
    cob_path.write_text(
        json.dumps(cobertura_res, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_pipeline(
    config_dir: Path,
    out_dir: Path,
    base_url: str = "http://127.0.0.1:11434/v1",
    model: str = "Qwen3.8-2B",
    api_key: str = "",
    timeout: float = 600.0,
    seed: int = 7,
    max_items: int = 40,
    max_chars_item: int = 500,
    min_items: int = 5,
    solo_voces: list[str] | None = None,
    dry_run: bool = False,
    llm_callable=None,
) -> tuple[dict[str, list[dict]], dict, dict | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = CorridaLogger(out_dir / "corrida.log")

    active_voices = [v.lower().strip() for v in solo_voces] if solo_voces else DEFAULT_VOICES

    logger.log("bundle", f"Cargando evidencia desde {config_dir}")
    deb_ev = load_debate_evidence(config_dir, max_chars_item)
    jnl_ev, jnl_ruido = load_journal_evidence(config_dir, max_chars_item)
    ses_ev = load_session_evidence(config_dir, max_chars_item)

    bundles: dict[str, list[dict]] = {}
    voces_cobertura: dict[str, dict] = {}
    total_items = 0

    for v in active_voices:
        raw_list = deb_ev.get(v, []) + jnl_ev.get(v, []) + ses_ev.get(v, [])
        bundle = build_bundle_for_voice(raw_list, max_items)
        bundles[v] = bundle
        total_items += len(bundle)

        items_por_tipo = {
            "debate": sum(1 for it in bundle if it["tipo"] == "debate"),
            "journal": sum(1 for it in bundle if it["tipo"] == "journal"),
            "sesion": sum(1 for it in bundle if it["tipo"] == "sesion"),
        }

        voces_cobertura[v] = {
            "items": len(bundle),
            "items_por_tipo": items_por_tipo,
            "journal_ruido_descartado": jnl_ruido.get(v, 0),
            "valido": False,
            "tasa_cobertura": 0.0,
            "tasa": 0.0,
        }
        logger.log(
            "bundle",
            f"Voz '{v}': {len(bundle)} items armados (ruido journal descartado: {jnl_ruido.get(v, 0)})",
        )

    if dry_run:
        cobertura_res = {
            "voces": voces_cobertura,
            "total_items": total_items,
            "tasa_cobertura_global": 0.0,
            "nacimiento_valido": False,
            **voces_cobertura,
        }
        (out_dir / "cobertura.json").write_text(
            json.dumps(cobertura_res, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.log("escritura", "Modo dry-run finalizado. Archivo cobertura.json escrito.")
        return bundles, cobertura_res, None

    # LLM execution
    if llm_callable is None:
        def default_llm(msgs):
            return call_llm(
                base_url=base_url,
                model=model,
                messages=msgs,
                api_key=api_key,
                timeout=timeout,
                seed=seed,
            )
        llm_fn = default_llm
    else:
        llm_fn = llm_callable

    perfiles_output: dict[str, dict] = {}
    now_iso = datetime.now(timezone.utc).isoformat()

    for v in active_voices:
        bundle = bundles[v]
        if len(bundle) < min_items:
            logger.log("validacion", f"Voz '{v}' tiene evidencia insuficiente ({len(bundle)} < {min_items})")
            p_dict = {
                "voz": v,
                "generado_en": now_iso,
                "modelo": model,
                "base_url": base_url,
                "semilla": seed,
                "bundle_items": len(bundle),
                "valido": False,
                "evidencia_insuficiente": True,
                "rasgos": [],
                "descartados": [],
                "resumen": "Evidencia insuficiente para inferir perfil.",
                "lexico": [],
                "evita": [],
                "interesante": [],
                "cobertura": {
                    "rasgos_totales": 0,
                    "rasgos_validos": 0,
                    "tasa": 0.0,
                },
            }
            perfiles_output[v] = p_dict
            voces_cobertura[v]["valido"] = False
            voces_cobertura[v]["tasa_cobertura"] = 0.0
            voces_cobertura[v]["tasa"] = 0.0
            continue

        soul = load_soul(config_dir, v)
        raw_profile, err = get_voice_profile(v, soul, bundle, llm_fn, logger)
        if raw_profile is None:
            val_res = {
                "valido": False,
                "resumen": f"Error obteniendo perfil: {err}",
                "rasgos": [],
                "descartados": [],
                "lexico": [],
                "evita": [],
                "interesante": [],
                "cobertura": {"rasgos_totales": 0, "rasgos_validos": 0, "tasa": 0.0},
            }
        else:
            val_res = validate_profile(raw_profile, bundle, logger, v)

        p_dict = {
            "voz": v,
            "generado_en": now_iso,
            "modelo": model,
            "base_url": base_url,
            "semilla": seed,
            "bundle_items": len(bundle),
            "valido": val_res["valido"],
            "rasgos": val_res["rasgos"],
            "descartados": val_res["descartados"],
            "resumen": val_res["resumen"],
            "lexico": val_res["lexico"],
            "evita": val_res["evita"],
            "interesante": val_res["interesante"],
            "cobertura": val_res["cobertura"],
        }
        perfiles_output[v] = p_dict
        voces_cobertura[v]["valido"] = val_res["valido"]
        voces_cobertura[v]["tasa_cobertura"] = val_res["cobertura"]["tasa"]
        voces_cobertura[v]["tasa"] = val_res["cobertura"]["tasa"]

    # Nacimiento evaluation
    nacimiento_res = evaluate_nacimiento(perfiles_output, llm_fn, logger)

    # Cobertura resumen
    total_rasgos = sum(p["cobertura"]["rasgos_totales"] for p in perfiles_output.values())
    valid_rasgos = sum(p["cobertura"]["rasgos_validos"] for p in perfiles_output.values())
    tasa_global = round(valid_rasgos / total_rasgos, 4) if total_rasgos > 0 else 0.0

    cobertura_res = {
        "voces": voces_cobertura,
        "total_items": total_items,
        "tasa_cobertura_global": tasa_global,
        "nacimiento_valido": nacimiento_res.get("valido", False),
        **voces_cobertura,
    }

    write_outputs(out_dir, perfiles_output, nacimiento_res, cobertura_res, logger)
    return bundles, cobertura_res, nacimiento_res


def _llm_stub(messages: list[dict]) -> str:
    user_content = ""
    for m in messages:
        if m.get("role") == "user":
            user_content = m.get("content", "")

    # Nacimiento request check
    if "nacimiento" in user_content.lower() or "estadísticas" in user_content.lower() or "estadisticas" in user_content.lower():
        ids = re.findall(r"\b[djs]-[0-9a-f]{10}\b", user_content)
        hueco_id = ids[0] if ids else "d-1234567890"
        return json.dumps({
            "debe_nacer": True,
            "motivo": "Falta una voz que represente la síntesis pragmática operacional entre Jane y Doktor.",
            "propuesta": {
                "voz_id": "artesano",
                "descripcion": "Enfocado en la implementación práctica y resolución de compromisos operacionales.",
                "evidencia_de_hueco": [hueco_id],
            },
        })

    # Jane profile request
    if "voz a analizar: jane" in user_content.lower() or "voz: jane" in user_content.lower():
        ids = re.findall(r"\b[djs]-[0-9a-f]{10}\b", user_content)
        id1 = ids[0] if len(ids) > 0 else "d-1111111111"
        id2 = ids[1] if len(ids) > 1 else "d-2222222222"
        id3 = ids[2] if len(ids) > 2 else "d-3333333333"
        return json.dumps({
            "interesante": ["Enfoque empírico radical", "Desconfianza en jerarquías centralizadas"],
            "perfil": {
                "resumen": "Defensora rigurosa del empirismo y de estructuras descentralizadas basadas en métricas.",
                "rasgos": [
                    {"rasgo": "Promueve la coordinación descentralizada frente a jerarquías formales rígidas.", "ev": [id1]},
                    {"rasgo": "Argumenta que los protocolos asíncronos mitigan costos de consenso eficientemente.", "ev": [id2]},
                    {"rasgo": "Sostiene que la adaptabilidad distribuida supera penalizaciones de latencia transitoria.", "ev": [id3]},
                    {"rasgo": "Afirma inventos no verificados en el debate empírico.", "ev": ["d-noexiste"]},
                ],
                "lexico": ["descentralizada", "asíncrono", "empírico", "latencia"],
                "evita": ["dogmatismo", "jerarquías sin métricas"],
            },
        })

    # Doktor profile request
    if "voz a analizar: doktor" in user_content.lower() or "voz: doktor" in user_content.lower():
        ids = re.findall(r"\b[djs]-[0-9a-f]{10}\b", user_content)
        id1 = ids[0] if len(ids) > 0 else "d-4444444444"
        id2 = ids[1] if len(ids) > 1 else "d-5555555555"
        return json.dumps({
            "interesante": ["Preocupación por estabilidad y convergencia", "Crítica a costos de consenso"],
            "perfil": {
                "resumen": "Defensor de la formalidad y la predictibilidad jerárquica frente al caos.",
                "rasgos": [
                    {"rasgo": "Sostiene que sin jerarquía formal el costo de consenso colapsa decisiones.", "ev": [id1]},
                    {"rasgo": "Advierte sobre latencias inaceptables de protocolos asíncronos en incertidumbre.", "ev": [id2]},
                ],
                "lexico": ["jerarquía", "consenso", "formal", "latencia"],
                "evita": ["optimismo no sustentado"],
            },
        })

    return json.dumps({"interesante": [], "perfil": {"resumen": "", "rasgos": [], "lexico": [], "evita": []}})


def run_self_check() -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_path = Path(tmp_str)
        cfg = tmp_path / "config"
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"

        # Setup fixture files
        deb_dir = cfg / "bate_debate" / "registry"
        deb_dir.mkdir(parents=True, exist_ok=True)
        (deb_dir / "debate-01.md").write_text(
            "# Debate 1\n\n"
            "**Participantes:**\n- Debater: Jane\n- Debater: Doktor\n\n"
            "**Jane** (argument):\n"
            "La evidencia empírica demuestra que la coordinación descentralizada supera esquemas rígidos.\n\n"
            "**Doktor** (rebuttal):\n"
            "Sin una jerarquía formal, el costo de consenso crece cuadráticamente y colapsa.\n\n"
            "**Jane** (argument):\n"
            "El costo de consenso se mitiga con protocolos asíncronos bien diseñados.\n\n"
            "**Doktor** (rebuttal):\n"
            "Los protocolos asíncronos introducen latencias inaceptables bajo alta incertidumbre.\n",
            encoding="utf-8",
        )
        (deb_dir / "debate-02.md").write_text(
            "# Debate 2\n\n"
            "**Participantes:**\n- Debater: Jane\n- Debater: Doktor\n\n"
            "**Jane** (argument):\n"
            "La adaptabilidad distribuida compensa penalizaciones de latencia transitoria.\n\n"
            "**Doktor** (rebuttal):\n"
            "Esa adaptabilidad es un mito sin un árbitro que asegure convergencia determinística.\n\n"
            "**Jane** (argument):\n"
            "La convergencia determinística puede alcanzarse con algoritmos probabilísticos formales.\n\n"
            "**Doktor** (rebuttal):\n"
            "Las garantías formales fallan en la práctica al violarse supuestos de red.\n",
            encoding="utf-8",
        )

        for v in ["jane", "doktor"]:
            v_dir = cfg / "voices" / v
            v_dir.mkdir(parents=True, exist_ok=True)
            (v_dir / "JOURNAL.md").write_text(
                "[auto] voice_idle at 2026-10-24 10:00:00\n"
                "[auto] voice_idle at 2026-10-24 10:05:00\n"
                "[auto] voice_idle at 2026-10-24 10:10:00\n"
                f"[orchestrator] {v.capitalize()} completó revisión de contratos de red.\n",
                encoding="utf-8",
            )
            (v_dir / "SOUL.md").write_text(
                f"{v.capitalize()} es analítica y fundamenta cada aseveración.",
                encoding="utf-8",
            )

        logs_dir = cfg / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / "session-01.jsonl").write_text(
            '{"event": "user_turn", "text": "Inicio de sesión"}\n'
            '{"event": "voice_turn", "voice": "jane", "text": "Iniciando verificación de consistencia."}\n'
            '{"event": "voice_turn", "voice": "doktor", "text": "Comprobando invariantes del modelo."}\n'
            '{"event": "classifier", "tag": "system"}\n',
            encoding="utf-8",
        )

        # Run 1
        bundles1, cob1, nac1 = run_pipeline(
            config_dir=cfg,
            out_dir=out1,
            solo_voces=["jane", "doktor"],
            llm_callable=_llm_stub,
        )

        # Run 2 (Determinism check)
        bundles2, cob2, nac2 = run_pipeline(
            config_dir=cfg,
            out_dir=out2,
            solo_voces=["jane", "doktor"],
            llm_callable=_llm_stub,
        )

        # Assertion 1: bundle de jane tiene ≥2 items de debate y journal aportó exactamente 1 (3 idle descartadas)
        jane_debate_items = cob1["voces"]["jane"]["items_por_tipo"]["debate"]
        jane_journal_items = cob1["voces"]["jane"]["items_por_tipo"]["journal"]
        jane_idle_ruido = cob1["voces"]["jane"]["journal_ruido_descartado"]
        assert jane_debate_items >= 2, f"Jane debate items ({jane_debate_items}) < 2"
        assert jane_journal_items == 1, f"Jane journal items ({jane_journal_items}) != 1"
        assert jane_idle_ruido == 3, f"Jane idle descartadas ({jane_idle_ruido}) != 3"

        # Assertion 2: rasgo con id inexistente quedó en descartados con motivo id_inexistente: d-noexiste
        jane_profile_file = out1 / "perfiles" / "jane.json"
        assert jane_profile_file.exists(), "perfiles/jane.json no existe"
        jane_prof = json.loads(jane_profile_file.read_text(encoding="utf-8"))
        descartados = jane_prof.get("descartados", [])
        assert any(
            d.get("motivo") == "id_inexistente: d-noexiste" for d in descartados
        ), f"No se encontró descarte por 'id_inexistente: d-noexiste' en {descartados}"

        # Assertion 3: cobertura.jane.tasa == 0.75 y valido is True
        class CoberturaAccessor:
            def __init__(self, data):
                for k, v in data.get("voces", {}).items():
                    setattr(self, k, SimpleNamespace(**v))

        cob_obj = CoberturaAccessor(cob1)
        assert cob_obj.jane.tasa == 0.75, f"cobertura.jane.tasa ({cob_obj.jane.tasa}) != 0.75"
        assert cob_obj.jane.valido is True, f"cobertura.jane.valido ({cob_obj.jane.valido}) is not True"

        # Assertion 4: cobertura.doktor.valido is False (2 rasgos < 3)
        assert cob_obj.doktor.valido is False, f"cobertura.doktor.valido ({cob_obj.doktor.valido}) is not False"

        # Assertion 5: nacimiento.json parsea y debe_nacer es bool
        nac_file = out1 / "nacimiento.json"
        assert nac_file.exists(), "nacimiento.json no existe"
        nac_data = json.loads(nac_file.read_text(encoding="utf-8"))
        assert isinstance(nac_data.get("debe_nacer"), bool), "debe_nacer no es booleano"

        # Assertion 6: los .md existen y contienen [ev:
        jane_md = out1 / "perfiles" / "jane.md"
        doktor_md = out1 / "perfiles" / "doktor.md"
        assert jane_md.exists() and "[ev: " in jane_md.read_text(encoding="utf-8"), "jane.md no cumple formato"
        assert doktor_md.exists() and "[ev: " in doktor_md.read_text(encoding="utf-8"), "doktor.md no cumple formato"

        # Assertion 7: correr dos veces produce ids de evidencia idénticos (determinismo)
        ids1_jane = [it["id"] for it in bundles1["jane"]]
        ids2_jane = [it["id"] for it in bundles2["jane"]]
        ids1_doktor = [it["id"] for it in bundles1["doktor"]]
        ids2_doktor = [it["id"] for it in bundles2["doktor"]]
        assert ids1_jane == ids2_jane, "IDs de Jane no son determinísticos entre corridas"
        assert ids1_doktor == ids2_doktor, "IDs de Doktor no son determinísticos entre corridas"

    print("SELF-CHECK OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reflexión de Voces - Job periódico de análisis de memoria y personalidad"
    )
    parser.add_argument(
        "--config-dir",
        default="~/.config/kateto",
        help="Ruta base de configuración de kateto",
    )
    parser.add_argument(
        "--out",
        default="/run/media/chaos/terciario/proyectos/kateto-train/out/reflexion",
        help="Directorio destino para perfiles, reportes y logs",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434/v1",
        help="URL base de la API OpenAI-compatible",
    )
    parser.add_argument(
        "--model",
        default="Qwen3.8-2B",
        help="Nombre del modelo LLM",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="API key opcional",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout en segundos para llamadas al LLM",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Semilla para respuestas del modelo",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=40,
        help="Cantidad máxima de ítems de evidencia por voz",
    )
    parser.add_argument(
        "--max-chars-item",
        type=int,
        default=500,
        help="Límite de caracteres por ítem de evidencia",
    )
    parser.add_argument(
        "--min-items",
        type=int,
        default=5,
        help="Mínimo de ítems requeridos para solicitar perfil",
    )
    parser.add_argument(
        "--solo",
        action="append",
        default=None,
        help="Filtrar por voz específica (repetible)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Ejecutar autoverificación local sin llamadas de red",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Armar bundles y cobertura sin llamar al LLM",
    )

    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    config_dir = Path(args.config_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    run_pipeline(
        config_dir=config_dir,
        out_dir=out_dir,
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout=args.timeout,
        seed=args.seed,
        max_items=args.max_items,
        max_chars_item=args.max_chars_item,
        min_items=args.min_items,
        solo_voces=args.solo,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
