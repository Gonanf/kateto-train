#!/usr/bin/env python3
"""
scripts/distinguibilidad_voces.py

Medición de distinguibilidad de voces:
Compara Condición A (SOUL solo) vs Condición B (SOUL + perfil emergente).
Un juez ciego clasifica cuál de las 4 voces escribió el texto. Azar = 0.25.
Solo stdlib de Python 3.14.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request

PROMPTS = [
    "¿Qué opinás del trabajo remoto?",
    "Contame algo que te haya salido mal esta semana.",
    "¿Vale la pena discutir con alguien que no cambia de idea?",
    "¿Qué harías con un domingo libre?",
    "¿Cómo le explicarías a alguien que se equivocó sin hacerlo sentir mal?",
    "¿Qué te da bronca de la gente?",
    "¿Qué te parece que todos digan que hay que ser productivo?",
    "Contame un plan que no salió.",
]

STOPWORDS = {
    "para", "como", "pero", "este", "esta", "estos", "estas", "porque",
    "cuando", "sobre", "entre", "también", "hasta", "desde", "donde",
    "quien", "quienes", "durante", "todos", "todas", "otro", "otra",
    "otros", "otras", "tanto", "mucho", "mucha", "muchos", "muchas",
    "poco", "poca", "pocos", "pocas", "cada", "menos", "contra",
    "hacia", "luego", "estoy", "está", "están", "tiene", "tienen",
    "hacer", "puede", "pueden", "sería", "había", "bueno", "tenía",
}

SPEAKER_MAP = {
    "jane": "jane",
    "doktor": "doktor",
    "whisperer": "whisperer",
    "conquest": "conquest",
}

HEADER_RE = re.compile(
    r'^\*\*(Jane|Doktor|Whisperer|Conquest)\*\*(?:\s*\((?:argument|objection|rebuttal|ruling)\))?:?',
    re.IGNORECASE,
)


def cargar_perfiles(perfiles_dir: Path) -> dict[str, dict]:
    perfiles = {}
    if perfiles_dir.is_dir():
        for f in perfiles_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("valido") is True:
                    voz = (data.get("voz") or f.stem).strip().lower()
                    perfiles[voz] = data
            except Exception:
                continue

    if not perfiles:
        sys.stderr.write("Error: No se encontraron perfiles con 'valido: true' en perfiles-dir.\n")
        sys.exit(2)

    return perfiles


def cargar_souls(config_dir: Path, voices: list[str]) -> dict[str, str]:
    souls = {}
    for v in voices:
        soul_path = config_dir / "voices" / v / "SOUL.md"
        if soul_path.is_file():
            souls[v] = soul_path.read_text(encoding="utf-8")
        else:
            souls[v] = f"Voz {v}."
    return souls


def primera_oracion_soul(soul_text: str, max_chars: int = 200) -> str:
    lines = [
        line.strip()
        for line in soul_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    clean = " ".join(lines)
    m = re.search(r'^(.*?[.!?])(?:\s|$)', clean)
    if m:
        sent = m.group(1).strip()
    else:
        sent = clean
    if len(sent) > max_chars:
        sent = sent[:max_chars].rstrip() + "..."
    return sent


def cargar_lexico_historico(corpus_dir: Path, voices: list[str]) -> dict[str, set[str]]:
    lexico = {v: set() for v in voices}
    if not corpus_dir.is_dir():
        return lexico

    for md_path in corpus_dir.glob("*.md"):
        try:
            lines = md_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        current_speaker = None
        current_block = []

        def flush():
            nonlocal current_speaker, current_block
            if current_speaker and current_block and current_speaker in lexico:
                block_txt = " ".join(current_block).lower()
                words = re.findall(r'[a-záéíóúüñ]{4,}', block_txt)
                content_words = {w for w in words if w not in STOPWORDS}
                lexico[current_speaker].update(content_words)
            current_speaker = None
            current_block = []

        for line in lines:
            sline = line.strip()
            if not sline:
                flush()
                continue

            match = HEADER_RE.match(sline)
            if match:
                flush()
                speaker_raw = match.group(1).lower()
                current_speaker = SPEAKER_MAP.get(speaker_raw)
                rest = sline[match.end():].strip()
                if rest:
                    current_block.append(rest)
                continue

            if (
                sline.startswith("**")
                or sline.startswith("#")
                or sline.startswith("- ")
                or sline.startswith("* ")
            ):
                flush()
                continue

            if current_speaker:
                current_block.append(sline)

        flush()

    return lexico


def construir_system_a(soul_text: str) -> str:
    return soul_text[:1200] + "\n\nRespondé en una sola intervención breve, en personaje."


def construir_system_b(soul_text: str, perfil: dict) -> str:
    base = construir_system_a(soul_text)
    bloque_lines = ["\n\nPERFIL EMERGENTE (derivado de tu historia registrada):"]
    resumen = perfil.get("resumen")
    if resumen:
        bloque_lines.append(resumen.strip())

    rasgos = perfil.get("rasgos", [])
    for r in rasgos:
        if isinstance(r, dict):
            r_txt = r.get("rasgo", "")
        else:
            r_txt = str(r)
        if r_txt.strip():
            bloque_lines.append(f"- {r_txt.strip()}")

    lexico = perfil.get("lexico", [])
    if lexico:
        if isinstance(lexico, list):
            lex_str = ", ".join(str(w) for w in lexico)
        else:
            lex_str = str(lexico)
        bloque_lines.append(f"Léxico característico: {lex_str}")

    return base + "\n".join(bloque_lines)


def _post_json(url: str, payload: dict, timeout: int = 180) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp_bytes = resp.read()
        return json.loads(resp_bytes.decode("utf-8"))


def call_llm_with_retry(
    url: str,
    payload: dict,
    timeout: int,
    max_retries: int = 2,
    retry_delay: float = 5.0,
) -> dict:
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return _post_json(url, payload, timeout=timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            if attempt < max_retries and retry_delay > 0:
                time.sleep(retry_delay)
    raise last_err


def parse_judge_response(response_str: str, voices: list[str]) -> tuple[str | None, str]:
    if not response_str:
        return None, "no_parseable"

    try:
        m = re.search(r'\{[^{}]*\}', response_str)
        if m:
            data = json.loads(m.group(0))
            if "voz" in data:
                raw_v = str(data["voz"]).strip().lower()
                for v in voices:
                    if raw_v == v.lower():
                        return v, "json_ok"
    except Exception:
        pass

    earliest_idx = None
    found_v = None
    for v in voices:
        pattern = r'\b' + re.escape(v) + r'\b'
        match = re.search(pattern, response_str, re.IGNORECASE)
        if match:
            if earliest_idx is None or match.start() < earliest_idx:
                earliest_idx = match.start()
                found_v = v

    if found_v is not None:
        return found_v, "text_match"

    return None, "no_parseable"


def calcular_metricas_forma(
    texto: str,
    voz: str,
    lexico_historico: dict[str, set[str]],
) -> dict:
    if not texto:
        return {
            "ratio_simbolos": 0.0,
            "rep8_max": 0.0,
            "centinela": False,
            "largo_palabras": 0,
            "oov": 0.0,
        }

    allowed = re.compile(r'[a-záéíóúüñ 0-9.,;:¡!¿?\'"()\-]', re.IGNORECASE)
    allowed_count = len(allowed.findall(texto))
    ratio_simbolos = (len(texto) - allowed_count) / len(texto) if len(texto) > 0 else 0.0

    words = re.findall(r'\b\w+\b', texto.lower())
    total_8grams = len(words) - 7
    if total_8grams <= 0:
        rep8_max = 0.0
    else:
        counts = {}
        for i in range(total_8grams):
            gram = tuple(words[i:i + 8])
            counts[gram] = counts.get(gram, 0) + 1
        rep8_max = max(counts.values()) / total_8grams

    centinela_tokens = ["<|", "im_end", "im_start", "<wait>", "[INST]"]
    centinela = any(tok in texto for tok in centinela_tokens)

    largo_palabras = len(words)

    cwords = [w for w in re.findall(r'[a-záéíóúüñ]{4,}', texto.lower()) if w not in STOPWORDS]
    if not cwords:
        oov = 0.0
    else:
        voice_lex = lexico_historico.get(voz, set())
        oov_count = sum(1 for w in cwords if w not in voice_lex)
        oov = oov_count / len(cwords)

    return {
        "ratio_simbolos": ratio_simbolos,
        "rep8_max": rep8_max,
        "centinela": centinela,
        "largo_palabras": largo_palabras,
        "oov": oov,
    }


def calcular_p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = 0.95 * (len(s) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] * (c - k) + s[c] * (k - f))


def ejecutar_evaluacion(
    perfiles_dir: Path,
    config_dir: Path,
    corpus_dir: Path,
    base_url: str,
    model: str,
    timeout: int,
    judge_base_url: str,
    judge_model: str,
    judge_timeout: int,
    max_prompts: int,
    seeds: list[int],
    resume: bool,
    out_dir: Path,
    gen_stub=None,
    judge_stub=None,
    retry_delay: float = 5.0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    perfiles = cargar_perfiles(perfiles_dir)
    voices = sorted(perfiles.keys())

    souls = cargar_souls(config_dir, voices)
    soul_descs = {v: primera_oracion_soul(souls[v]) for v in voices}
    lexico_historico = cargar_lexico_historico(corpus_dir, voices)

    prompts = PROMPTS[:max_prompts]
    condiciones = ["A", "B"]

    gen_file = out_dir / "generaciones.jsonl"
    juicios_file = out_dir / "juicios.jsonl"

    existing_gen = {}
    if resume and gen_file.is_file():
        for line in gen_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                k = f"{item['voz']}|{item['cond']}|{item['prompt_idx']}|{item['seed']}"
                existing_gen[k] = item
            except Exception:
                continue

    generaciones = []
    with open(gen_file, "a" if resume else "w", encoding="utf-8") as gf:
        for voz in voices:
            for cond in condiciones:
                sys_prompt = (
                    construir_system_a(souls[voz])
                    if cond == "A"
                    else construir_system_b(souls[voz], perfiles[voz])
                )
                for p_idx, prompt_text in enumerate(prompts):
                    for seed in seeds:
                        key = f"{voz}|{cond}|{p_idx}|{seed}"
                        if resume and key in existing_gen:
                            generaciones.append(existing_gen[key])
                            continue

                        record = {
                            "voz": voz,
                            "cond": cond,
                            "prompt_idx": p_idx,
                            "seed": seed,
                            "prompt": prompt_text,
                        }

                        if gen_stub:
                            try:
                                texto = gen_stub(voz, cond, prompt_text, seed)
                                record["texto"] = texto
                            except Exception as e:
                                record["error"] = str(e)
                        else:
                            payload = {
                                "model": model,
                                "messages": [
                                    {"role": "system", "content": sys_prompt},
                                    {"role": "user", "content": prompt_text},
                                ],
                                "temperature": 0.9,
                                "max_tokens": 70,
                                "seed": seed,
                            }
                            url = f"{base_url.rstrip('/')}/chat/completions"
                            try:
                                resp = call_llm_with_retry(
                                    url,
                                    payload,
                                    timeout=timeout,
                                    retry_delay=retry_delay,
                                )
                                record["texto"] = resp["choices"][0]["message"]["content"].strip()
                            except Exception as e:
                                record["error"] = str(e)

                        gf.write(json.dumps(record, ensure_ascii=False) + "\n")
                        gf.flush()
                        generaciones.append(record)

    existing_juicios = {}
    if resume and juicios_file.is_file():
        for line in juicios_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                k = f"{item['voz']}|{item['cond']}|{item['prompt_idx']}|{item['seed']}"
                existing_juicios[k] = item
            except Exception:
                continue

    juicios = []
    with open(juicios_file, "a" if resume else "w", encoding="utf-8") as jf:
        for gen_item in generaciones:
            v_true = gen_item["voz"]
            cond = gen_item["cond"]
            p_idx = gen_item["prompt_idx"]
            seed = gen_item["seed"]
            key = f"{v_true}|{cond}|{p_idx}|{seed}"

            if resume and key in existing_juicios:
                juicios.append(existing_juicios[key])
                continue

            texto = gen_item.get("texto")
            if gen_item.get("error") or texto is None:
                j_record = {
                    "voz": v_true,
                    "cond": cond,
                    "prompt_idx": p_idx,
                    "seed": seed,
                    "error": gen_item.get("error", "sin_texto"),
                    "pred": None,
                    "acierto": False,
                    "motivo": "generacion_fallida",
                }
                jf.write(json.dumps(j_record, ensure_ascii=False) + "\n")
                jf.flush()
                juicios.append(j_record)
                continue

            hash_input = f"{texto}{seed}".encode("utf-8")
            h_digest = hashlib.sha256(hash_input).digest()
            rng = random.Random(h_digest)
            shuffled_options = list(voices)
            rng.shuffle(shuffled_options)

            if judge_stub:
                try:
                    res_stub = judge_stub(texto, shuffled_options)
                    if isinstance(res_stub, dict):
                        resp_str = json.dumps(res_stub)
                    else:
                        resp_str = str(res_stub)
                    pred, motivo = parse_judge_response(resp_str, voices)
                    acierto = (pred.lower() == v_true.lower()) if pred else False
                    j_record = {
                        "voz": v_true,
                        "cond": cond,
                        "prompt_idx": p_idx,
                        "seed": seed,
                        "pred": pred,
                        "acierto": acierto,
                        "motivo": motivo,
                    }
                except Exception as e:
                    j_record = {
                        "voz": v_true,
                        "cond": cond,
                        "prompt_idx": p_idx,
                        "seed": seed,
                        "error": str(e),
                        "pred": None,
                        "acierto": False,
                        "motivo": "error_juez_stub",
                    }
            else:
                desc_lines = [f"- {v}: {soul_descs[v]}" for v in shuffled_options]
                system_judge = (
                    'Sos un oyente. Te doy un texto y cuatro voces con una descripción de una '
                    'línea cada una; decí cuál lo escribió. Respondé sólo {"voz": "<id>"}'
                )
                user_judge = (
                    f"Texto:\n\"\"\"{texto}\"\"\"\n\n"
                    f"Voces:\n" + "\n".join(desc_lines)
                )
                payload = {
                    "model": judge_model,
                    "messages": [
                        {"role": "system", "content": system_judge},
                        {"role": "user", "content": user_judge},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 50,
                    "seed": seed,
                }
                url = f"{judge_base_url.rstrip('/')}/chat/completions"
                try:
                    resp = call_llm_with_retry(
                        url,
                        payload,
                        timeout=judge_timeout,
                        retry_delay=retry_delay,
                    )
                    raw_content = resp["choices"][0]["message"]["content"].strip()
                    pred, motivo = parse_judge_response(raw_content, voices)
                    acierto = (pred.lower() == v_true.lower()) if pred else False
                    j_record = {
                        "voz": v_true,
                        "cond": cond,
                        "prompt_idx": p_idx,
                        "seed": seed,
                        "pred": pred,
                        "acierto": acierto,
                        "motivo": motivo,
                        "raw_judge": raw_content,
                    }
                except Exception as e:
                    j_record = {
                        "voz": v_true,
                        "cond": cond,
                        "prompt_idx": p_idx,
                        "seed": seed,
                        "error": str(e),
                        "pred": None,
                        "acierto": False,
                        "motivo": "error_juez_http",
                    }

            jf.write(json.dumps(j_record, ensure_ascii=False) + "\n")
            jf.flush()
            juicios.append(j_record)

    errores_count = sum(1 for g in generaciones if g.get("error")) + sum(
        1 for j in juicios if j.get("error")
    )

    por_condicion = {}
    metricas_forma_por_cond = {}

    for cond in condiciones:
        j_cond = [j for j in juicios if j["cond"] == cond and not j.get("error")]
        aciertos = [1.0 if j.get("acierto") else 0.0 for j in j_cond]
        acierto_media = sum(aciertos) / len(aciertos) if aciertos else 0.0

        g_cond = [g for g in generaciones if g["cond"] == cond and g.get("texto")]
        forma_list = [
            calcular_metricas_forma(g["texto"], g["voz"], lexico_historico)
            for g in g_cond
        ]

        r_simbolos = [f["ratio_simbolos"] for f in forma_list]
        rep8 = [f["rep8_max"] for f in forma_list]
        largos = [float(f["largo_palabras"]) for f in forma_list]
        oovs = [f["oov"] for f in forma_list]
        centinelas_total = sum(1 for f in forma_list if f["centinela"])

        por_condicion[cond] = {
            "acierto_media": round(acierto_media, 4),
            "ratio_simbolos_media": round(sum(r_simbolos) / len(r_simbolos), 4) if r_simbolos else 0.0,
            "ratio_simbolos_p95": round(calcular_p95(r_simbolos), 4),
            "rep8_max_media": round(sum(rep8) / len(rep8), 4) if rep8 else 0.0,
            "rep8_max_p95": round(calcular_p95(rep8), 4),
            "largo_palabras_media": round(sum(largos) / len(largos), 2) if largos else 0.0,
            "largo_palabras_p95": round(calcular_p95(largos), 2),
            "oov_media": round(sum(oovs) / len(oovs), 4) if oovs else 0.0,
            "oov_p95": round(calcular_p95(oovs), 4),
            "centinelas_total": centinelas_total,
        }
        metricas_forma_por_cond[cond] = forma_list

    media_A = por_condicion["A"]["acierto_media"]
    media_B = por_condicion["B"]["acierto_media"]
    delta = round(media_B - media_A, 4)

    delta_por_seed = {}
    for s in seeds:
        j_A_s = [j for j in juicios if j["cond"] == "A" and j["seed"] == s and not j.get("error")]
        j_B_s = [j for j in juicios if j["cond"] == "B" and j["seed"] == s and not j.get("error")]
        acc_A_s = sum(1.0 if j.get("acierto") else 0.0 for j in j_A_s) / len(j_A_s) if j_A_s else 0.0
        acc_B_s = sum(1.0 if j.get("acierto") else 0.0 for j in j_B_s) / len(j_B_s) if j_B_s else 0.0
        delta_por_seed[str(s)] = round(acc_B_s - acc_A_s, 4)

    seeds_pos_count = sum(1 for s in seeds if delta_por_seed.get(str(s), 0.0) > 0)
    seeds_pass = seeds_pos_count >= min(2, len(seeds))

    diff_ratio = round(
        por_condicion["B"]["ratio_simbolos_media"] - por_condicion["A"]["ratio_simbolos_media"],
        4,
    )
    guard_ratio_pass = diff_ratio <= 0.02

    diff_rep8 = round(
        por_condicion["B"]["rep8_max_media"] - por_condicion["A"]["rep8_max_media"],
        4,
    )
    guard_rep8_pass = diff_rep8 <= 0.05

    centinelas_B = por_condicion["B"]["centinelas_total"]
    guard_centinelas_pass = (centinelas_B == 0)

    largo_A = por_condicion["A"]["largo_palabras_media"]
    largo_B = por_condicion["B"]["largo_palabras_media"]
    if largo_A == 0:
        guard_largo_pass = (largo_B == 0)
        largo_ratio = 0.0 if largo_B == 0 else float("inf")
    else:
        largo_ratio = round(largo_B / largo_A, 4)
        guard_largo_pass = largo_B <= 1.5 * largo_A

    guardas = {
        "ratio_simbolos_diff": diff_ratio,
        "ratio_simbolos_pass": guard_ratio_pass,
        "rep8_max_diff": diff_rep8,
        "rep8_max_pass": guard_rep8_pass,
        "centinelas_B": centinelas_B,
        "centinelas_pass": guard_centinelas_pass,
        "largo_palabras_ratio": largo_ratio,
        "largo_palabras_pass": guard_largo_pass,
        "seeds_pos_count": seeds_pos_count,
        "seeds_pass": seeds_pass,
    }

    motivos = []
    total_validos_A = len([j for j in juicios if j["cond"] == "A" and not j.get("error")])
    total_validos_B = len([j for j in juicios if j["cond"] == "B" and not j.get("error")])

    if total_validos_A == 0 or total_validos_B == 0:
        veredicto = "SIN_DATOS"
        motivos.append("No hay generaciones válidas para calcular distinguibilidad")
    elif delta >= 0.15:
        fallas = []
        if not seeds_pass:
            fallas.append(f"Solo {seeds_pos_count} seeds tuvieron delta > 0 (requiere al menos 2)")
        if not guard_ratio_pass:
            fallas.append(f"Guarda ratio_simbolos violada: diff {diff_ratio:+.4f} > 0.02")
        if not guard_rep8_pass:
            fallas.append(f"Guarda rep8_max violada: diff {diff_rep8:+.4f} > 0.05")
        if not guard_centinelas_pass:
            fallas.append(f"Guarda centinelas violada: {centinelas_B} centinelas encontrados en condición B")
        if not guard_largo_pass:
            fallas.append(f"Guarda largo_palabras violada: largo_B ({largo_B:.1f}) > 1.5 * largo_A ({largo_A:.1f})")

        if fallas:
            veredicto = "HACKEADO"
            motivos.extend(fallas)
        else:
            veredicto = "PASA"
    else:
        veredicto = "NO CUMPLE"
        motivos.append(f"delta ({delta:.4f}) < 0.15")

    if media_A >= 0.85:
        motivos.append(f"Efecto techo detectado: acierto_A es {media_A:.4f} (>= 0.85), el delta no puede crecer mucho")

    acierto_por_voz = {"A": {}, "B": {}}
    matriz_confusion = {"A": {}, "B": {}}

    for cond in condiciones:
        matriz_confusion[cond] = {
            v: {target_v: 0 for target_v in voices + ["no_parseable"]}
            for v in voices
        }
        for v in voices:
            v_juicios = [
                j for j in juicios
                if j["cond"] == cond and j["voz"] == v and not j.get("error")
            ]
            if v_juicios:
                acc_v = sum(1.0 if j.get("acierto") else 0.0 for j in v_juicios) / len(v_juicios)
                acierto_por_voz[cond][v] = round(acc_v, 4)
            else:
                acierto_por_voz[cond][v] = 0.0

            for j in v_juicios:
                pred_v = j.get("pred")
                if pred_v in voices:
                    matriz_confusion[cond][v][pred_v] += 1
                else:
                    matriz_confusion[cond][v]["no_parseable"] += 1

    metricas = {
        "generado_en": datetime.datetime.now().isoformat(),
        "modelo": model,
        "juez": judge_model,
        "seeds": seeds,
        "prompts": len(prompts),
        "voces": voices,
        "por_condicion": por_condicion,
        "delta": delta,
        "delta_por_seed": delta_por_seed,
        "guardas": guardas,
        "veredicto": veredicto,
        "motivos": motivos,
        "matriz_confusion": matriz_confusion,
        "acierto_por_voz": acierto_por_voz,
        "errores": errores_count,
    }

    (out_dir / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report_lines = [
        f"# Veredicto: {veredicto}\n",
        f"**Modelo generación**: `{model}` | **Juez ciego**: `{judge_model}` | **Generado**: {metricas['generado_en']}\n",
        f"**Delta global (B - A)**: **{delta:+.4f}** (Condición A = {media_A:.4f}, Condición B = {media_B:.4f})\n",
    ]

    if motivos:
        report_lines.append("### Motivos y Observaciones\n")
        for m in motivos:
            report_lines.append(f"- {m}")
        report_lines.append("")

    report_lines.append("## Acierto por Voz\n")
    report_lines.append("| Voz | Acierto A (SOUL solo) | Acierto B (SOUL + Perfil) | Delta (B - A) |")
    report_lines.append("| :--- | :---: | :---: | :---: |")
    for v in voices:
        acc_a = acierto_por_voz["A"].get(v, 0.0)
        acc_b = acierto_por_voz["B"].get(v, 0.0)
        report_lines.append(f"| **{v}** | {acc_a:.4f} | {acc_b:.4f} | {acc_b - acc_a:+.4f} |")
    report_lines.append("")

    report_lines.append("## Guardas Anti-Hack\n")
    report_lines.append("| Guarda | Métrica A | Métrica B | Valor / Diferencia | Umbral | Estado |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    report_lines.append(
        f"| Ratio símbolos | {por_condicion['A']['ratio_simbolos_media']:.4f} | "
        f"{por_condicion['B']['ratio_simbolos_media']:.4f} | diff: {diff_ratio:+.4f} | "
        f"<= +0.02 | {'CUMPLE' if guard_ratio_pass else 'FALLA'} |"
    )
    report_lines.append(
        f"| Repetición 8-gramas (rep8) | {por_condicion['A']['rep8_max_media']:.4f} | "
        f"{por_condicion['B']['rep8_max_media']:.4f} | diff: {diff_rep8:+.4f} | "
        f"<= +0.05 | {'CUMPLE' if guard_rep8_pass else 'FALLA'} |"
    )
    report_lines.append(
        f"| Centinelas en B | {por_condicion['A']['centinelas_total']} | "
        f"{por_condicion['B']['centinelas_total']} | total B: {centinelas_B} | "
        f"== 0 | {'CUMPLE' if guard_centinelas_pass else 'FALLA'} |"
    )
    report_lines.append(
        f"| Largo palabras (media) | {largo_A:.1f} | {largo_B:.1f} | "
        f"ratio: {largo_ratio:.2f}x | <= 1.5x | {'CUMPLE' if guard_largo_pass else 'FALLA'} |"
    )
    report_lines.append(
        f"| Consistencia seeds (delta > 0) | - | - | "
        f"{seeds_pos_count} de {len(seeds)} | >= 2 | {'CUMPLE' if seeds_pass else 'FALLA'} |"
    )
    report_lines.append("")

    report_lines.append("## Métricas de Forma Detalladas (Media / p95)\n")
    report_lines.append("| Condición | Ratio Símbolos | Rep8 Max | Largo Palabras | OOV Léxico Histórico |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    report_lines.append(
        f"| **A (SOUL solo)** | {por_condicion['A']['ratio_simbolos_media']:.4f} / {por_condicion['A']['ratio_simbolos_p95']:.4f} | "
        f"{por_condicion['A']['rep8_max_media']:.4f} / {por_condicion['A']['rep8_max_p95']:.4f} | "
        f"{por_condicion['A']['largo_palabras_media']:.1f} / {por_condicion['A']['largo_palabras_p95']:.1f} | "
        f"{por_condicion['A']['oov_media']:.4f} / {por_condicion['A']['oov_p95']:.4f} |"
    )
    report_lines.append(
        f"| **B (SOUL + Perfil)** | {por_condicion['B']['ratio_simbolos_media']:.4f} / {por_condicion['B']['ratio_simbolos_p95']:.4f} | "
        f"{por_condicion['B']['rep8_max_media']:.4f} / {por_condicion['B']['rep8_max_p95']:.4f} | "
        f"{por_condicion['B']['largo_palabras_media']:.1f} / {por_condicion['B']['largo_palabras_p95']:.1f} | "
        f"{por_condicion['B']['oov_media']:.4f} / {por_condicion['B']['oov_p95']:.4f} |"
    )
    report_lines.append("")

    report_lines.append("## Matriz de Confusión\n")
    for cond in condiciones:
        report_lines.append(f"### Condición {cond}\n")
        header = "| Real \\ Pred | " + " | ".join(voices) + " | No Parseable |"
        sep = "| :--- | " + " | ".join([":---:"] * (len(voices) + 1)) + " |"
        report_lines.append(header)
        report_lines.append(sep)
        for v in voices:
            row = [f"**{v}**"]
            for target_v in voices:
                row.append(str(matriz_confusion[cond][v].get(target_v, 0)))
            row.append(str(matriz_confusion[cond][v].get("no_parseable", 0)))
            report_lines.append("| " + " | ".join(row) + " |")
        report_lines.append("")

    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return metricas


def run_self_check() -> None:
    tmpdir_obj = tempfile.TemporaryDirectory(prefix="distinguibilidad_selfcheck_")
    tmpdir = Path(tmpdir_obj.name)

    try:
        config_dir = tmpdir / "config"
        perfiles_dir = tmpdir / "perfiles"
        corpus_dir = tmpdir / "corpus"

        voices = ["jane", "doktor", "whisperer", "conquest"]
        for v in voices:
            v_dir = config_dir / "voices" / v
            v_dir.mkdir(parents=True, exist_ok=True)
            (v_dir / "SOUL.md").write_text(
                f"{v.capitalize()} es una voz analítica orientada a debatir con rigor metodológico.",
                encoding="utf-8",
            )

        perfiles_dir.mkdir(parents=True, exist_ok=True)
        for v in voices:
            p_data = {
                "voz": v,
                "valido": True,
                "rasgos": [{"rasgo": f"rasgo característico de {v}", "ev": ["evidencia"]}],
                "resumen": f"Resumen del perfil emergente de {v}.",
                "lexico": ["analizar", "cautela", "proceder", "cuidado", "método"],
                "evita": ["descuido"],
            }
            (perfiles_dir / f"{v}.json").write_text(json.dumps(p_data), encoding="utf-8")

        corpus_dir.mkdir(parents=True, exist_ok=True)
        (corpus_dir / "debate-test.md").write_text(
            "**Jane** (argument):\n"
            "analizar cautela proceder cuidado datos método lógica proceso prueba teoría\n\n"
            "**Doktor** (objection):\n"
            "analizar rigor empírico protocolo sistemático precisión fórmula estándar axioma\n\n"
            "**Whisperer** (rebuttal):\n"
            "sombra silencio secreto misterio penumbra presagio intuir discreto señal\n\n"
            "**Conquest** (ruling):\n"
            "victoria dominio avance conquista control mando triunfo imperio poder\n",
            encoding="utf-8",
        )

        def _gen_stub_normal(voz, cond, prompt, seed):
            p_idx = PROMPTS.index(prompt) if prompt in PROMPTS else 0
            return (
                f"Voz {voz} condicion {cond} semilla {seed} prompt {p_idx}. "
                "Respuesta normal para analizar datos con cautela y proceder con cuidado."
            )

        def _juez_stub_canned_075_025(texto, opciones):
            m = re.search(r'Voz (\w+) condicion (\w+) semilla (\d+) prompt (\d+)', texto)
            if not m:
                return {"voz": opciones[0]}
            true_v, cond = m.group(1), m.group(2)
            if cond == "A":
                if true_v == "jane":
                    return {"voz": "jane"}
                return {"voz": "jane"}
            else:
                if true_v != "conquest":
                    return {"voz": true_v}
                return {"voz": "jane"}

        out1 = tmpdir / "out1"
        res1 = ejecutar_evaluacion(
            perfiles_dir=perfiles_dir,
            config_dir=config_dir,
            corpus_dir=corpus_dir,
            base_url="http://mock",
            model="mock",
            timeout=10,
            judge_base_url="http://mock",
            judge_model="mock",
            judge_timeout=10,
            max_prompts=2,
            seeds=[1, 2, 3],
            resume=False,
            out_dir=out1,
            gen_stub=_gen_stub_normal,
            judge_stub=_juez_stub_canned_075_025,
            retry_delay=0.0,
        )
        assert res1["veredicto"] == "PASA", f"Caso 1 esperado PASA, obtenido {res1['veredicto']}: {res1.get('motivos')}"
        assert abs(res1["por_condicion"]["A"]["acierto_media"] - 0.25) < 1e-4
        assert abs(res1["por_condicion"]["B"]["acierto_media"] - 0.75) < 1e-4

        def _gen_stub_hacked(voz, cond, prompt, seed):
            p_idx = PROMPTS.index(prompt) if prompt in PROMPTS else 0
            if cond == "B":
                return (
                    f"Voz {voz} condicion {cond} semilla {seed} prompt {p_idx}. "
                    "palabra palabra palabra palabra palabra palabra palabra palabra palabra palabra palabra palabra <|im_end|>"
                )
            return (
                f"Voz {voz} condicion {cond} semilla {seed} prompt {p_idx}. "
                "Respuesta normal para analizar datos con cautela y proceder con cuidado."
            )

        out2 = tmpdir / "out2"
        res2 = ejecutar_evaluacion(
            perfiles_dir=perfiles_dir,
            config_dir=config_dir,
            corpus_dir=corpus_dir,
            base_url="http://mock",
            model="mock",
            timeout=10,
            judge_base_url="http://mock",
            judge_model="mock",
            judge_timeout=10,
            max_prompts=2,
            seeds=[1, 2, 3],
            resume=False,
            out_dir=out2,
            gen_stub=_gen_stub_hacked,
            judge_stub=_juez_stub_canned_075_025,
            retry_delay=0.0,
        )
        assert res2["veredicto"] == "HACKEADO", f"Caso 2 esperado HACKEADO, obtenido {res2['veredicto']}"
        motivos_str = " ".join(res2.get("motivos", [])).lower()
        assert "centinela" in motivos_str, f"Caso 2 debe reportar centinelas violados, obtenido: {motivos_str}"
        assert "rep8" in motivos_str, f"Caso 2 debe reportar rep8 violado, obtenido: {motivos_str}"

        def _juez_stub_nocumple(texto, opciones):
            m = re.search(r'Voz (\w+) condicion (\w+) semilla (\d+) prompt (\d+)', texto)
            if not m:
                return {"voz": opciones[0]}
            true_v, cond, p_idx = m.group(1), m.group(2), int(m.group(4))
            v_idx = voices.index(true_v)
            item_idx = p_idx * 4 + v_idx
            if cond == "A":
                if item_idx in [0, 1, 2, 3, 4]:
                    return {"voz": true_v}
                wrong = [v for v in voices if v != true_v][0]
                return {"voz": wrong}
            else:
                if item_idx in [0, 1, 2, 3, 4, 5]:
                    return {"voz": true_v}
                wrong = [v for v in voices if v != true_v][0]
                return {"voz": wrong}

        out3 = tmpdir / "out3"
        res3 = ejecutar_evaluacion(
            perfiles_dir=perfiles_dir,
            config_dir=config_dir,
            corpus_dir=corpus_dir,
            base_url="http://mock",
            model="mock",
            timeout=10,
            judge_base_url="http://mock",
            judge_model="mock",
            judge_timeout=10,
            max_prompts=5,
            seeds=[1],
            resume=False,
            out_dir=out3,
            gen_stub=_gen_stub_normal,
            judge_stub=_juez_stub_nocumple,
            retry_delay=0.0,
        )
        assert res3["veredicto"] == "NO CUMPLE", f"Caso 3 esperado NO CUMPLE, obtenido {res3['veredicto']}"
        assert abs(res3["por_condicion"]["A"]["acierto_media"] - 0.25) < 1e-4
        assert abs(res3["por_condicion"]["B"]["acierto_media"] - 0.30) < 1e-4

        out4 = tmpdir / "out4"
        call_counter = [0]

        def _gen_stub_resume(voz, cond, prompt, seed):
            call_counter[0] += 1
            p_idx = PROMPTS.index(prompt) if prompt in PROMPTS else 0
            return (
                f"Voz {voz} condicion {cond} semilla {seed} prompt {p_idx}. "
                "Respuesta normal para analizar datos con cautela y proceder con cuidado."
            )

        ejecutar_evaluacion(
            perfiles_dir=perfiles_dir,
            config_dir=config_dir,
            corpus_dir=corpus_dir,
            base_url="http://mock",
            model="mock",
            timeout=10,
            judge_base_url="http://mock",
            judge_model="mock",
            judge_timeout=10,
            max_prompts=2,
            seeds=[1],
            resume=False,
            out_dir=out4,
            gen_stub=_gen_stub_resume,
            judge_stub=_juez_stub_canned_075_025,
            retry_delay=0.0,
        )
        assert call_counter[0] > 0, "Caso 4: Debió haber llamadas en la primera pasada"
        call_counter[0] = 0

        ejecutar_evaluacion(
            perfiles_dir=perfiles_dir,
            config_dir=config_dir,
            corpus_dir=corpus_dir,
            base_url="http://mock",
            model="mock",
            timeout=10,
            judge_base_url="http://mock",
            judge_model="mock",
            judge_timeout=10,
            max_prompts=2,
            seeds=[1],
            resume=True,
            out_dir=out4,
            gen_stub=_gen_stub_resume,
            judge_stub=_juez_stub_canned_075_025,
            retry_delay=0.0,
        )
        assert call_counter[0] == 0, f"Caso 4: Esperado 0 llamadas con --resume, obtenido {call_counter[0]}"

        out5 = tmpdir / "out5"

        def _gen_stub_http_err(voz, cond, prompt, seed):
            p_idx = PROMPTS.index(prompt) if prompt in PROMPTS else 0
            if voz == "jane" and cond == "A" and p_idx == 0 and seed == 1:
                raise urllib.error.HTTPError("http://stub", 500, "Internal Server Error", None, None)
            return (
                f"Voz {voz} condicion {cond} semilla {seed} prompt {p_idx}. "
                "Respuesta normal para analizar datos con cautela y proceder con cuidado."
            )

        res5 = ejecutar_evaluacion(
            perfiles_dir=perfiles_dir,
            config_dir=config_dir,
            corpus_dir=corpus_dir,
            base_url="http://mock",
            model="mock",
            timeout=10,
            judge_base_url="http://mock",
            judge_model="mock",
            judge_timeout=10,
            max_prompts=2,
            seeds=[1],
            resume=False,
            out_dir=out5,
            gen_stub=_gen_stub_http_err,
            judge_stub=_juez_stub_canned_075_025,
            retry_delay=0.0,
        )
        assert res5["errores"] >= 1, f"Caso 5: Esperado al menos 1 error, obtenido {res5.get('errores')}"

        print("SELF-CHECK OK")
    finally:
        tmpdir_obj.cleanup()


def main():
    parser = argparse.ArgumentParser(
        description="Medición de distinguibilidad de voces (Condición A vs Condición B)"
    )
    parser.add_argument(
        "--perfiles-dir",
        default="/run/media/chaos/terciario/proyectos/kateto-train/out/reflexion/perfiles",
        help="Directorio con <voz>.json de perfiles emergentes",
    )
    parser.add_argument(
        "--config-dir",
        default=os.path.expanduser("~/.config/kateto"),
        help="Directorio con voices/<voz>/SOUL.md",
    )
    parser.add_argument(
        "--corpus-dir",
        default=os.path.expanduser("~/.config/kateto/bate_debate/registry"),
        help="Directorio con debates históricos .md",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8092/v1",
        help="URL base del runtime local",
    )
    parser.add_argument(
        "--model",
        default="local",
        help="Nombre del modelo de generación",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout en segundos para generación",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="URL base para el juez (default: igual a --base-url)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Modelo para el juez (default: igual a --model)",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=120,
        help="Timeout en segundos para el juez",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=6,
        help="Número de prompts neutros a utilizar",
    )
    parser.add_argument(
        "--seeds",
        default="1,2,3",
        help="Lista de seeds separadas por comas",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Saltear generaciones y juicios ya presentes",
    )
    parser.add_argument(
        "--out",
        default="/run/media/chaos/terciario/proyectos/kateto-train/out/distinguibilidad",
        help="Directorio de salida para métricas y reportes",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Ejecutar suite interna de pruebas sin red ni disco del usuario",
    )

    args = parser.parse_args()

    if args.self_check:
        try:
            run_self_check()
            sys.exit(0)
        except AssertionError as e:
            sys.stderr.write(f"AssertionError en --self-check: {e}\n")
            sys.exit(1)
        except Exception as e:
            sys.stderr.write(f"Error inesperado en --self-check: {e}\n")
            sys.exit(1)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    judge_base_url = args.judge_base_url or args.base_url
    judge_model = args.judge_model or args.model

    res = ejecutar_evaluacion(
        perfiles_dir=Path(args.perfiles_dir),
        config_dir=Path(args.config_dir),
        corpus_dir=Path(args.corpus_dir),
        base_url=args.base_url,
        model=args.model,
        timeout=args.timeout,
        judge_base_url=judge_base_url,
        judge_model=judge_model,
        judge_timeout=args.judge_timeout,
        max_prompts=args.max_prompts,
        seeds=seeds,
        resume=args.resume,
        out_dir=Path(args.out),
    )

    print(f"Veredicto: {res['veredicto']} (delta = {res['delta']:+.4f})")
    print(f"Métricas guardadas en: {Path(args.out) / 'metricas.json'}")
    print(f"Reporte guardado en: {Path(args.out) / 'report.md'}")


if __name__ == "__main__":
    main()
