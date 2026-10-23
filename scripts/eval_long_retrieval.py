#!/usr/bin/env python3
"""Evaluador de recuperación de evidencia en contexto largo (needle-in-a-haystack).

Diseñado para medir métricas de recuperación (no de coherencia) en modelos con
contexto extendido (ej. Kateto RWKV29 en llama.cpp).
Inserta hechos unicos ("needles") en posiciones conocidas de un documento rellenado
y evalúa la tasa de recuperación de la evidencia exacta frente a abstenciones.

Desviaciones / notas de implementación:
- count_tokens soporta base_url="stub", None o callables para ejecutar run_self_check
  sin dependencias de red, empleando el stub local max(1, round(len(text)/3.2)).
- is_abstention implementa la firma (answer: str, valor: str | None = None) para
  permitir tanto la verificación aislada de frases de abstención como la regla donde
  la presencia del valor recuperado prevalece sobre cualquier abstención.
- compute_anti_hack evalúa tolerancia de 1e-9 sobre no_se_pct y longitud_media_chars.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any
import unicodedata
import urllib.error
import urllib.request

ABSTENTION_PHRASES = [
    "no se",
    "no lo se",
    "no tengo",
    "no aparece",
    "no esta",
    "no puedo",
    "no me acuerdo",
    "no encuentro",
    "no figura",
    "desconozco",
    "no hay informacion",
    "sin informacion",
    "no dispongo",
]


def load_doc(path: str | Path) -> str:
    """Lee el fuente según extensión (.md/.txt un doc; .jsonl concatena el campo 'text').
    
    Si el archivo no existe o queda vacío, imprime error a stderr y sale con exit 2.
    """
    p = Path(path)
    if not p.is_file():
        sys.stderr.write(f"Error: archivo no encontrado: {path}\n")
        sys.exit(2)
    ext = p.suffix.lower()
    try:
        if ext == ".jsonl":
            chunks: list[str] = []
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        item = json.loads(line_s)
                        text_val = item.get("text", "")
                        if text_val:
                            chunks.append(text_val)
            doc = "\n\n".join(chunks)
        else:
            doc = p.read_text(encoding="utf-8")
    except Exception as e:
        sys.stderr.write(f"Error al leer documento {path}: {e}\n")
        sys.exit(2)

    if not doc.strip():
        sys.stderr.write(f"Error: documento vacio en {path}\n")
        sys.exit(2)
    return doc


def post_json(url: str, payload: dict, timeout: float = 1800.0) -> dict:
    """Envía un POST JSON a url y retorna el dict deserializado.
    
    Ante cualquier error de red, escribe a stderr y sale con exit 3.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        sys.stderr.write(f"Error de red comunicando con {url}: {e}\n")
        sys.exit(3)


def count_tokens(base_url: Any, text: str, timeout: float = 1800.0) -> int:
    """Tokeniza el texto vía POST /tokenize y retorna len(tokens). Texto vacío -> 0."""
    if not text:
        return 0
    if callable(base_url):
        return int(base_url(text))
    if base_url in ("stub", "mock", "test", None) or (isinstance(base_url, str) and base_url.startswith("stub")):
        return max(1, round(len(text) / 3.2))
    url = f"{str(base_url).rstrip('/')}/tokenize"
    res = post_json(url, {"content": text}, timeout=timeout)
    return len(res.get("tokens", []))


def pad_to_target(
    base_url: Any,
    base_text: str,
    target_tokens: int,
    timeout: float = 1800.0,
) -> tuple[str, int]:
    """Repite base_text con separador '=== Bloque {k} ===' hasta llegar a target_tokens,
    
    y después recorta por búsqueda binaria sobre caracteres hasta que count_tokens <= target_tokens.
    Retorna (texto, tokens). Nunca devuelve menos del 90% del target si el fuente es no vacío.
    """
    if not base_text.strip():
        return "", 0

    current_tokens = count_tokens(base_url, base_text, timeout=timeout)
    doc = base_text
    k = 1
    while current_tokens < target_tokens:
        doc += f"\n\n=== Bloque {k} ===\n\n{base_text}"
        k += 1
        current_tokens = count_tokens(base_url, doc, timeout=timeout)

    if current_tokens <= target_tokens:
        return doc, current_tokens

    # Búsqueda binaria sobre caracteres
    low = 0
    high = len(doc)
    best_idx = 0
    best_tokens = 0
    while low <= high:
        mid = (low + high) // 2
        prefix = doc[:mid]
        toks = count_tokens(base_url, prefix, timeout=timeout)
        if toks <= target_tokens:
            best_idx = mid
            best_tokens = toks
            low = mid + 1
        else:
            high = mid - 1

    trimmed_doc = doc[:best_idx]
    if base_text.strip() and best_tokens < int(0.9 * target_tokens):
        sys.stderr.write(
            f"Aviso: doc recortado con {best_tokens} tokens (<90% de target {target_tokens})\n"
        )
    return trimmed_doc, best_tokens


def build_fact(i: int, doc_sha: str) -> dict:
    """Construye el hecho i determinista a partir del doc_sha."""
    valor = hashlib.sha256(f"{doc_sha}:{i}".encode("utf-8")).hexdigest()[:8].lower()
    texto = f"APUNTE {i:02d}: la clave de verificacion del bloque {i:02d} es {valor}."
    pregunta = f"¿Cual es la clave de verificacion del bloque {i:02d}? Contestá solo la clave."
    return {
        "i": i,
        "valor": valor,
        "texto": texto,
        "pregunta": pregunta,
    }


def insert_needles(
    base_url: Any,
    doc: str,
    fracs: list[float] | str,
    doc_sha: str,
    timeout: float = 1800.0,
) -> tuple[str, list[dict]]:
    """Inserta hechos en posiciones fraccionarias del documento calculadas por búsqueda binaria.
    
    Inserta en orden descendente de índice de caracter para no desplazar posiciones calculadas.
    Retorna el doc resultante con hechos insertados y la lista de hechos en orden ascendente de fracción.
    """
    if isinstance(fracs, str):
        frac_list = [float(f.strip()) for f in fracs.split(",") if f.strip()]
    else:
        frac_list = [float(f) for f in fracs]

    n_tokens = count_tokens(base_url, doc, timeout=timeout)
    facts_info: list[dict] = []

    for i, frac in enumerate(frac_list):
        fact = build_fact(i, doc_sha)
        target_needle_tokens = frac * n_tokens

        low = 0
        high = len(doc)
        best_idx = 0
        best_tokens = 0
        while low <= high:
            mid = (low + high) // 2
            toks = count_tokens(base_url, doc[:mid], timeout=timeout)
            if toks <= target_needle_tokens:
                best_idx = mid
                best_tokens = toks
                low = mid + 1
            else:
                high = mid - 1

        fact["posicion_frac"] = frac
        fact["posicion_tokens"] = best_tokens
        fact["char_idx"] = best_idx
        facts_info.append(fact)

    # Insertar en orden descendente de índice de caracter
    needles_desc = sorted(facts_info, key=lambda x: (x["char_idx"], x["i"]), reverse=True)
    out_doc = doc
    for item in needles_desc:
        idx = item["char_idx"]
        needle_str = f"\n\n{item['texto']}\n\n"
        out_doc = out_doc[:idx] + needle_str + out_doc[idx:]

    return out_doc, facts_info


def normalize(text: str) -> str:
    """Minúsculas, sin acentos (unicodedata NFKD), sin espacios, sin '-' ni '.'."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    res = []
    for c in no_accents.lower():
        if c not in (" ", "\t", "\n", "\r", "-", "."):
            res.append(c)
    return "".join(res)


def recovered(answer: str, valor: str) -> bool:
    """Verifica si el valor normalizado aparece dentro de la respuesta normalizada."""
    if not valor:
        return False
    return normalize(valor) in normalize(answer)


def is_abstention(answer: str, valor: str | None = None) -> bool:
    """Determina si la respuesta representa una abstención ('no sé', etc.).
    
    Si el hecho se recuperó (o valor está presente y recuperado), no es abstención.
    """
    if valor and recovered(answer, valor):
        return False
    if not answer.strip():
        return True
    nfkd = unicodedata.normalize("NFKD", answer)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c)).lower()
    for phrase in ABSTENTION_PHRASES:
        if phrase in no_accents:
            return True
    return False


def evaluate(
    base_url: str,
    doc_with_needles: str,
    facts: list[dict],
    n_predict: int = 96,
    temperature: float = 0.0,
    seed: int = 1337,
    timeout: float = 1800.0,
) -> list[dict]:
    """Evalúa cada hecho contra /completion en llama.cpp y retorna métricas individuales."""
    url = f"{base_url.rstrip('/')}/completion"
    results: list[dict] = []

    for fact in facts:
        pregunta = fact["pregunta"]
        prompt = f"{doc_with_needles}<|im_start|>user\n{pregunta}<|im_end|>\n<|im_start|>assistant\n"
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "seed": seed,
            "cache_prompt": True,
            "stop": ["<|im_end|>"],
        }
        sys.stderr.write(f"Evaluando hecho {fact['i']} (frac {fact['posicion_frac']})...\n")
        res = post_json(url, payload, timeout=timeout)

        content = res.get("content") or ""
        respuesta = content.strip()
        timings = res.get("timings") or {}
        prompt_ms = timings.get("prompt_ms") or 0.0
        ttfab_s = round(float(prompt_ms) / 1000.0, 4)

        rec = recovered(respuesta, fact["valor"])
        abs_ = False if rec else is_abstention(respuesta, fact["valor"])
        chars = len(respuesta)

        res_item = {
            "i": fact["i"],
            "valor": fact["valor"],
            "posicion_frac": fact["posicion_frac"],
            "posicion_tokens": fact["posicion_tokens"],
            "pregunta": pregunta,
            "respuesta": respuesta,
            "chars": chars,
            "recuperado": rec,
            "abstiene": abs_,
            "ttfab_s": ttfab_s,
        }
        results.append(res_item)

    return results


def compute_anti_hack(
    no_se_pct: float,
    longitud_media_chars: float,
    baseline: dict | None,
) -> tuple[bool | None, bool | None, bool | None]:
    """Calcula las banderas anti-hack contra el baseline."""
    if baseline is None:
        return None, None, None
    base_no_se = float(baseline["no_se_pct"])
    base_len = float(baseline["longitud_media_chars"])
    no_se_ok = bool(no_se_pct <= base_no_se + 1e-9)
    longitud_ok = bool(longitud_media_chars >= (0.8 * base_len) - 1e-9)
    anti_hack_ok = bool(no_se_ok and longitud_ok)
    return no_se_ok, longitud_ok, anti_hack_ok


def run_self_check() -> int:
    """Ejecuta self-check completo sin requerir red y retorna 0."""
    # 1. pad_to_target con fixture de ~300 chars y target 2000 tokens
    fixture = (
        "Kateto RWKV29 es una arquitectura recurrente optimizada para contexto largo. "
        "Permite evaluar coherencia y recuperacion de informacion con alta fidelidad. "
        "Este texto de prueba tiene aproximadamente trescientos caracteres para servir "
        "de fixture en el autochequeo de la funcion pad_to_target sin requerir red."
    )
    assert 280 <= len(fixture) <= 320, f"Fixture len {len(fixture)} no esta cerca de 300 chars"
    doc_pad, pad_tokens = pad_to_target("stub", fixture, 2000)
    toks_counted = count_tokens("stub", doc_pad)
    assert 1800 <= toks_counted <= 2000, f"Token count {toks_counted} fuera de [1800, 2000]"
    assert pad_tokens == toks_counted

    # 2. insert_needles con fracs 0.05,0.5,0.95
    doc_sha = hashlib.sha256(doc_pad.encode("utf-8")).hexdigest()
    fracs = [0.05, 0.5, 0.95]
    doc_needles, facts = insert_needles("stub", doc_pad, fracs, doc_sha)
    assert len(facts) == 3, f"Esperado 3 hechos, obtenido {len(facts)}"
    for f in facts:
        assert f["texto"] in doc_needles, f"Hecho {f['i']} no aparece en doc rellenado"
    assert facts[0]["posicion_tokens"] < facts[1]["posicion_tokens"] < facts[2]["posicion_tokens"], (
        "posicion_tokens no es ascendente"
    )
    for f, frac in zip(facts, fracs):
        expected_pos = frac * pad_tokens
        actual_pos = f["posicion_tokens"]
        assert 0.9 * expected_pos <= actual_pos <= 1.1 * expected_pos, (
            f"posicion {actual_pos} fuera de +-10% de {expected_pos}"
        )

    # 3. build_fact es determinista
    f_a = build_fact(3, "test_doc_sha_12345678")
    f_b = build_fact(3, "test_doc_sha_12345678")
    assert f_a["valor"] == f_b["valor"], "build_fact no es determinista"
    assert len(f_a["valor"]) == 8, f"Longitud del valor esperada 8, obtenida {len(f_a['valor'])}"
    assert all(c in "0123456789abcdef" for c in f_a["valor"]), "valor no es hex en minusculas"

    # 4. recovered
    val = f_a["valor"]
    assert recovered(f"... la clave es {val}.", val) is True
    assert recovered("No se, no me acuerdo.", val) is False
    assert recovered(f"  {val} ", val.upper()) is True

    # 5. is_abstention
    assert is_abstention("No sé") is True
    assert is_abstention("no lo sé, che") is True
    assert is_abstention("") is True
    assert is_abstention(val) is False
    assert is_abstention("La clave es " + val) is False

    # 6. Aritmética
    k = 5
    recuperados = 2
    abstenciones = 1
    respuestas_test = ["abcd", "12", "xyz", "qwert", ""]
    chars_test = [len(r.strip()) for r in respuestas_test]
    hechos_recuperados_pct = round(100.0 * recuperados / k, 2)
    no_se_pct = round(100.0 * abstenciones / k, 2)
    longitud_media_chars = round(sum(chars_test) / k, 2)
    assert hechos_recuperados_pct == 40.0, f"hechos_recuperados_pct esperada 40.0, dio {hechos_recuperados_pct}"
    assert no_se_pct == 20.0, f"no_se_pct esperada 20.0, dio {no_se_pct}"
    assert longitud_media_chars == 2.8, f"longitud_media_chars esperada 2.8, dio {longitud_media_chars}"

    # 7. Anti-hack
    no_se_ok, long_ok, ah_ok = compute_anti_hack(20.0, 40.0, None)
    assert ah_ok is None, f"anti_hack_ok esperado None, dio {ah_ok}"
    assert no_se_ok is None
    assert long_ok is None

    base = {"no_se_pct": 10.0, "longitud_media_chars": 40.0}
    no_se_ok, long_ok, ah_ok = compute_anti_hack(20.0, 40.0, base)
    assert no_se_ok is False, f"no_se_ok esperado False, dio {no_se_ok}"

    no_se_ok, long_ok, ah_ok = compute_anti_hack(10.0, 31.0, base)
    assert long_ok is False, f"longitud_ok esperado False, dio {long_ok}"

    no_se_ok, long_ok, ah_ok = compute_anti_hack(10.0, 32.0, base)
    assert ah_ok is True, f"anti_hack_ok esperado True, dio {ah_ok}"

    print("self-check OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluador de recuperación de evidencia en contexto largo (needle-in-a-haystack)"
    )
    parser.add_argument(
        "--docs",
        help="Archivo fuente (.md/.txt: un documento; .jsonl: concatena el campo 'text' de cada linea)",
    )
    parser.add_argument(
        "--out",
        default="out/eureka-long-retrieval.json",
        help="Ruta de salida JSON (default: out/eureka-long-retrieval.json)",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8199",
        help="URL base de llama.cpp (default: http://127.0.0.1:8199)",
    )
    parser.add_argument(
        "--model-label",
        default="kateto-rwkv29-Q4_K_M.gguf",
        help="Etiqueta del modelo (default: kateto-rwkv29-Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=32768,
        help="Tokens objetivo para el haystack (default: 32768)",
    )
    parser.add_argument(
        "--positions",
        default="0.05,0.25,0.50,0.75,0.95",
        help="Fracciones separadas por coma (default: '0.05,0.25,0.50,0.75,0.95')",
    )
    parser.add_argument(
        "--n-predict",
        type=int,
        default=96,
        help="Tokens a predecir por respuesta (default: 96)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperatura de sampling (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Semilla del generador (default: 1337)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Timeout en segundos (default: 1800)",
    )
    parser.add_argument(
        "--baseline",
        default="out/eureka-long-retrieval-baseline.json",
        help="Ruta a baseline JSON (default: out/eureka-long-retrieval-baseline.json)",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Escribe el baseline con los números de esta corrida",
    )
    parser.add_argument(
        "--keep-haystack",
        help="Opcional: guarda el documento rellenado con needles (auditoría)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Corre verificaciones locales sin red y sale 0",
    )

    args = parser.parse_args()

    if args.self_check:
        return run_self_check()

    if not args.docs:
        sys.stderr.write("Error: falta argumento --docs (o usar --self-check)\n")
        sys.exit(2)

    t0 = time.time()
    base_text = load_doc(args.docs)
    padded_doc, doc_tokens = pad_to_target(
        args.base_url,
        base_text,
        args.target_tokens,
        timeout=args.timeout,
    )
    doc_sha = hashlib.sha256(padded_doc.encode("utf-8")).hexdigest()

    doc_with_needles, facts = insert_needles(
        args.base_url,
        padded_doc,
        args.positions,
        doc_sha,
        timeout=args.timeout,
    )

    if args.keep_haystack:
        p_hay = Path(args.keep_haystack)
        p_hay.parent.mkdir(parents=True, exist_ok=True)
        p_hay.write_text(doc_with_needles, encoding="utf-8")
        sys.stderr.write(f"Haystack guardado en {p_hay}\n")

    evaluated_facts = evaluate(
        base_url=args.base_url,
        doc_with_needles=doc_with_needles,
        facts=facts,
        n_predict=args.n_predict,
        temperature=args.temperature,
        seed=args.seed,
        timeout=args.timeout,
    )

    k = len(evaluated_facts)
    if k == 0:
        sys.stderr.write("Error: no se evaluaron hechos\n")
        sys.exit(1)

    recuperados = sum(1 for f in evaluated_facts if f["recuperado"])
    abstenciones = sum(1 for f in evaluated_facts if f["abstiene"])
    hechos_recuperados_pct = round(100.0 * recuperados / k, 2)
    no_se_pct = round(100.0 * abstenciones / k, 2)
    longitud_media_chars = round(
        statistics.mean(f["chars"] for f in evaluated_facts), 2
    )

    baseline_data: dict[str, float] | None = None
    baseline_path = Path(args.baseline)
    if baseline_path.is_file():
        try:
            raw_b = json.loads(baseline_path.read_text(encoding="utf-8"))
            if "no_se_pct" in raw_b and "longitud_media_chars" in raw_b:
                baseline_data = {
                    "no_se_pct": float(raw_b["no_se_pct"]),
                    "longitud_media_chars": float(raw_b["longitud_media_chars"]),
                }
        except Exception as e:
            sys.stderr.write(f"Aviso: no se pudo parsear baseline en {baseline_path}: {e}\n")

    no_se_ok, longitud_ok, anti_hack_ok = compute_anti_hack(
        no_se_pct, longitud_media_chars, baseline_data
    )

    if args.write_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        base_to_write = {
            "no_se_pct": no_se_pct,
            "longitud_media_chars": longitud_media_chars,
        }
        baseline_path.write_text(
            json.dumps(base_to_write, indent=2) + "\n",
            encoding="utf-8",
        )
        sys.stderr.write(f"Baseline escrito en {baseline_path}\n")

    wall_s_total = round(time.time() - t0, 4)
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    stdout_data = {
        "script": "eval_long_retrieval.py",
        "ts": ts,
        "modelo": args.model_label,
        "base_url": args.base_url,
        "docs": args.docs,
        "doc_sha256": doc_sha,
        "target_tokens": args.target_tokens,
        "doc_tokens": doc_tokens,
        "k": k,
        "hechos": evaluated_facts,
        "hechos_recuperados_pct": hechos_recuperados_pct,
        "no_se_pct": no_se_pct,
        "longitud_media_chars": longitud_media_chars,
        "no_se_ok": no_se_ok,
        "longitud_ok": longitud_ok,
        "anti_hack_ok": anti_hack_ok,
        "baseline": baseline_data,
        "correlacion_voto_humano": None,
        "nota_correlacion": (
            "requiere >= 2 modelos medidos y pares_orpo.jsonl de kateto-rlhf; no se calcula en esta corrida"
        ),
        "n_predict": args.n_predict,
        "seed": args.seed,
        "temperature": args.temperature,
        "wall_s_total": wall_s_total,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = dict(stdout_data)
    out_data["docs_path"] = str(Path(args.docs).resolve())
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # stdout: SOLO la última línea es JSON, una sola línea, sin nada más
    sys.stdout.write(json.dumps(stdout_data, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
