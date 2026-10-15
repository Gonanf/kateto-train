"""
Verificación de saneamiento del dataset de SFT para RWKV-7 Kateto.
Falla si:
1. Reaparece alguna clave 'question' o 'answer' en los datasets de salida.
2. Vuelve a entrar cualquier línea o par (query, response) de youtube_asr_dataset.jsonl.
3. Se detectan meta-prompts en inglés o artefactos de moderación/pipeline.
4. Falta el esquema único ('query', 'response').
"""

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

OUTPUT_FILES = [
    DATA_DIR / "rwkv_kateto_base_train.jsonl",
    DATA_DIR / "rwkv_kateto_base_train.fmtA.jsonl",
    DATA_DIR / "rwkv_kateto_base_eval.jsonl",
    DATA_DIR / "rwkv_kateto_seco.jsonl",
    DATA_DIR / "rwkv_kateto_seco.fmtA.jsonl",
    DATA_DIR / "rwkv_kateto_streamer.jsonl",
    DATA_DIR / "rwkv_kateto_streamer.fmtA.jsonl",
    DATA_DIR / "kateto_v2_combined_train.jsonl",
    DATA_DIR / "kateto_v2_combined_eval.jsonl",
]

ASR_FILE = DATA_DIR / "youtube_asr_dataset.jsonl"
QUARANTINE_FILE = DATA_DIR / "quarantine.jsonl"

def load_asr_signatures():
    """Carga firmas exactas de pares y respuestas del ASR crudo."""
    asr_pairs = set()
    asr_responses = set()
    if not ASR_FILE.exists():
        return asr_pairs, asr_responses

    with open(ASR_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            q = (item.get("question") or item.get("query") or "").strip()
            r = (item.get("answer") or item.get("response") or "").strip()
            if r.endswith("<|endoftext|>"):
                r = r[:-len("<|endoftext|>")].strip()
            if q and r:
                asr_pairs.add((q.lower(), r.lower()))
            # Las respuestas ASR son las transcripciones con artefactos fonéticos
            if len(r) > 20 and not r.startswith("<tool_call>"):
                asr_responses.add(r.lower())
    return asr_pairs, asr_responses


def test_output_files_exist():
    """Verifica que todos los archivos saneados existan y no estén vacíos."""
    for path in OUTPUT_FILES:
        assert path.exists(), f"Archivo de salida faltante: {path}"
        assert path.stat().st_size > 0, f"Archivo vacío: {path}"


def test_unique_schema_no_legacy_keys():
    """Verifica esquema único: 'query' y 'response' presentes, 'question' y 'answer' PROHIBIDAS."""
    for path in OUTPUT_FILES:
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                assert "question" not in item, f"Clave 'question' prohibida encontrada en {path.name}:{idx}"
                assert "answer" not in item, f"Clave 'answer' prohibida encontrada en {path.name}:{idx}"
                assert "query" in item, f"Clave 'query' obligatoria ausente en {path.name}:{idx}"
                assert "response" in item, f"Clave 'response' obligatoria ausente en {path.name}:{idx}"
                assert isinstance(item["query"], str) and len(item["query"].strip()) > 0, f"Query vacía en {path.name}:{idx}"
                assert isinstance(item["response"], str) and len(item["response"].strip()) > 0, f"Response vacía en {path.name}:{idx}"


def test_no_asr_leakage():
    """Comprobación real de que NINGUNA línea, par o respuesta del ASR está en las salidas."""
    asr_pairs, asr_responses = load_asr_signatures()
    assert len(asr_pairs) > 0, "No se encontraron firmas ASR para comparar."

    for path in OUTPUT_FILES:
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                q = item["query"].strip().lower()
                r = item["response"].strip().lower()
                lbl = item.get("source_label", "")

                assert "youtube-asr" not in lbl, f"Etiqueta de ASR encontrada en {path.name}:{idx}"
                assert (q, r) not in asr_pairs, f"Fuga de par ASR exacto en {path.name}:{idx}"
                if len(r) > 20 and not r.startswith("<tool_call>") and not r.startswith("```"):
                    assert r not in asr_responses, f"Transcripción ASR encontrada en {path.name}:{idx} (response: '{r[:60]}...')"


def test_no_pipeline_artifacts_or_english_meta_prompts():
    """Verifica que no existan meta-prompts o artefactos conocidos en las salidas."""
    forbidden_snippets = [
        "the user wants me",
        "user safety",
        "let me break",
        "coherent with",
        "given answer",
        "here is the",
        "como modelo de",
        "[soloto]",
    ]

    for path in OUTPUT_FILES:
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                q = item["query"].lower()
                r = item["response"].lower()
                for snippet in forbidden_snippets:
                    assert snippet not in q, f"Snippet prohibido '{snippet}' en query de {path.name}:{idx}"
                    # Solo validar r si no es código/json
                    if not item["response"].strip().startswith("<tool_call>") and not item["response"].strip().startswith("```"):
                        assert snippet not in r, f"Snippet prohibido '{snippet}' en response de {path.name}:{idx}"


def test_quarantine_integrity():
    """Verifica que el archivo de cuarentena exista y contenga los campos requeridos."""
    assert QUARANTINE_FILE.exists(), f"Archivo de cuarentena ausente: {QUARANTINE_FILE}"
    assert QUARANTINE_FILE.stat().st_size > 0, "Archivo de cuarentena vacío"

    valid_filters = {"idioma", "artefacto", "asimetria", "asr_leak"}
    count = 0
    with open(QUARANTINE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            count += 1
            item = json.loads(line)
            assert "query" in item, "Cuarentena sin clave 'query'"
            assert "response" in item, "Cuarentena sin clave 'response'"
            assert "quarantine_filter" in item, "Cuarentena sin 'quarantine_filter'"
            assert item["quarantine_filter"] in valid_filters, f"Filtro desconocido en cuarentena: {item['quarantine_filter']}"

    assert count >= 1000, f"Cantidad de elementos en cuarentena sospechosamente baja ({count})"


if __name__ == "__main__":
    test_output_files_exist()
    test_unique_schema_no_legacy_keys()
    test_no_asr_leakage()
    test_no_pipeline_artifacts_or_english_meta_prompts()
    test_quarantine_integrity()
    print("✓ Todos los tests de saneamiento pasaron exitosamente.")
