#!/usr/bin/env python3
"""Tests stdlib para scripts/extract_real_turns.py.

Transcripts sinteticos en tmp dirs (no toca el index real).
Directo: python3 scripts/test_extract_real_turns.py
Pytest:  python3 -m pytest scripts/test_extract_real_turns.py -q
"""
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("extract_real_turns.py")
_spec = importlib.util.spec_from_file_location("extract_real_turns", SCRIPT)
_er = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_er)
FMT = re.compile(
    r"<\|im_user\|>.+<\|im_end\|>\n<\|im_start\|>seco\n"
    r"<\|(?:no_response|wait)\|><\|im_end\|>", re.S)

FRASES_REALES = [
    "Propiedad asociativa. No importa que agrupes algunos términos en una "
    "suma, el resultado sigue siendo el mismo.",
    "La raíz de 50 es 5 raíz 2, la raíz de 72 es 6 raíz 2 y la raíz de 90 "
    "es 3 raíz 10.",
    "Por ejemplo, el modelo tenía que hacer paso por paso, o varios modelos "
    "tenían que hacer paso por paso.",
]
RUIDO_REAL = [
    "No, no, no, no.",
    "Gracias por ver el video.",
    "Más información en www.ejemplo.com",
]


def write_tr(path, segs, name="vid.transcript.json"):
    """segs: lista de (texto, inicio_s, fin_s). Un bloque base 0."""
    tr = [{"start_timestamp": int(s * 100), "end_timestamp": int(e * 100),
           "segment": t, "words": []} for t, s, e in segs]
    data = [{"speaker": "X", "audio_path": "",
             "audio_times": [0.0, 1.0], "transcription": tr}]
    p = Path(path) / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def run(idx, out, rep, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--index-root", str(idx),
         "--out-dir", str(out), "--report", str(rep), *extra],
        capture_output=True, text=True)


def lines(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


class TestExtract(unittest.TestCase):
    def test_es_ruido_falsos_positivos(self):
        for f in FRASES_REALES:
            self.assertFalse(_er.es_ruido(f), f)

    def test_es_ruido_detecta_ruido(self):
        for f in RUIDO_REAL:
            self.assertTrue(_er.es_ruido(f), f)

    def test_no_response_emision_real(self):
        with tempfile.TemporaryDirectory() as d:
            idx, out = Path(d) / "i", Path(d) / "o"
            idx.mkdir()
            rep = Path(d) / "r.json"
            segs = [(t, i * 2.0, i * 2.0 + 1.5)
                    for i, t in enumerate(FRASES_REALES + RUIDO_REAL)]
            write_tr(idx, segs, "c.transcript.json")
            r = run(idx, out, rep)
            self.assertEqual(r.returncode, 0, r.stderr)
            emitidas = [x["text"] for x in lines(out / "real_no_response.jsonl")]
            for f in FRASES_REALES:
                self.assertFalse(any(f in t for t in emitidas), f)
            for f in RUIDO_REAL:
                self.assertTrue(any(f in t for t in emitidas), f)

    def test_no_response_y_wait_formato(self):
        with tempfile.TemporaryDirectory() as d:
            idx, out = Path(d) / "i", Path(d) / "o"
            idx.mkdir()
            rep = Path(d) / "r.json"
            write_tr(idx, [("[Música]", 0.0, 1.0),
                           ("relleno suficiente para largo minimo", 1.0, 2.0)],
                     "a.transcript.json")
            write_tr(idx, [("quiero decir que", 0.0, 1.0),
                           ("algo nuevo aqui ahora", 3.0, 4.0)],
                     "b.transcript.json")
            r = run(idx, out, rep)
            self.assertEqual(r.returncode, 0, r.stderr)
            no = lines(out / "real_no_response.jsonl")
            wa = lines(out / "real_wait.jsonl")
            self.assertTrue(any("<|no_response|>" in x["text"] for x in no))
            self.assertTrue(any("<|wait|>" in x["text"] for x in wa))
            for x in no + wa:
                self.assertTrue(FMT.fullmatch(x["text"]), x["text"])
            self.assertTrue(all(x["scenario"] == "real_asr_noise" and
                                x["cat"] == "F" and x["fuente"] == "real"
                                for x in no))
            self.assertTrue(all(x["scenario"] == "real_asr_wait" and
                                x["cat"] == "G" and x["fuente"] == "real"
                                for x in wa))

    def test_exclusion_por_contenido(self):
        with tempfile.TemporaryDirectory() as d:
            idx, out = Path(d) / "i", Path(d) / "o"
            idx.mkdir()
            rep = Path(d) / "r.json"
            write_tr(idx, [("hablamos de warmatch en este video de prueba "
                            "con palabras suficientes de relleno", 0.0, 1.0)],
                     "ok.transcript.json")
            r = run(idx, out, rep)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(lines(out / "real_no_response.jsonl"), [])
            self.assertEqual(lines(out / "real_wait.jsonl"), [])
            repj = json.loads(rep.read_text(encoding="utf-8"))
            self.assertTrue(any(e["archivo"] == "ok.transcript.json" and
                                e["regla"] == "contenido"
                                for e in repj["archivos_excluidos"]))

    def test_pausa_corta_no_wait(self):
        with tempfile.TemporaryDirectory() as d:
            idx, out = Path(d) / "i", Path(d) / "o"
            idx.mkdir()
            rep = Path(d) / "r.json"
            write_tr(idx, [("quiero decir que", 0.0, 1.0),
                           ("algo nuevo aqui ahora", 1.5, 2.5)])
            r = run(idx, out, rep)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(lines(out / "real_wait.jsonl"), [])

    def test_dedupe(self):
        with tempfile.TemporaryDirectory() as d:
            idx, out = Path(d) / "i", Path(d) / "o"
            idx.mkdir()
            rep = Path(d) / "r.json"
            write_tr(idx, [("[Aplausos]", 0.0, 1.0), ("[Aplausos]", 2.0, 3.0),
                           ("[Aplausos]", 4.0, 5.0)])
            r = run(idx, out, rep)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(len(lines(out / "real_no_response.jsonl")), 1)

    def test_dry_run_no_crea(self):
        with tempfile.TemporaryDirectory() as d:
            idx, out = Path(d) / "i", Path(d) / "o"
            idx.mkdir()
            rep = Path(d) / "r.json"
            write_tr(idx, [("[Música]", 0.0, 1.0)])
            r = run(idx, out, rep, "--dry-run")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertFalse(out.exists())
            self.assertFalse(rep.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
