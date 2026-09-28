#!/usr/bin/env python3
"""Extraccion de turnos reales (NO_RESPONSE / WAIT / artefactos de ASR).

Fuente: transcripts de video-rag (<nombre>.transcript.json).
Solo stdlib. Idempotente: reescribe, nunca appendea.
"""
import argparse
import json
import re
import sys
from pathlib import Path

KEYWORDS = ("workmatch", "warmatch", "sherut", "podcast", "entrevista")
ACOUSTIC = ("musica", "música", "aplausos", "risas", "silencio", "ruido")
CLOSING = ("subtitulos", "subtítulos", "amara.org", "suscrib",
           "dale like", "gracias por ver")
CONNECTORS = frozenset([
    "que", "y", "pero", "porque", "o", "para", "de", "en", "con", "si",
    "cuando", "como", "cómo", "este", "eh", "ehh", "mmm", "bueno",
    "pará", "para", "esperá", "espera",
])
MULETILLAS = frozenset(["eh", "ehh", "mmm", "este", "bueno"])
ENGLISH = frozenset(["checkout", "feed", "deploy", "build", "merge",
                     "branch", "commit", "push", "pull", "issue", "script"])
NUMBERS = frozenset([
    "cero", "uno", "una", "dos", "tres", "cuatro", "cinco", "seis",
    "siete", "ocho", "nueve", "diez", "once", "doce", "trece", "catorce",
    "quince", "dieciseis", "dieciséis", "veinte", "treinta", "cuarenta",
    "cincuenta", "sesenta", "setenta", "ochenta", "noventa", "cien",
    "ciento", "cientos", "mil", "miles", "millon", "millón", "millones",
])
WORD_RE = re.compile(r"[\wáéíóúüñÁÉÍÓÚÜÑ]+", re.UNICODE)
ALPHA_RE = re.compile(r"[a-záéíóúüñ]+")
FMT = re.compile(
    r"<\|im_user\|>.+<\|im_end\|>\n<\|im_start\|>seco\n"
    r"<\|(?:no_response|wait)\|><\|im_end\|>", re.S)
CAP_POR_VIDEO = 40
PAUSA_MIN = 1.8


def norm(t):
    return re.sub(r"\s+", " ", t.strip().lower())


def video_name(p):
    n = p.name
    return n[:-len(".transcript.json")] if n.endswith(".transcript.json") else p.stem


def load_turns(data):
    turns = []
    for block in data:
        at = block.get("audio_times") or [0, 0]
        try:
            base = float(at[0])
        except (TypeError, ValueError, IndexError):
            base = 0.0
        for tr in block.get("transcription") or []:
            seg = tr.get("segment")
            if not seg:
                continue
            t = seg.strip()
            if not t:
                continue
            try:
                ini = base + float(tr.get("start_timestamp", 0)) / 100.0
                fin = base + float(tr.get("end_timestamp", 0)) / 100.0
            except (TypeError, ValueError):
                continue
            turns.append({"texto": t, "inicio": ini, "fin": fin})
    return turns


def es_ruido(t):
    low = t.lower()
    if "www." in low or "http" in low:
        return True
    if "[" in t and "]" in t and any(k in low for k in ACOUSTIC):
        return True
    if any(k in low for k in CLOSING):
        return True
    toks = ALPHA_RE.findall(low)
    if 4 <= len(toks) <= 8 and len(set(toks)) == 1:
        return True
    return False


def ultima_palabra(t):
    toks = WORD_RE.findall(t.lower())
    return toks[-1] if toks else ""


def es_wait(t, pausa):
    if pausa < PAUSA_MIN:
        return False
    s = t.strip()
    if not s or s[-1] in ".?!":
        return False
    if len(WORD_RE.findall(s)) < 3:
        return False
    return ultima_palabra(s) in CONNECTORS


def artefactos(t):
    out = []
    s = t.strip()
    low = t.lower()
    toks = WORD_RE.findall(low)
    if not s or s[-1] not in ".?!":
        out.append("sin_cierre")
    if any(w in MULETILLAS for w in toks) or re.search(r"\bo\s+sea\b", low):
        out.append("muletillas")
    if any(a == b for a, b in zip(toks, toks[1:])):
        out.append("repeticion_consecutiva")
    if any(w in ENGLISH for w in toks):
        out.append("palabra_inglesa")
    if "www." in low or "http" in low:
        out.append("url_o_dominio")
    if re.search(r"\d", t) or any(w in NUMBERS for w in toks):
        out.append("numero_dicho")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    idx = Path(a.index_root)
    files = sorted(idx.glob("*.transcript.json"))
    excluidos = []
    incluidos = []
    for f in files:
        low_name = f.name.lower()
        if f.stat().st_size < 100:
            excluidos.append({"archivo": f.name, "regla": "tamanio_menor_100b"})
            continue
        if any(k in low_name for k in KEYWORDS):
            excluidos.append({"archivo": f.name, "regla": "nombre"})
            continue
        try:
            raw = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            excluidos.append({"archivo": f.name, "regla": "error_lectura"})
            continue
        if any(k in raw.lower() for k in KEYWORDS):
            excluidos.append({"archivo": f.name, "regla": "contenido"})
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            excluidos.append({"archivo": f.name, "regla": "json_invalido"})
            continue
        incluidos.append((f, data))
    if a.limit is not None:
        incluidos = incluidos[:a.limit]

    no_resp, waits, prompts = [], [], []
    seen_no, seen_wait, seen_pr = set(), set(), set()
    por_video = {}
    descartadas = 0

    for f, data in incluidos:
        v = video_name(f)
        c = por_video.setdefault(v, {"no_response": 0, "wait": 0, "asr_prompts": 0})
        turns = load_turns(data)
        n_no = n_wait = 0
        for i, tr in enumerate(turns):
            t = tr["texto"]
            if es_ruido(t):
                k = norm(t)
                if k not in seen_no and n_no < CAP_POR_VIDEO:
                    txt = ("<|im_user|>" + t + "<|im_end|>\n<|im_start|>seco\n"
                           "<|no_response|><|im_end|>")
                    if FMT.fullmatch(txt):
                        seen_no.add(k)
                        no_resp.append({"text": txt, "scenario": "real_asr_noise",
                                        "cat": "F", "video": v, "fuente": "real"})
                        n_no += 1
                    else:
                        descartadas += 1
        for i, tr in enumerate(turns[:-1]):
            pausa = turns[i + 1]["inicio"] - tr["fin"]
            t = tr["texto"]
            if es_wait(t, pausa):
                k = norm(t)
                if k not in seen_wait and n_wait < CAP_POR_VIDEO:
                    txt = ("<|im_user|>" + t + "<|im_end|>\n<|im_start|>seco\n"
                           "<|wait|><|im_end|>")
                    if FMT.fullmatch(txt):
                        seen_wait.add(k)
                        waits.append({"text": txt, "scenario": "real_asr_wait",
                                      "cat": "G", "video": v, "fuente": "real"})
                        n_wait += 1
                    else:
                        descartadas += 1
        for tr in turns:
            t = tr["texto"]
            if len(WORD_RE.findall(t)) < 4:
                continue
            k = norm(t)
            if k in seen_pr:
                continue
            seen_pr.add(k)
            prompts.append({"prompt": ("<|im_user|>" + t + "<|im_end|>\n"
                                       "<|im_start|>seco\n"),
                            "video": v, "inicio_s": round(tr["inicio"], 2),
                            "fin_s": round(tr["fin"], 2),
                            "artefactos": artefactos(t)})
        c["no_response"], c["wait"] = n_no, n_wait
        c["asr_prompts"] = sum(1 for p in prompts if p["video"] == v)

    por_video = {v: c for v, c in por_video.items() if sum(c.values()) > 0}
    tot_cat = {"no_response": len(no_resp), "wait": len(waits),
               "asr_prompts": len(prompts)}
    resumen = ("videos_procesados=%d excluidos=%d no_response=%d wait=%d "
               "asr_prompts=%d descartadas_por_formato=%d"
               % (len(incluidos), len(excluidos), len(no_resp), len(waits),
                  len(prompts), descartadas))
    print(resumen)
    for e in excluidos:
        print("excluido: %s (%s)" % (e["archivo"], e["regla"]))

    if a.dry_run:
        return 0

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("real_no_response.jsonl", no_resp),
                       ("real_wait.jsonl", waits),
                       ("real_asr_prompts.jsonl", prompts)):
        with open(out / name, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"script": "scripts/extract_real_turns.py",
           "test_resultado": "", "dry_run_limit2": "",
           "archivos_excluidos": excluidos,
           "totales_por_categoria": tot_cat, "totales_por_video": por_video,
           "descartadas_por_formato": descartadas,
           "ejemplos_por_categoria": {"no_response": no_resp[:3],
                                      "wait": waits[:3]},
           "dudas": []}
    rp = Path(a.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
