#!/usr/bin/env python3
"""
build_long_docs.py — Arma documentos largos (una conversacion coherente por muestra)
para el entrenamiento incremental infctx (ctx 1M, VRAM ~ chunk_ctx).

Fuentes:
  A1 (preferida): data/hermes_qa_7k.jsonl agrupada por session_id. Un documento =
      todas las muestras de esa sesion concatenadas en orden, en formato chat:
      <|im_user|>{u}<|im_end|>\\n<|im_start|>{voz}\\n{a}<|im_end|>
  A2: data/youtube_asr_dataset.jsonl agrupada por video_id (mismo formato chat).
      Son QA de transcripcion/comprension derivados de los streams; cada video =
      un documento (no se mezclan videos distintos: el estado infctx arrastra
      identidad y mezclar rompe la coherencia).
  B: transcripts crudos de video-rag: <index_root>/<stem>.transcript.json con forma
      [{speaker, audio_times, transcription:[{segment,...}]}]. Cada archivo = un
      documento en texto plano "Speaker: segmento" por linea. index_root se lee de
      ~/.config/video-rag/config.toml (no se asume).
      FILTRO POR CONTENIDO (corpus SOLO streams de estudio): si el texto menciona
      workmatch, sherut, podcast o cliente, el archivo queda afuera y se lista
      en estadisticas.json con su regla.
  C: transcripts crudos de YouTube: <txt_dir>/<videoId>_trozo_NN.txt generados con
      whisper.cpp en una pasada (sin hablantes, sin marcas). Un documento = todos los
      trozos del mismo video concatenados en orden _NN, en texto plano. No se parte
      en trozos si supera max_tokens: se emite lo que entra y se registra el descarte.

Reglas: --target-tokens (default 1000000: un documento junta varios elementos de
fuentes/temas distintos hasta llegar al target; el salto de tema lleva UN turno
conector generado por el gateway con voz Kateto),
--max-tokens (default 1000000, compat: mismo rol que target),
--min-tokens (default 1000: por debajo el ELEMENTO se descarta).
Si un elemento solo ya supera el target, se emite igual en su propio documento
(nunca se corta un transcript a la mitad).
Salida: data/largo/rwkv_kateto_largo_train.jsonl ({"text": ...} por linea) +
data/largo/rwkv_kateto_largo_youtube.jsonl + data/largo/estadisticas.json.
--stats imprime conteo, histograma y total.
Tokens estimados con len(texto)/3.2 (barato; el tokenizer real no compra nada aca).
"""
import argparse
import glob
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT / "data"
HERMES_QA = DATA_DIR / "hermes_qa_7k.jsonl"
YOUTUBE_ASR = DATA_DIR / "youtube_asr_dataset.jsonl"
OUT_DIR = DATA_DIR / "largo"
OUT_JSONL = OUT_DIR / "rwkv_kateto_largo_train.jsonl"
OUT_YOUTUBE = OUT_DIR / "rwkv_kateto_largo_youtube.jsonl"
OUT_STATS = OUT_DIR / "estadisticas.json"
DEFAULT_CONECTORES_CACHE = OUT_DIR / "conectores_cache.json"
VIDEO_RAG_CONFIG = Path(os.path.expanduser("~/.config/video-rag/config.toml"))
DEFAULT_TXT_DIR = Path(os.environ.get("KATETO_TRANSCRIPTS", "/home/chaos/tmp/kateto-corpus/transcripts"))

EST_PER_TOKEN = 3.2  # chars por token (estimador barato del brief)
BUCKETS = [8000, 16000, 32000, 64000, 128000, 256000, 512000, 1048576]

# (subcadena minuscula, regla informada)
FILTER_RULES = [
    ("workmatch", "menciona WorkMatch (proyecto/cliente, no stream de estudio)"),
    ("sherut", "menciona Sherut (proyecto/cliente)"),
    ("podcast", "menciona podcast (no es stream de estudio)"),
    # NOTA: "cliente" es ruidoso (pesca "open cliente", "cliente archinstall",
    # "comunicacion entre clientes..."). Se aplica literal igual porque el brief
    # lo pide, y se deja el contexto en estadisticas.json para auditar.
    ("cliente", "menciona 'cliente' (material de cliente)"),
]


def est_tokens(text):
    return len(text) / EST_PER_TOKEN


def chat_turn(user, answer, voice):
    return f"<|im_user|>{user}<|im_end|>\n<|im_start|>{voice}\n{answer}<|im_end|>"


def read_index_root(override=None):
    if override:
        return override
    try:
        txt = VIDEO_RAG_CONFIG.read_text(encoding="utf-8")
        m = re.search(r'^\s*index_root\s*=\s*"([^"]+)"', txt, re.M)
        if m:
            return m.group(1)
    except FileNotFoundError:
        pass
    return None


def load_fuente_a(voice):
    """Devuelve (docs, info). docs: lista de (doc_id, text, fuente)."""
    docs = []
    info = {"hermes_sesiones": 0, "hermes_muestras": 0, "youtube_videos": 0, "youtube_muestras": 0}
    if HERMES_QA.exists():
        sesiones = OrderedDict()
        with open(HERMES_QA, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                info["hermes_muestras"] += 1
                q = (d.get("question") or d.get("query") or "").strip()
                a = (d.get("answer") or d.get("response") or "").strip()
                if not q or not a:
                    continue
                sesiones.setdefault(d.get("session_id", "?"), []).append((q, a))
        info["hermes_sesiones"] = len(sesiones)
        for sid, pares in sesiones.items():
            text = "\n".join(chat_turn(q, a, voice) for q, a in pares) + "\n"
            docs.append((f"hermes:{sid}", text, "hermes_session"))
    if YOUTUBE_ASR.exists():
        videos = OrderedDict()
        with open(YOUTUBE_ASR, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                info["youtube_muestras"] += 1
                q = (d.get("query") or d.get("question") or "").strip()
                a = (d.get("response") or d.get("answer") or "").strip()
                if not q or not a:
                    continue
                videos.setdefault(d.get("video_id", "?"), []).append((q, a))
        info["youtube_videos"] = len(videos)
        for vid, pares in videos.items():
            text = "\n".join(chat_turn(q, a, voice) for q, a in pares) + "\n"
            docs.append((f"youtube:{vid}", text, "youtube_video"))
    return docs, info


def transcript_text(dd):
    parts = []
    for item in dd:
        speaker = item.get("speaker", "?")
        for tr in item.get("transcription", []) or []:
            seg = (tr.get("segment") or "").strip()
            if seg:
                parts.append(f"{speaker}: {seg}")
    return "\n".join(parts) + ("\n" if parts else "")


def load_fuente_b(index_root):
    """Devuelve (docs, excluidos, info)."""
    docs, excluidos = [], []
    info = {"archivos": 0, "vacios": 0}
    if not index_root or not os.path.isdir(index_root):
        info["error"] = f"index_root inexistente: {index_root}"
        return docs, excluidos, info
    files = sorted(glob.glob(os.path.join(index_root, "*.transcript.json")))
    info["archivos"] = len(files)
    for f in files:
        stem = os.path.basename(f)
        try:
            dd = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            excluidos.append({"file": stem, "regla": f"ilegible ({e})", "contexto": ""})
            continue
        txt = transcript_text(dd) if isinstance(dd, list) else ""
        if not txt.strip():
            info["vacios"] += 1
            excluidos.append({"file": stem, "regla": "vacio/sin texto", "contexto": ""})
            continue
        tl = txt.lower()
        hit = None
        for sub, regla in FILTER_RULES:
            if sub in tl:
                m = re.search(r".{0,60}" + re.escape(sub) + r".{0,60}", tl)
                hit = (regla, (m.group(0) if m else "")[:160])
                break
        if hit:
            excluidos.append({"file": stem, "regla": hit[0], "contexto": hit[1]})
            continue
        docs.append((f"videorag:{stem.replace('.transcript.json', '')}", txt, "videorag_transcript"))
    return docs, excluidos, info


def load_fuente_c(txt_dir):
    """Lee transcripts de YouTube en texto plano (<videoId>_trozo_NN.txt).
    Un documento = todos los trozos del mismo video concatenados en orden _NN.
    Devuelve (docs, info). docs: lista de (doc_id, text, fuente).
    """
    docs = []
    info = {"directorio": str(txt_dir) if txt_dir else None, "archivos": 0, "videos": 0}
    if not txt_dir or not os.path.isdir(txt_dir):
        info["error"] = f"txt_dir inexistente: {txt_dir}"
        return docs, info
    files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    info["archivos"] = len(files)
    pattern = re.compile(r"^(.+)_trozo_(\d+)\.txt$")
    videos = OrderedDict()
    for f in files:
        base = os.path.basename(f)
        m = pattern.match(base)
        if not m:
            continue
        vid, idx = m.group(1), int(m.group(2))
        videos.setdefault(vid, []).append((idx, f))
    info["videos"] = len(videos)
    for vid, chunks in sorted(videos.items()):
        chunks.sort(key=lambda t: t[0])
        parts = []
        for _, fpath in chunks:
            try:
                with open(fpath, encoding="utf-8") as fh:
                    t = fh.read().strip()
                    if t:
                        parts.append(t)
            except Exception as e:
                pass
        if parts:
            text = "\n".join(parts) + "\n"
            docs.append((f"youtube_txt:{vid}", text, "youtube_txt"))
    return docs, info


def split_max(doc_id, text, fuente, max_tokens):
    """Pica docs que superan --max-tokens en trozos de max_tokens (por chars)."""
    if est_tokens(text) <= max_tokens:
        return [(doc_id, text, fuente)]
    cap = int(max_tokens * EST_PER_TOKEN)
    lines = text.splitlines(keepends=True)
    out, cur, n = [], "", 0
    for ln in lines:
        if len(cur) + len(ln) > cap and cur:
            n += 1
            out.append((f"{doc_id}#p{n}", cur, fuente))
            cur = ""
        cur += ln
    if cur.strip():
        n += 1
        out.append((f"{doc_id}#p{n}", cur, fuente))
    return out


def resolve_api_key(explicit=None):
    """Misma key que build_qa_dataset.py: explicita > OPENAI_API_KEY >
    FREELLMAPI_KEY > opencode.json > sk-no-key."""
    if explicit:
        return explicit
    for k in ("OPENAI_API_KEY", "FREELLMAPI_KEY"):
        v = os.getenv(k)
        if v:
            return v
    try:
        cfg = Path.home() / ".config" / "opencode" / "opencode.json"
        if cfg.exists():
            m = re.search(r"freellmapi-[A-Za-z0-9_-]+", cfg.read_text())
            if m:
                return m.group(0)
    except Exception:
        pass
    return "sk-no-key"


def _retry_after_s(err, default):
    """Si el 429 trae Retry-After, respetarlo. Devuelve segundos (float)."""
    try:
        ra = err.headers.get("Retry-After") if getattr(err, "headers", None) else None
        if ra is not None and str(ra).strip():
            return max(float(str(ra).strip().split(",")[0]), 0.0)
    except Exception:
        pass
    return default


def gateway_chat(base_url, model, key, prompt, max_tokens, timeout, retries=4,
                 espera_max_s=120, contador=None):
    """POST {base}/chat/completions con stdlib (sin deps nuevas).
    Devuelve (texto, agotado_por_429). Nunca levanta: ante corte del gateway
    devuelve ("", True/False) y la corrida sigue sin conector.
    Backoff creciente 5s, 15s, 45s... (5*3**i) capado por espera_max_s;
    si hay Retry-After se respeta (también capado). contador (dict opcional)
    suma cada 429/5xx visto para el reporte."""
    system = ("Sos Kateto (seco), streamer argentino. Respondes UNICAMENTE con el turno "
              "de transicion pedido, en rioplatense con voseo, una o dos frases. "
              "Sin explicaciones, sin comillas, sin marcadores, sin hablar de vos mismo.")
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "temperature": 0.9, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    last = None
    for i in range(retries):
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.loads(r.read())
            return (((d.get("choices") or [{}])[0].get("message", {}).get("content") or "").strip(),
                    False)
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429 or 500 <= e.code < 600:
                if contador is not None:
                    contador["n_429"] = contador.get("n_429", 0) + 1
                base_wait = 5 * (3 ** i)
                espera = _retry_after_s(e, base_wait)
                espera = min(max(espera, 0.0), espera_max_s)
                print(f"    conector: gateway {e.code}, reintento {i + 1}/{retries} en {espera:g}s")
                time.sleep(espera)
                continue
            try:
                print(f"    conector: gateway {e.code} {e.read()[:200]}")
            except Exception:
                print(f"    conector: gateway {e.code}")
            return "", False
        except Exception as e:
            last = e
            print(f"    conector: fallo gateway ({e}), sigo sin conector")
            return "", False
    print(f"    conector: reintentos agotados ({last}), sigo sin conector")
    return "", True


def prompt_conector(prev_id, prev_tail, next_id, next_text_head, voice):
    return (f"Veniamos hablando de esto:\n\"\"\"{prev_tail[-400:]}\"\"\"\n"
            f"Ahora la charla salta a esto otro:\n\"\"\"{next_text_head[:400]}\"\"\"\n"
            f"Escribi el turno que cierra lo primero y abre lo segundo. "
            f"No expliques nada tecnico ni inventes datos del tema nuevo.\n"
            f"Ejemplo de respuesta valida (solo el turno, nada mas):\n"
            f"Bueno, eso ya fue, dejalo ahi. Che, ahora contame que onda con esto otro que me pasaste.\n"
            f"Devolve SOLO el turno: no lo presentes, no lo expliques, no digas 'aca va' ni "
            f"describas lo que vas a hacer. Directamente el turno.")


# Respuestas-meta del teacher (explica la tarea en vez de dar el turno): se
# descartan, el salto queda sin conector y se cuenta como omitido.
_RX_META = re.compile(r"first,|transition turn|previous topic|new topic|as an ai|"
                      r"como (modelo|ia|asistente)|lo siento|lamento|en que te (ayudo|puedo ayudar)|"
                      r"el usuario (quiere|escribi)|bloque de texto|como modelo de lenguaje",
                      re.I)


def limpiar_conector(raw, max_chars=400):
    t = (raw or "").strip().strip('"').strip()
    t = re.sub(r"^(KATETO|SECO|VOS|CHAT|ASSISTANT)\s*:\s*", "", t, flags=re.I).strip()
    t = t.split("\n")[0].strip()
    if _RX_META.search(t):
        return ""
    return t[:max_chars].strip()


class Conectores:
    """Genera y cachea turnos de transicion. Clave = par de elementos + voz."""

    def __init__(self, args):
        self.off = args.no_conectores
        self.cupo = args.max_conectores  # solo llamadas reales al gateway
        self.args = args
        self.cache_path = Path(args.conectores_cache)
        try:
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) \
                if self.cache_path.exists() else {}
        except Exception:
            self.cache = {}
        self.generados = 0
        self.hits = 0
        self.omitidos = 0
        self.omitidos_429 = 0
        self.llamadas = 0  # llamadas reales al gateway (exitos + 429; cache no cuenta)
        self.contador = {"n_429": 0}

    def turno(self, prev_id, prev_text, next_id, next_text):
        if self.off:
            return None
        key = f"{prev_id}>>{next_id}::{self.args.voice}"
        if key in self.cache and self.cache[key].strip():
            self.hits += 1
            return self.cache[key]
        if self.cupo is not None and self.llamadas >= self.cupo:
            self.omitidos += 1
            return None
        delay = getattr(self.args, "conectores_delay_s", 0) or 0
        if delay > 0:
            time.sleep(delay)
        self.llamadas += 1
        base = (self.args.openai_base_url or os.getenv("OPENAI_BASE_URL")
                or os.getenv("LLAMA_BASE_URL") or "http://127.0.0.1:3001/v1")
        model = (self.args.openai_model or os.getenv("MODEL_NAME")
                  or os.getenv("OPENAI_MODEL") or "auto")
        key_api = resolve_api_key(self.args.openai_api_key)
        prompt = prompt_conector(prev_id, prev_text, next_id, next_text, self.args.voice)
        raw, agotado_429 = gateway_chat(base, model, key_api, prompt,
                           self.args.conector_max_tokens, self.args.conector_timeout,
                           retries=getattr(self.args, "conectores_retries", 4) or 4,
                           espera_max_s=getattr(self.args, "conectores_espera_max_s", 120) or 120,
                           contador=self.contador)
        if agotado_429:
            self.omitidos_429 += 1
            return None
        txt = limpiar_conector(raw)
        if not txt:
            self.omitidos += 1
            return None
        self.cache[key] = txt
        self.generados += 1
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(self.cache, open(self.cache_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
        except Exception as e:
            print(f"    conector: no pude guardar cache ({e})")
        return txt

    def stats(self):
        return {"generados_gateway": self.generados, "hits_cache": self.hits,
                "omitidos": self.omitidos, "omitidos_por_429": self.omitidos_429,
                "respuestas_429": self.contador.get("n_429", 0),
                "cache_path": str(self.cache_path),
                "cache_entradas": len(self.cache)}


def agrupar(elementos, target_tokens, conectores, voice, max_documentos=None):
    """Junta elementos hasta el target. Cada salto lleva un turno conector de voz.
    Un elemento solo que supera el target sale en su propio documento (sin cortar).
    Devuelve lista de (doc_id, text, elementos:[(id, fuente)], n_conectores)."""
    # Mas grandes primero: llena cerca del target con menos saltos.
    orden = sorted(elementos, key=lambda t: -len(t[1]))
    docs, cur, cur_chars = [], [], 0
    pendientes = 0
    cap = int(target_tokens * EST_PER_TOKEN)
    lleno = lambda: max_documentos is not None and len(docs) >= max_documentos

    def emitir():
        nonlocal cur, cur_chars
        if not cur:
            return
        partes, ids, n_con = [], [], 0
        for j, (eid, etext, efuente) in enumerate(cur):
            if j > 0:
                c = conectores.turno(ids[-1][0], cur[j - 1][1], eid, etext)
                if c:
                    partes.append(f"<|im_start|>{voice}\n{c}<|im_end|>")
                    n_con += 1
            partes.append(etext if etext.endswith("\n") else etext + "\n")
            ids.append((eid, efuente))
        did = f"mix:{ids[0][0]}+{len(ids) - 1}" if len(ids) > 1 else ids[0][0]
        docs.append((did, "".join(partes), ids, n_con))
        cur, cur_chars = [], 0

    for el in orden:
        if lleno():
            pendientes += 1
            continue
        if cur and cur_chars + len(el[1]) > cap:
            emitir()
        if lleno():
            pendientes += 1
            continue
        cur.append(el)
        cur_chars += len(el[1])
        if cur_chars >= cap:
            emitir()
    if cur:
        if lleno():
            pendientes += len(cur)
        else:
            emitir()
    return docs, pendientes


def build(args):
    index_root = read_index_root(args.index_root)
    txt_dir = args.txt_dir
    docs_a, info_a = load_fuente_a(args.voice)
    docs_b, excluidos_b, info_b = load_fuente_b(index_root)
    docs_c, info_c = load_fuente_c(txt_dir)

    # Elementos: unidad minima (nunca se corta). Filtro de contenido intacto
    # (load_fuente_b ya excluyo workmatch/sherut/podcast/cliente).
    elementos = [(i, t, f) for i, t, f in docs_a + docs_b + docs_c]
    cortos_elem = 0
    chicos = []
    for el in elementos:
        if est_tokens(el[1]) < args.min_tokens:
            cortos_elem += 1
        else:
            chicos.append(el)

    conectores = Conectores(args)
    docs, pendientes = agrupar(chicos, args.target_tokens, conectores, args.voice,
                               max_documentos=args.max_documentos)
    largos, cortos_doc = [], 0
    for doc_id, text, ids, n_con in docs:
        if est_tokens(text) < args.min_tokens and len(docs) > 1:
            cortos_doc += 1  # cola que no llego al minimo
            continue
        largos.append((doc_id, text, [f for _, f in ids], [i for i, _ in ids], n_con))

    largos.sort(key=lambda t: -len(t[1]))
    toks = [est_tokens(t) for _, t, _, _, _ in largos]
    hist = {}
    lo = 0
    for b in BUCKETS:
        hist[f"{lo}-{b}"] = sum(1 for x in toks if lo <= x < b)
        lo = b
    hist[f">={BUCKETS[-1]}"] = sum(1 for x in toks if x >= BUCKETS[-1])
    por_fuente = {}
    for _, _, fuentes, _, _ in largos:
        for f in set(fuentes):
            por_fuente[f] = por_fuente.get(f, 0) + 1
    stats = {
        "params": {"target_tokens": args.target_tokens, "max_tokens": args.max_tokens,
                   "min_tokens": args.min_tokens, "voice": args.voice, "index_root": index_root,
                   "txt_dir": str(txt_dir) if txt_dir else None,
                    "no_conectores": args.no_conectores, "max_documentos": args.max_documentos,
                    "max_conectores": args.max_conectores,
                    "conectores_delay_s": getattr(args, "conectores_delay_s", 0),
                    "conectores_retries": getattr(args, "conectores_retries", 4),
                    "conectores_espera_max_s": getattr(args, "conectores_espera_max_s", 120)},
        "fuente_a": info_a,
        "fuente_b": info_b,
        "fuente_c": info_c,
        "n_docs": len(largos),
        "n_elementos": len(chicos),
        "elementos_pendientes_por_max_documentos": pendientes,
        "descartados_por_min_elemento": cortos_elem,
        "descartados_por_min_doc_cola": cortos_doc,
        "total_tokens_est": round(sum(toks), 1),
        "max_tokens_est": round(max(toks), 1) if toks else 0,
        "docs_ge_target": sum(1 for x in toks if x >= args.target_tokens),
        "histograma_tokens_est": hist,
        "por_fuente": por_fuente,
        "n_docs_mixtos": sum(1 for _, _, f, _, _ in largos if len(set(f)) > 1),
        "conectores": conectores.stats(),
        "truncados_fuente_c": [],
        "excluidos_fuente_b": excluidos_b,
        "docs": [{"id": i, "fuentes": sorted(set(s)), "n_elementos": len(eids),
                  "n_conectores": nc, "chars": len(t), "tokens_est": round(est_tokens(t), 1)}
                 for i, t, s, eids, nc in largos],
    }
    return largos, stats


def main():
    ap = argparse.ArgumentParser(description="Arma documentos largos para infctx (1 doc = 1 conversacion).")
    ap.add_argument("--out", default=str(OUT_JSONL))
    ap.add_argument("--out-youtube", default=str(OUT_YOUTUBE), help="Salida para documentos de YouTube txt solos")
    ap.add_argument("--stats-json", default=str(OUT_STATS))
    ap.add_argument("--target-tokens", type=float, default=1000000)
    ap.add_argument("--max-tokens", type=float, default=1000000)
    ap.add_argument("--min-tokens", type=float, default=1000)
    ap.add_argument("--voice", default="seco")
    ap.add_argument("--no-conectores", action="store_true",
                    help="une elementos sin llamar al gateway (sirve sin modelo)")
    ap.add_argument("--max-documentos", type=int, default=None,
                    help="tope de documentos emitidos (para corridas chicas)")
    ap.add_argument("--max-conectores", type=int, default=None,
                    help="tope de llamadas reales al gateway (cache no cuenta)")
    ap.add_argument("--conectores-cache", default=str(DEFAULT_CONECTORES_CACHE))
    ap.add_argument("--openai-base-url", default=None,
                    help="default env OPENAI_BASE_URL/LLAMA_BASE_URL o http://127.0.0.1:3001/v1")
    ap.add_argument("--openai-model", default=None,
                    help="default env MODEL_NAME/OPENAI_MODEL o auto")
    ap.add_argument("--openai-api-key", default=None,
                    help="default env OPENAI_API_KEY/FREELLMAPI_KEY u opencode.json")
    ap.add_argument("--conector-timeout", type=float, default=120)
    ap.add_argument("--conector-max-tokens", type=int, default=120)
    ap.add_argument("--conectores-delay-s", type=float, default=2.0,
                    help="pausa antes de cada llamada real al gateway (cache no duerme)")
    ap.add_argument("--conectores-retries", type=int, default=4,
                    help="intentos por conector ante 429/5xx (backoff 5s,15s,45s... respeta Retry-After)")
    ap.add_argument("--conectores-espera-max-s", type=float, default=120,
                    help="tope por espera del backoff; agotados se omiten y cuentan como omitidos_por_429")
    ap.add_argument("--index-root", default=None)
    default_txt = str(DEFAULT_TXT_DIR) if DEFAULT_TXT_DIR.is_dir() else None
    ap.add_argument("--txt-dir", default=default_txt, help="Directorio con transcripts .txt de YouTube")
    ap.add_argument("--stats", action="store_true", help="imprime histograma y totales")
    args = ap.parse_args()
    largos, stats = build(args)
    if args.stats:
        print(f"documentos largos: {stats['n_docs']} "
              f"(elementos<{args.min_tokens:g}tok descartados: {stats['descartados_por_min_elemento']}, "
              f"colas: {stats['descartados_por_min_doc_cola']})")
        print(f"total tokens est: {stats['total_tokens_est']:.0f} | "
              f"max doc: {stats['max_tokens_est']:.0f} | >=target({args.target_tokens:g}): "
              f"{stats['docs_ge_target']} | mixtos: {stats['n_docs_mixtos']}")
        c = stats["conectores"]
        print(f"conectores: gateway={c['generados_gateway']} cache={c['hits_cache']} "
              f"omitidos={c['omitidos']} omitidos_por_429={c.get('omitidos_por_429', 0)} "
              f"respuestas_429={c.get('respuestas_429', 0)}")
        print("histograma (tokens est):")
        for k, v in stats["histograma_tokens_est"].items():
            print(f"  {k}: {v}")
        print(f"por fuente (docs que la incluyen): {stats['por_fuente']}")
        print(f"fuente A: {stats['fuente_a']} | fuente B: {stats['fuente_b']} | fuente C: {stats['fuente_c']}")
        if stats.get("truncados_fuente_c"):
            print(f"truncados fuente C ({len(stats['truncados_fuente_c'])}):")
            for t in stats["truncados_fuente_c"]:
                print(f"  {t['id']}: {t['tokens_original']} -> {t['tokens_emitidos']} (-{t['tokens_descartados']})")
        if stats["excluidos_fuente_b"]:
            print(f"excluidos fuente B ({len(stats['excluidos_fuente_b'])}):")
            for e in stats["excluidos_fuente_b"]:
                print(f"  {e['file']}: {e['regla']}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for _, text, _, _, _ in largos:
            fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    if args.out_youtube:
        out_yt = Path(args.out_youtube)
        out_yt.parent.mkdir(parents=True, exist_ok=True)
        yt_docs = [(i, t) for i, t, s, _, _ in largos
                   if s and all(f == "youtube_txt" for f in s)]
        with open(out_yt, "w", encoding="utf-8") as fh:
            for _, text in yt_docs:
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        print(f"escritos {len(yt_docs)} docs solo-youtube_txt -> {out_yt}")
    sj = Path(args.stats_json)
    sj.parent.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(sj, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"escritos {len(largos)} docs -> {out} + {sj}")


if __name__ == "__main__":
    main()
