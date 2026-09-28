#!/usr/bin/env python3
"""
ingest_youtube_streams.py — Pipeline de ingesta, transcripción ASR con whisper.cpp Vulkan y
extracción de dataset conversacional / ASR / turn-taking a partir de los directos de YouTube.

Características:
- Descarga audio a 16kHz mono WAV con yt-dlp.
- Transcribe con whisper-cli (ggml-large-v3-turbo-q5_0.bin) usando Vulkan en RX 6500 XT.
- Limpia audio crudo tras transcribir para no saturar disco.
- Reanuda automáticamente (salta videos ya transcriptos).
- Extrae:
  1. Ejemplos de transcripción ASR directa y corrección de audio/dictado.
  2. Párrafos de habla espontánea argentina para SFT de estilo y vocabulario.
  3. Muestras de turn-taking: vacilaciones/pausas -> <WAIT>, ruido/hallucination -> <NO_RESPONSE>.
- Salida: data/youtube_asr_dataset.jsonl
"""
import os, sys, re, json, subprocess, argparse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
URLS_FILE = PROJECT / "data" / "youtube_urls.txt"
RAW_AUDIO_DIR = PROJECT / "data" / "youtube_raw_audio"
TRANSCRIPTS_DIR = PROJECT / "data" / "youtube_transcripts"
OUT_DATASET = PROJECT / "data" / "youtube_asr_dataset.jsonl"

WHISPER_CLI = Path(os.environ.get("WHISPER_CLI", "/run/media/chaos/terciario/proyectos/asr-benchmark/engines/whisper-cpp/build/bin/whisper-cli"))
MODEL_BIN = Path(os.environ.get("WHISPER_MODEL_BIN", "/run/media/chaos/terciario/proyectos/asr-benchmark/models/ggml-large-v3-turbo-q5_0.bin"))

SYSTEM_PROMPT = "Sos Kateto, rioplatense seco, con opinion propia."

# Filtro de secretos si está disponible
KATETO_TOOLS = Path.home() / "proyectos" / "OpenaiBuildWeek" / "Kateto"
if str(KATETO_TOOLS) not in sys.path:
    sys.path.insert(0, str(KATETO_TOOLS))
try:
    from kateto.tools.dataset_filter.secrets_filter import contains_secret
except Exception:
    def contains_secret(text):
        class R: found = False
        return R()

def extract_video_id(url: str) -> str:
    url = url.strip()
    m = re.search(r"(?:v=|youtu\.be/|live/|embed/)([A-Za-z0-9_\-]{11})", url)
    if m:
        return m.group(1)
    clean = re.sub(r"[^A-Za-z0-9_\-]", "_", url)
    return clean[-15:]

def download_audio(url: str, video_id: str, out_wav: Path) -> bool:
    print(f"Descargando audio para {video_id} -> {out_wav.name}...")
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "wav",
        "--postprocessor-args", "-ar 16000 -ac 1 -c:a pcm_s16le",
        "-o", str(out_wav),
        url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error descargando {url}: {res.stderr[:200]}")
        return False
    return out_wav.exists()

def transcribe_whisper(wav_path: Path, out_base: Path) -> bool:
    print(f"Transcribiendo con whisper.cpp (turbo-v3-q5 Vulkan): {wav_path.name}...")
    cmd = [
        str(WHISPER_CLI),
        "-m", str(MODEL_BIN),
        "-l", "es",
        "-oj",
        "-osrt",
        "-of", str(out_base),
        "-f", str(wav_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    json_path = out_base.with_suffix(".json")
    if res.returncode != 0 or not json_path.exists():
        print(f"Error transcribiendo {wav_path.name}: {res.stderr[:200]}")
        return False
    return True

def parse_transcripts_to_dataset(transcripts_dir: Path, out_jsonl: Path):
    print(f"\n=== Generando dataset desde transcripciones en {transcripts_dir} ===")
    json_files = sorted(transcripts_dir.glob("*.json"))
    if not json_files:
        print("No se encontraron archivos .json de transcripción.")
        return 0

    dataset_items = []
    seen = set()

    for jf in json_files:
        vid = jf.stem
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error leyendo {jf.name}: {e}")
            continue

        segments = data.get("transcription", [])
        if not segments:
            continue

        chunk = []
        last_to_ms = 0

        for seg in segments:
            offsets = seg.get("offsets", {})
            from_ms = offsets.get("from", 0)
            to_ms = offsets.get("to", 0)
            text = seg.get("text", "").strip()

            if not text:
                continue

            low_text = text.lower()
            # Detección de ruido / alucinaciones ASR -> <NO_RESPONSE>
            if any(h in low_text for h in ["subtítulos por", "amara.org", "suscribite", "dale like", "[música]", "[aplausos]", "gracias por ver"]):
                dataset_items.append({
                    "question": text,
                    "answer": "<NO_RESPONSE><|endoftext|>",
                    "query": text,
                    "response": "<NO_RESPONSE>",
                    "source_label": "youtube-asr-noise",
                    "video_id": vid,
                    "openai": {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": text},
                            {"role": "assistant", "content": "<NO_RESPONSE>"}
                        ]
                    }
                })
                continue

            # Detección de vacilaciones con pausa larga -> <WAIT>
            if (text.endswith("...") or text.endswith(" eh") or text.endswith(" pará")) and (from_ms - last_to_ms > 1800):
                dataset_items.append({
                    "question": text,
                    "answer": "<WAIT><|endoftext|>",
                    "query": text,
                    "response": "<WAIT>",
                    "source_label": "youtube-asr-wait",
                    "video_id": vid,
                    "openai": {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": text},
                            {"role": "assistant", "content": "<WAIT>"}
                        ]
                    }
                })

            chunk.append(text)
            cur_len = sum(len(x) for x in chunk)
            gap = from_ms - last_to_ms if last_to_ms > 0 else 0

            # Acumular hasta completar una idea coherente (80 - 300 caracteres)
            if cur_len >= 70 and (gap > 2000 or cur_len >= 220 or text.endswith((".", "?", "!"))):
                full_speech = " ".join(chunk).strip()
                if 30 <= len(full_speech) <= 650 and not contains_secret(full_speech).found:
                    key = full_speech[:80].lower()
                    if key not in seen:
                        seen.add(key)
                        # Caso 1: Transcripción ASR directa
                        dataset_items.append({
                            "question": f"Transcribí el audio en español rioplatense:\n\"{full_speech.lower()}\"",
                            "answer": full_speech,
                            "query": f"Transcribí el audio en español rioplatense:\n\"{full_speech.lower()}\"",
                            "response": full_speech,
                            "source_label": "youtube-asr-transcription",
                            "video_id": vid,
                            "openai": {
                                "messages": [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": f"Transcribí el siguiente fragmento de audio con puntuación y formato adecuado:\n\"{full_speech.lower()}\""},
                                    {"role": "assistant", "content": full_speech}
                                ]
                            }
                        })
                        # Caso 2: Comprensión conversacional del tema del directo
                        if len(full_speech) > 100:
                            dataset_items.append({
                                "question": f"¿De qué hablaste en este tramo del directo?\n\"{full_speech}\"",
                                "answer": f"Estaba comentando sobre el tema: {full_speech[:150]}... explicando la lógica en caliente.",
                                "query": f"¿De qué hablaste en este tramo del directo?\n\"{full_speech}\"",
                                "response": f"Estaba comentando sobre el tema: {full_speech[:150]}... explicando la lógica en caliente.",
                                "source_label": "youtube-asr-comprehension",
                                "video_id": vid,
                                "openai": {
                                    "messages": [
                                        {"role": "system", "content": SYSTEM_PROMPT},
                                        {"role": "user", "content": f"¿De qué se habla en este fragmento del stream?\n\"{full_speech}\""},
                                        {"role": "assistant", "content": f"Ahí estaba explicando en vivo: {full_speech[:120]}."}
                                    ]
                                }
                            })
                chunk = []

            last_to_ms = to_ms

        if chunk:
            full_speech = " ".join(chunk).strip()
            if 30 <= len(full_speech) <= 650 and not contains_secret(full_speech).found:
                dataset_items.append({
                    "question": f"Transcribí el audio en español rioplatense:\n\"{full_speech.lower()}\"",
                    "answer": full_speech,
                    "query": f"Transcribí el audio en español rioplatense:\n\"{full_speech.lower()}\"",
                    "response": full_speech,
                    "source_label": "youtube-asr-transcription",
                    "video_id": vid,
                    "openai": {
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Transcribí el fragmento:\n\"{full_speech.lower()}\""},
                            {"role": "assistant", "content": full_speech}
                        ]
                    }
                })

    print(f"Total ejemplos generados desde transcripciones: {len(dataset_items)}")

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for it in dataset_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"Dataset de ASR guardado en: {out_jsonl} ({out_jsonl.stat().st_size / 1024**2:.2f} MB)")
    return len(dataset_items)

def main():
    parser = argparse.ArgumentParser(description="Ingesta y ASR de videos de YouTube con whisper.cpp")
    parser.add_argument("--process-only", action="store_true", help="Solo procesar transcripciones existentes en jsonl sin descargar")
    parser.add_argument("--limit", type=int, default=0, help="Limitar a N videos (0 = todos)")
    args = parser.parse_args()

    RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.process_only:
        if not URLS_FILE.exists():
            print(f"Error: {URLS_FILE} no existe")
            sys.exit(1)

        urls = [line.strip() for line in open(URLS_FILE) if line.strip() and not line.startswith("#")]
        if args.limit > 0:
            urls = urls[:args.limit]

        print(f"=== Iniciando ingesta de {len(urls)} videos/directos ===")
        for idx, url in enumerate(urls, 1):
            vid = extract_video_id(url)
            out_json = TRANSCRIPTS_DIR / f"{vid}.json"
            out_base = TRANSCRIPTS_DIR / vid

            print(f"\n[{idx}/{len(urls)}] Video ID: {vid} ({url})")
            if out_json.exists() and out_json.stat().st_size > 100:
                print(f"  -> Ya transcripto ({out_json.name}), omitiendo descarga.")
                continue

            wav_file = RAW_AUDIO_DIR / f"{vid}.wav"
            ok_dl = download_audio(url, vid, wav_file)
            if not ok_dl:
                print(f"  -> Omitiendo por error de descarga.")
                continue

            ok_trans = transcribe_whisper(wav_file, out_base)
            if ok_trans:
                print(f"  -> Transcripción completada: {out_json.name}")
            else:
                print(f"  -> Falló la transcripción.")

            if wav_file.exists():
                try:
                    os.remove(wav_file)
                    print(f"  -> Audio temporal {wav_file.name} eliminado.")
                except Exception as e:
                    print(f"  -> No se pudo eliminar wav: {e}")

            import time
            time.sleep(3)

    parse_transcripts_to_dataset(TRANSCRIPTS_DIR, OUT_DATASET)

if __name__ == "__main__":
    main()
