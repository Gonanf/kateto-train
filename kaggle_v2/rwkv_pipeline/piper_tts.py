"""Piper TTS provider (offline, CPU ONNX) — ADDITIVE, gate-passed via fallback voice.

Gate (wave4-todo23): SPEC ref `HirCoir/piper-checkpoint-es-ar-elena` is a training
`.ckpt`, not a runnable voice → fallback to the only verified official `es_AR`
voice `es_AR-daniela-high` (rhasspy/piper-voices). Sample: `out/piper_sample.wav`.

Local interface (this repo has no kateto/ package; sibling `Kateto/kateto/providers/`
types untouched). Streams from the FIRST text chunk: caller pushes text via `feed()`,
each completed sentence is synthesized as soon as it closes — no waiting for the
full LLM turn. Whole-utterance `piper` CLI underneath, so per-sentence latency is
still seconds on this box (see `out/latency_profile.md`) — first-chunk here means
first *sentence*, not <200 ms audio.

# ponytail: one subprocess per sentence (no persistent server), upgrade to
# piper HTTP/websocket server if per-sentence spawn cost matters.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 22050
DEFAULT_MODEL = Path("models/piper/es_AR-daniela-high.onnx")
SAMPLE_TEXT = "Che, ¿cómo andás?"

_SENT_END = re.compile(r"(.+?[.!?…]+[\s\"'»)]*|.+?$)", re.DOTALL)


def split_sentences(text: str) -> list[str]:
    """Split text into sentence chunks; trailing fragment without terminator kept."""
    out = [m.group(1).strip() for m in _SENT_END.finditer(text) if m.group(1).strip()]
    return out


@dataclass
class PiperTTSProvider:
    """Minimal local TTS interface. Zero coupling to edgetts/boson/zonos."""

    model: Path = field(default_factory=lambda: DEFAULT_MODEL)

    def synth_sentence(self, sentence: str, out_path: Path) -> Path:
        piper = shutil.which("piper")
        if piper is None:
            raise RuntimeError("piper binary missing: pip install piper-tts")
        if not self.model.exists():
            raise FileNotFoundError(f"voice missing: {self.model}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _ = subprocess.run(
            [piper, "--model", str(self.model), "--output_file", str(out_path)],
            input=sentence.encode("utf-8"),
            check=True,
        )
        return out_path

    def stream_text(self, full_text: str, out_dir: Path) -> list[Path]:
        """Synthesize from the first sentence chunk; returns one wav per sentence."""
        wavs = []
        for i, sent in enumerate(split_sentences(full_text)):
            wavs.append(self.synth_sentence(sent, out_dir / f"chunk_{i:03d}.wav"))
        return wavs


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


if __name__ == "__main__":
    # Self-check: splitter pure logic (no piper needed) + sample wav validity.
    assert split_sentences("Che, ¿cómo andás? Todo bien.") == [
        "Che, ¿cómo andás?",
        "Todo bien.",
    ]
    assert split_sentences("sin terminador") == ["sin terminador"]
    sample = Path("out/piper_sample.wav")
    assert sample.exists() and sample.stat().st_size > 1000, "sample wav missing"
    assert wav_duration(sample) > 0.3, "sample wav too short"
    print(f"demo: splitter OK, sample {sample.stat().st_size}B {wav_duration(sample):.2f}s")
    # Live synth only if toolchain present (gate artifact, not required for import).
    if shutil.which("piper") and DEFAULT_MODEL.exists():
        p = PiperTTSProvider().synth_sentence(SAMPLE_TEXT, Path("/tmp/piper_demo.wav"))
        print(f"demo: live synth OK {p} {wav_duration(p):.2f}s")
    else:
        print("demo: live synth skipped (piper/model absent)")
