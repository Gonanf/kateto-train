#!/usr/bin/env python3
"""Old `User:/Assistant:` <-> ChatML converter + validity gate (todo 10).

`<|im_user|>`, `<|im_start|>`, `<|im_end|>` are PLAIN-TEXT multi-token
delimiters everywhere (todo 1 verdict: single-token ids MISSING from
`rwkv_vocab_v20230424.txt`). Validity regexes below match literal text,
never token ids.

Canonical envelope (voice defaults to `seco`):
    <|im_user|>{q}\\n<|im_start|>{voice}\\n{r}\\n<|im_end|>

Row shapes handled (single chokepoint): mixer `{"text"}`, flat
`query/response`, debate `conversations human/gpt`, ORPO
`prompt/chosen/rejected`.

Usage:
    python scripts/convert_to_kateto_format.py --check   # validity gate
    ... < in.jsonl > out.jsonl                            # to ChatML (default)
    ... --reverse < in.jsonl > out.jsonl                  # back to legacy
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT_LOG = ROOT / "out" / "chatml_validity.log"

IM_USER = "<|im_user|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VOICES = ("seco", "streamer", "jane", "doktor", "whisperer")
DEFAULT_VOICE = "seco"

OLD_RE = re.compile(r"^User:\s*(.*?)\s*\n\nAssistant:\s*(.*)$", re.S)
NEW_RE = re.compile(
    r"^<\|im_user\|>(.*?)\n<\|im_start\|>(.*?)\n(.*)\n<\|im_end\|>\s*$", re.S
)
FULL_ENVELOPE_RE = NEW_RE  # alias: validity = literal-text match
HUMAN_RE = re.compile(r"^<\|im_user\|>(.+)$", re.S)
ASSIST_RE = re.compile(r"^<\|im_start\|>(.*?)\n(.*)\n<\|im_end\|>\s*$", re.S)


def old_to_new(text: str, voice: str = DEFAULT_VOICE) -> str:
    """Legacy `User:/Assistant:` row -> ChatML envelope; new rows pass through."""
    t = text.strip()
    if FULL_ENVELOPE_RE.match(t):
        return t  # already new: normalize by accepting as-is
    m = OLD_RE.match(t)
    if not m:
        raise ValueError(f"unrecognized row shape: {t[:80]!r}")
    q, r = m.group(1).strip(), m.group(2).strip()
    return f"{IM_USER}{q}\n{IM_START}{voice}\n{r}\n{IM_END}"


def new_to_old(text: str):
    """ChatML envelope -> (legacy `User:/Assistant:` row, voice)."""
    m = NEW_RE.match(text.strip())
    if not m:
        raise ValueError(f"not a ChatML envelope: {text[:80]!r}")
    q, voice, r = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    return f"User: {q}\n\nAssistant: {r}", voice


def convert_row(obj: dict[str, Any], reverse: bool = False) -> dict[str, Any]:
    """Convert one row of any known shape; unknown keys pass through."""
    obj = dict(obj)
    if "text" in obj and isinstance(obj["text"], str):
        obj["text"] = (
            new_to_old(obj["text"])[0] if reverse else old_to_new(obj["text"])
        )
    elif "query" in obj or "question" in obj:
        q = obj.get("query") or obj.get("question") or ""
        r = obj.get("response") or obj.get("answer") or ""
        if q and r:
            if reverse:
                obj["text"] = new_to_old(old_to_new(f"User: {q}\n\nAssistant: {r}"))[0]
            else:
                obj["text"] = old_to_new(f"User: {q.strip()}\n\nAssistant: {r.strip()}")
    elif isinstance(obj.get("conversations"), list):
        convs = obj["conversations"]
        for c in convs:
            if c.get("from") == "human" and not reverse:
                v = c.get("value", "")
                if not HUMAN_RE.match(v.strip()):
                    c["value"] = f"{IM_USER}{v.strip()}"
            elif c.get("from") == "gpt" and not reverse:
                v = c.get("value", "").strip()
                if not ASSIST_RE.match(v):
                    vm = re.match(r"^(seco|streamer|jane|doktor|whisperer)\s*\n?", v)
                    voice = vm.group(1) if vm else DEFAULT_VOICE
                    v = re.sub(r"^(seco|streamer|jane|doktor|whisperer)\s*\n?", "", v)
                    c["value"] = f"{IM_START}{voice}\n{v}\n{IM_END}"
    for key in ("prompt", "chosen", "rejected"):
        if not isinstance(obj.get(key), str):
            continue
        obj[key] = (
            _orpo_to_old(key, obj[key]) if reverse else _orpo_to_new(key, obj[key])
        )
    return obj


def _orpo_to_new(key: str, val: str) -> str:
    v = val.strip()
    if key == "prompt":
        return v if HUMAN_RE.match(v) else f"{IM_USER}{v}"
    if ASSIST_RE.match(v):
        return v
    return f"{IM_START}{DEFAULT_VOICE}\n{v}\n{IM_END}"


def _orpo_to_old(key: str, val: str) -> str:
    v = val.strip()
    if key == "prompt":
        m = HUMAN_RE.match(v)
        return m.group(1).strip() if m else v
    m = ASSIST_RE.match(v)
    return f"Assistant: {m.group(2).strip()}" if m else v


def check() -> int:
    """Gate: 200 old rows round-trip old->new->old identical + shape validity."""
    lines = []
    mismatches = 0
    checked = 0

    base = ROOT / "data" / "rwkv_kateto_base_train.jsonl"
    n = 0
    with open(base, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or n >= 200:
                break
            t = json.loads(line).get("text", "")
            if not t.startswith("User:"):
                continue
            n += 1
            try:
                back, _ = new_to_old(old_to_new(t))
                if back != t:
                    mismatches += 1
                    lines.append(f"MISMATCH row {n}: {t[:60]!r}")
            except ValueError as e:
                mismatches += 1
                lines.append(f"ERROR row {n}: {e}")
    checked += n
    lines.append(f"legacy round-trip: {n}/200 rows, mismatches={mismatches}")

    for path, kind in [
        ("data/debate_speech.jsonl", "debate"),
        ("data/orpo/anti_sycophancy_rioplatense.jsonl", "orpo"),
    ]:
        p = ROOT / path
        if not p.exists():
            lines.append(f"{kind}: {path} absent, skipped")
            continue
        bad, total = 0, 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                total += 1
                obj = json.loads(line)
                try:
                    if kind == "debate":
                        for c in obj["conversations"]:
                            v = c.get("value", "").strip()
                            ok = HUMAN_RE.match(v) if c.get("from") == "human" else ASSIST_RE.match(v)
                            if not ok:
                                bad += 1
                    else:
                        if not HUMAN_RE.match(obj["prompt"].strip()):
                            bad += 1
                        for k in ("chosen", "rejected"):
                            if not ASSIST_RE.match(obj[k].strip()):
                                bad += 1
                except (KeyError, AttributeError):
                    bad += 1
        mismatches += bad
        checked += total
        lines.append(f"{kind}: {total} rows, bad={bad}")

    lines.append(f"TOTAL checked={checked} mismatches={mismatches}")
    OUT_LOG.parent.mkdir(exist_ok=True)
    OUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if mismatches == 0 else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Kateto legacy <-> ChatML converter")
    ap.add_argument("--check", action="store_true", help="validity gate over 200 rows")
    ap.add_argument("--reverse", action="store_true", help="ChatML -> legacy")
    ap.add_argument("infile", nargs="?", help="input jsonl (default stdin)")
    ap.add_argument("outfile", nargs="?", help="output jsonl (default stdout)")
    args = ap.parse_args()
    if args.check:
        sys.exit(check())
    fin = open(args.infile, encoding="utf-8") if args.infile else sys.stdin
    fout = open(args.outfile, "w", encoding="utf-8") if args.outfile else sys.stdout
    with fin, fout if args.outfile else _noop():
        for line in fin:
            if not line.strip():
                continue
            fout.write(json.dumps(convert_row(json.loads(line), args.reverse), ensure_ascii=False) + "\n")


class _noop:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
