#!/usr/bin/env python3
"""Diversity gates for persona voice datasets (followup-A-diversify).

Gates per file: (a) zero exact-duplicate rows; (b) >=95% unique full
response strings; (c) >=40 distinct response OPENINGS (first 6 words);
(d) no single 8-gram in >5% of rows; (e) zero `tool_call` substrings.
Metrics -> out/diversity_report.log. Exit 0 iff all gates green.
System python3, stdlib only.
"""
import json
import os
import sys
from collections import Counter

FILES = {
    "data/rwkv_kateto_jane.jsonl": 500,
    "data/rwkv_kateto_doktor.jsonl": 550,
    "data/rwkv_kateto_whisperer.jsonl": 350,
    "data/rwkv_kateto_streamer.jsonl": 500,
}
VOICE = {"data/rwkv_kateto_jane.jsonl": "jane",
         "data/rwkv_kateto_doktor.jsonl": "doktor",
         "data/rwkv_kateto_whisperer.jsonl": "whisperer",
         "data/rwkv_kateto_streamer.jsonl": "streamer"}
REPORT = "out/diversity_report.log"


def split_row(text):
    """Return (prompt, voice, response) from ChatML plain-text envelope."""
    pre, _, post = text.partition("<|im_start|>")
    prompt = pre.replace("<|im_user|>", "").strip()
    voice_line, _, _ = post.partition("<|im_end|>")
    parts = voice_line.strip().split("\n", 1)
    voice = parts[0].strip()
    resp = parts[1].strip() if len(parts) > 1 else ""
    return prompt, voice, resp


def grams(words, n):
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def check(path, expected):
    errors, metrics = [], {}
    rows = [l for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
    metrics["rows"] = len(rows)
    if len(rows) != expected:
        errors.append(f"row count {len(rows)} != {expected}")
    bad_json = 0
    texts, prompts, resps, openings = [], [], [], []
    gram_rows = Counter()
    toolcall = 0
    for i, l in enumerate(rows):
        try:
            t = json.loads(l)["text"]
        except Exception:
            bad_json += 1
            continue
        if not isinstance(t, str) or "<|im_user|>" not in t or "<|im_start|>" not in t \
                or "<|im_end|>" not in t or "tool_call" in t:
            if "tool_call" in l:
                toolcall += 1
            if not isinstance(t, str):
                bad_json += 1
            continue
        if "tool_call" in t:
            toolcall += 1
        prompt, voice, resp = split_row(t)
        if voice != VOICE[path]:
            errors.append(f"row {i}: voice '{voice}' != '{VOICE[path]}'")
            break
        if not prompt or not resp:
            errors.append(f"row {i}: empty prompt/response")
            break
        texts.append(t)
        prompts.append(prompt)
        resps.append(resp)
        openings.append(" ".join(resp.lower().split()[:6]))
        seen = set(grams(resp.lower().split(), 8))
        for g in seen:
            gram_rows[g] += 1
    metrics["bad_json"] = bad_json
    if bad_json:
        errors.append(f"{bad_json} malformed rows")
    # (a) zero exact-duplicate rows
    dupes = len(texts) - len(set(texts))
    metrics["exact_dupe_rows"] = dupes
    if dupes:
        errors.append(f"(a) {dupes} exact-duplicate rows")
    # (b) >=95% unique responses
    ur = len(set(resps)) / max(len(resps), 1)
    metrics["uniq_responses"] = f"{len(set(resps))}/{len(resps)}={ur:.4f}"
    if ur < 0.95:
        errors.append(f"(b) unique-response ratio {ur:.4f} < 0.95")
    # (c) >=40 distinct openings
    metrics["uniq_openings"] = len(set(openings))
    metrics["uniq_prompts"] = len(set(prompts))
    if len(set(openings)) < 40:
        errors.append(f"(c) only {len(set(openings))} distinct openings < 40")
    # (d) no 8-gram in >5% of rows
    top = gram_rows.most_common(1)
    share = (top[0][1] / max(len(resps), 1)) if top else 0.0
    metrics["top_8gram_share"] = f"{share:.4f}" + (f" e.g. '{top[0][0][:80]}'" if top else "")
    metrics["top_8gram_rows"] = top[0][1] if top else 0
    if share > 0.05:
        errors.append(f"(d) top 8-gram in {share:.2%} of rows > 5%")
    # (e) zero tool_call
    metrics["tool_call_hits"] = toolcall
    if toolcall:
        errors.append(f"(e) {toolcall} tool_call hits")
    return errors, metrics


def main():
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    all_ok, lines = True, []
    for path, n in FILES.items():
        if not os.path.exists(path):
            print(f"MISSING {path}", file=sys.stderr)
            lines.append(f"{path}: MISSING FILE -> FAIL");
            all_ok = False
            continue
        errors, m = check(path, n)
        status = "GREEN" if not errors else "RED"
        if errors:
            all_ok = False
        lines.append(f"== {path} (expect {n}) -> {status}")
        for k, v in m.items():
            lines.append(f"  {k}: {v}")
        for e in errors:
            lines.append(f"  GATE-FAIL: {e}")
    report = "\n".join(lines) + "\n"
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(report, end="")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
