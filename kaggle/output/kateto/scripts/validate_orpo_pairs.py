#!/usr/bin/env python3
"""Validator for data/orpo/anti_sycophancy_rioplatense.jsonl (plan todo 9).

Checks each row is a valid {prompt, chosen, rejected} ORPO pair in Kateto
ChatML plaintext-delimiter form. Malformed rows are quarantined to a .bad
file; the build continues and counts are reported to out/orpo_pairs.log.

Exit 0 iff valid >= 200. Exit 1 otherwise.
"""
import json
import re
import sys
from pathlib import Path

SRC = Path("data/orpo/anti_sycophancy_rioplatense.jsonl")
BAD = Path("data/orpo/anti_sycophancy_rioplatense.jsonl.bad")
LOG = Path("out/orpo_pairs.log")
MIN_VALID = 200

VOSEO = re.compile(
    r"(che|mir[áa]|fijate|ten[ée]s|dale|and[áa]|contame|decime|hac[ée]|"
    r"pon[ée]|tom[áa]|dej[áa]|sab[ée]s|pod[ée]s|quer[ée]s|posta|boludo|"
    r"pibe|tranqui|qued[áa]s|pensalo|comprate)",
    re.IGNORECASE,
)
QWEN_TMPL = ("<|im_start|>user", "<|im_start|>assistant", "<|im_start|>system")


def check(row, lineno):
    """Return None if valid, else a reason string."""
    if not isinstance(row, dict):
        return f"line {lineno}: not a JSON object"
    for k in ("prompt", "chosen", "rejected"):
        if k not in row:
            return f"line {lineno}: missing key '{k}'"
        if not isinstance(row[k], str) or not row[k].strip():
            return f"line {lineno}: empty/non-string '{k}'"
    if row["chosen"].strip() == row["rejected"].strip():
        return f"line {lineno}: chosen == rejected"
    if "<|im_user|>" not in row["prompt"]:
        return f"line {lineno}: prompt lacks <|im_user|>"
    for k in ("chosen", "rejected"):
        v = row[k]
        if "<|im_start|>" not in v or "<|im_end|>" not in v:
            return f"line {lineno}: {k} lacks ChatML plaintext delimiters"
        if "seco" not in v:
            return f"line {lineno}: {k} lacks voice tag 'seco'"
        if any(t in v for t in QWEN_TMPL):
            return f"line {lineno}: {k} uses forbidden Qwen template"
    if not VOSEO.search(row["chosen"]):
        return f"line {lineno}: chosen lacks voseo marker"
    return None


def main():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    valid, bad = 0, []
    for i, raw in enumerate(lines, 1):
        if not raw.strip():
            bad.append((i, f"line {i}: blank line", raw))
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            bad.append((i, f"line {i}: bad JSON ({e})", raw))
            continue
        reason = check(row, i)
        if reason:
            bad.append((i, reason, raw))
        else:
            valid += 1
    BAD.write_text("".join(r + "\n" for _, _, r in bad), encoding="utf-8")
    report = (
        f"src={SRC} total={len(lines)} valid={valid} "
        f"malformed={len(bad)} quarantined={BAD}\n"
    )
    for _, reason, _ in bad:
        report += f"BAD: {reason}\n"
    report += "PASS\n" if valid >= MIN_VALID and not bad else (
        "FAIL: need >=200 valid and 0 malformed\n" if bad or valid < MIN_VALID else "FAIL\n"
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0 if (valid >= MIN_VALID and not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
