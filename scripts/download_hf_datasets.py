#!/usr/bin/env python3
"""download_hf_datasets.py — Fetch the 8 SPEC §3 HF datasets to data/hf/.

Full run writes data/hf/<slug>/ + data/hf/SOURCES.md (schema + license
per dataset). Gated/renamed repos are skipped with reason, rest continue.
Metadata probes are stdlib-only (urllib); full snapshot fetch prefers
huggingface_hub.snapshot_download when importable, else records SKIP with
the manual verification command. Never hardcodes HF tokens.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT / "data" / "hf"

# (hf_id, use, schema_note). License resolved live from the Hub API;
# UNKNOWN + verification command when unreachable (no hardcoding).
DATASETS = [
    ("marianbasti/cordeba", "train",
     "CordeBA oral Rioplatense; splits/fields via API probe; feeds Reddit AR + CordeBA 5k slice (SPEC §3)."),
    ("DataCreatorAI/Anti-Sycophancy-DPO", "train",
     "DPO pairs (prompt/chosen/rejected); feeds Anchors de Autonomia anti-sycophancy slice."),
    ("stindardlogic/sycophancy-reduction-dpo-100k", "train",
     "DPO pairs ~100k (prompt/chosen/rejected); feeds anti-sycophancy slice."),
    ("NickyNicky/function-calling-sharegpt_chatml_gemma_agent", "train",
     "ShareGPT ChatML tool-calling (messages/tools schema); feeds Tools & Debate syntax slice."),
    ("somosnlp-hackathon-2026/che-boludo-benchmark", "eval-only",
     "EVAL-ONLY, never train input. Rioplatense benchmark; eval split only."),
    ("iberbench/iberbench_all", "train",
     "IberBench multitask (HAHA subset: filter is_humorous=True, rewrite to rioplatense)."),
    ("latam-gpt/Trueque-Benchmark-beta-0.1", "eval-only",
     "EVAL-ONLY, never train input. Trueque benchmark; eval split only."),
    ("swzzzzc/Anti-Sycophancy-RepE-384", "train",
     "RepE steering vectors (384-dim); feeds anti-sycophancy slice."),
]

EVAL_ONLY = {d[0] for d in DATASETS if d[1] == "eval-only"}

API_URL = "https://huggingface.co/api/datasets/{}"


def slug(hf_id):
    return hf_id.split("/")[-1]


def probe(hf_id, timeout=20):
    """Metadata-only probe. Returns (ok, license, siblings_note / error)."""
    req = urllib.request.Request(API_URL.format(hf_id),
                                 headers={"User-Agent": "kateto-train/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            meta = json.loads(r.read().decode("utf-8"))
    except Exception as e:  # gated/renamed/offline -> skip with reason
        return False, "UNKNOWN", "{}: {}".format(type(e).__name__, e)
    tags = (meta.get("tags") or [])
    lic = ((meta.get("cardData") or {}).get("license")
           or meta.get("license") or "UNKNOWN")
    sibs = meta.get("siblings") or []
    names = [s.get("rfilename", "") for s in sibs[:8]]
    note = "files: {}; tags: {}".format(
        ", ".join(names) if names else "n/a", ", ".join(tags[:8]) or "n/a")
    if meta.get("gated") not in (None, False):
        return False, lic, "GATED: " + note
    return True, lic, note


def verify_cmd(hf_id):
    return ("curl -s https://huggingface.co/api/datasets/{} | "
            "python3 -c \"import json,sys; m=json.load(sys.stdin); "
            "print(m.get('cardData'), m.get('gated'), m.get('siblings'))\"".format(hf_id))


def fetch_dataset(hf_id, dest):
    """Best-effort snapshot fetch; returns (ok, reason)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False, ("SKIP (huggingface_hub missing): git clone "
                       "https://huggingface.co/datasets/{} {}".format(hf_id, dest))
    try:
        snapshot_download(repo_id=hf_id, repo_type="dataset",
                          local_dir=str(dest), max_workers=4)
        return True, "downloaded"
    except Exception as e:
        return False, "{}: {}".format(type(e).__name__, e)


def write_sources(out, rows):
    lines = ["# HF Sources — SPEC §3", "",
             "| HF id | SPEC use | license | schema note |",
             "|---|---|---|---|"]
    for hf_id, use, lic, schema in rows:
        flag = " **EVAL-ONLY — never train input**" if hf_id in EVAL_ONLY else ""
        lines.append("| {} | {}{} | {} | {} |".format(hf_id, use, flag, lic, schema))
    lines += ["",
              "## License verification",
              "Per dataset (no tokens hardcoded):",
              ""]
    for hf_id, _, lic, _ in rows:
        if lic == "UNKNOWN":
            lines.append("- `{}`: `{}`".format(hf_id, verify_cmd(hf_id)))
    (out / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--datasets", nargs="*", default=[d[0] for d in DATASETS])
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--probe-timeout", type=int, default=20)
    args = ap.parse_args()

    want = {d[0]: d for d in DATASETS}
    ids = args.datasets if args.datasets else [d[0] for d in DATASETS]
    if args.list_only:
        for hf_id in ids:
            spec = want.get(hf_id, (hf_id, "train", "ad-hoc (not in SPEC §3)"))
            print("{} | use={} | {}".format(spec[0], spec[1], spec[2]))
        return 0

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for hf_id in ids:
        spec = want.get(hf_id, (hf_id, "train", "ad-hoc"))
        ok, lic, note = probe(hf_id, timeout=args.probe_timeout)
        schema = "{} | probe: {}".format(spec[2], note)
        rows.append((hf_id, spec[1], lic, schema))
        dest = out / slug(hf_id)
        if not ok:
            print("SKIP {}: {} [{}]".format(hf_id, note, lic), file=sys.stderr)
            failed.append(hf_id)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        fok, reason = fetch_dataset(hf_id, dest)
        print("{} {}: {}".format("OK" if fok else "SKIP", hf_id, reason))
        if not fok:
            failed.append(hf_id)
    write_sources(out, rows)
    print("wrote {}".format(out / "SOURCES.md"))
    if failed:
        print("skipped {}: {}".format(len(failed), ", ".join(failed)), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
