#!/usr/bin/env bash
set -euo pipefail

# 1. Run analyze_fork.py
output=$(python3 scripts/eureka_fork_llamacpp/analyze_fork.py --dir out/eureka-fork-llamacpp)
status_analyze=0

# 2 & 3. Verify JSON parsed from last line, assert 23 exact keys, assert gain_pin_device > 5
last_line=$(echo "$output" | tail -n 1)

eval_result=$(python3 -c '
import json, sys

expected_keys = {
    "gain_pin_device", "gain_quant_q4_vs_q8", "gain_fa", "gain_ub1024",
    "gain_ub128", "gain_threads_t4", "gain_kvquant", "gain_pp_ub1024", "gain_pp_ub128",
    "rwkv_vs_transformer_amd", "rwkv_vs_transformer_intel", "rwkv_vs_transformer_cpu",
    "techo_amd_rwkv29_q4_ts", "techo_amd_rwkv29_q8_ts", "headroom_kernel_pct", "tg_c1", "tg_c5",
    "tg_c6", "gbs_c1", "gbs_c5", "gbs_c6", "gbs_c12", "gbs_c8"
}

line = sys.argv[1]
data = json.loads(line)
keys = set(data.keys())
assert keys == expected_keys, f"Keys mismatch: diff={keys ^ expected_keys}"
gain_pin = data["gain_pin_device"]
assert gain_pin > 5, f"gain_pin_device {gain_pin} <= 5"
print(f"{len(keys)} {gain_pin}")
' "$last_line")

keys_count=$(echo "$eval_result" | awk '{print $1}')
gain_pin_val=$(echo "$eval_result" | awk '{print $2}')

# 4. Check MEDICION.md exists and size > 1500
medicion_file="out/eureka-fork-llamacpp/MEDICION.md"
if [[ ! -f "$medicion_file" ]]; then
  echo "Error: $medicion_file does not exist" >&2
  exit 1
fi

medicion_size=$(stat -c%s "$medicion_file")
if (( medicion_size <= 1500 )); then
  echo "Error: $medicion_file size $medicion_size <= 1500" >&2
  exit 1
fi

# 5. Check run_sweep.sh --dry-run prints 14 lines, all containing llama-bench
dry_run_out=$(bash scripts/eureka_fork_llamacpp/run_sweep.sh --dry-run)
dry_run_lines=$(echo "$dry_run_out" | wc -l)
dry_run_matching=$(echo "$dry_run_out" | grep -c "llama-bench" || true)

if (( dry_run_lines != 14 || dry_run_matching != 14 )); then
  echo "Error: dry-run expected 14 llama-bench lines, got $dry_run_lines lines ($dry_run_matching matching)" >&2
  exit 1
fi

# 6. Check table in MEDICION.md has no % techo > 100
python3 -c '
with open("out/eureka-fork-llamacpp/MEDICION.md", "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("##"):
            break
        if line.startswith("|"):
            cols = [p.strip() for p in line.split("|")[1:-1]]
            if not cols or cols[-1] == "% techo":
                continue
            last = cols[-1].replace("*", "").strip()
            if last.endswith("%"):
                val = float(last[:-1].strip())
                assert val <= 100.0, f"techo > 100%: {val} in line: {line}"
'

echo "VERIFY OK"
echo "1. analyze_fork exit code: $status_analyze"
echo "2. json keys: $keys_count exact keys"
echo "3. gain_pin_device: $gain_pin_val (> 5)"
echo "4. MEDICION.md size: $medicion_size bytes (> 1500)"
echo "5. dry-run lines with llama-bench: $dry_run_matching / $dry_run_lines"
echo "6. tabla sin techo > 100%: OK"
