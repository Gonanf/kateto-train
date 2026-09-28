# Kaggle joint-session checklist (2.9B 2xT4 pre-flight)

Prep-only doc for `notebooks/kateto_train_kaggle_2xt4.ipynb` (9 cells, validate-only, no GPU yet).
Source budgets: `docs/kaggle-dryrun.md`; state epochs: `config/state_tuning.yaml`.

## 1. Quota check (do first)

- Free tier ~30h/week T4 (~40h verified). SFT 5ep ~= 7-10h (3ep=4-6h -> per-ep 1.33-2h x5).
- SFT-only total: 7-10h + states ~2h = **9-12h**; one full retry = **18-24h < 30h -> FITS**.
- If ORPO re-spike flips to GO: +2-3h = 11-15h; retry = 22-30h -> fits but **tight**, retry may need a second week/account.
- Action: open Kaggle Settings -> Usage, confirm >= 15h remaining (25h+ if ORPO-GO is possible).

## 2. Secrets needed (nothing stored in repo)

- Kaggle API token (`kaggle.json`) attached as Kaggle Secret, never pasted into cells.
- HuggingFace read access (public `BlinkDL/rwkv7-g1` — no token normally needed).
- **ONE input needed from user**: private dataset slug for Option B staging.
  Cell 3 uses placeholder `TODO-owner/kateto-data` — owner replaces it live at session time.

## 3. Dataset upload order (Option A first, fall back down)

1. Option A: `git clone https://github.com/Gonanf/kateto-train` — carries data+config if pushed.
2. Option B: `kaggle datasets download -d TODO-owner/kateto-data -p /kaggle/working --unzip`, then copy into `/kaggle/working/kateto-train/`.
3. Option C: manual UI upload per file.
- Verify paths after staging (must match training cells):
  `data/rwkv_kateto_seco.jsonl` (530), `data/rwkv_kateto_streamer.jsonl` (500),
  `data/rwkv_kateto_jane.jsonl` (500), `data/rwkv_kateto_doktor.jsonl` (550),
  `data/rwkv_kateto_whisperer.jsonl` (350), `data/debate_speech.jsonl` (800),
  `data/rwkv_kateto_base_train.jsonl` (20506), `data/orpo/anti_sycophancy_rioplatense.jsonl` (208),
  `config/state_tuning.yaml`, `config/pissa_2.3b.yaml`, `config/orpo.yaml`, `docs/orpo-spike.md`.
- State-tune inputs = DIVERSIFIED voice sets (jane/doktor/whisperer/streamer rewritten, uniqueness-gated).
- OPEN DECISION (do not resolve here): mixer still uses the older Capa-1 mix for base training unless owner says otherwise — diversified sets are state-tune inputs only.

## 4. Cell run order

1. Cell 1 setup (NCCL flags, installs, `nvidia-smi` 2x16GB check).
2. Cell 2 model pin (`BlinkDL/rwkv7-g1`, sha `966f3420…`); 7B stays FENCED.
3. Cell 3 staging + converter + NO_RESPONSE [3,4]% gate.
4. Cell 4 no-pack confirm (`DATA_TYPE=jsonl`).
5. Cell 5 PiSSA r32/niter4 (LoRA-r16 fallback path noted).
6. Cell 6 SFT 5 epochs (first-epoch `it/s` = NCCL proof).
7. Cell 7 ORPO: default SKIP (`ORPO_RESPIKE` NO-GO, re-spike 1/8); run only on fresh GO + `out/orpo_spike.log`.
8. Cell 8 state tuning (doktor 15, jane 12, whisperer 10, streamer 15, seco 5).
9. Cell 9 eval gates + export.

## 5. Export step

- `mkdir -p /kaggle/working/kateto-ckpt && cp /kaggle/working/out/rwkv_kateto_base/*.pth /kaggle/working/kateto-ckpt/`
- Kaggle UI -> New Dataset from `kateto-ckpt` folder (or `kaggle datasets create -p ...`).
- Also download `pissa_fallback_reason.txt` (if present) + eval logs.

## 6. Rollback

- SFT OOM/fail: retry same cell with `--peft lora --peft_config '{"r":16,...}'` fallback line (cell 6 comment).
- ORPO OOM: skip (SFT-only is the accepted path; NO-GO is the current verdict).
- Quota exhausted mid-run: stop, export partial `out/` as dataset, resume next quota window — do not re-run cell 1 installs blindly.
- Bad mix (NO_RESPONSE out of band): build exits nonzero by design — fix source data, do not clamp.
