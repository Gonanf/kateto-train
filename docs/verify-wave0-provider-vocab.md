# Wave 0 Todo 1 — Provider target + ChatML vocab proof (2026-09-06)

## 1. Provider path

Command: `ls /run/media/chaos/terciario/proyectos/Kateto/kateto/providers/rwkv_rocm.py`

Result: **EXISTS** — exact stdout:

```
/run/media/chaos/terciario/proyectos/Kateto/kateto/providers/rwkv_rocm.py
```

Note: sibling repo (`Kateto`), NOT this repo. Per plan scope, C1 inference is built
in `rwkv_pipeline/infer_kateto.py` here either way; the sibling file is reference only.

## 2. Vocab probe

Commands:

```bash
python3 -c "t=open('RWKV-PEFT/rwkv_vocab_v20230424.txt',encoding='utf-8').read().splitlines(); [print(s) for s in ['<|im_user|>','<|im_start|>','<|im_end|>'] if s in t or print(s+' MISSING')]" > out/verify_vocab.txt
grep -c 'im_' RWKV-PEFT/rwkv_vocab_v20230424.txt
```

Results (`out/verify_vocab.txt`, verified via `cat`):

```
<|im_user|> MISSING
<|im_start|> MISSING
<|im_end|> MISSING
```

`grep -c 'im_'` → `0` (exit 1, no matches). SPEC §2.4:262 claim
("tokens … ya existen en el vocab, verificado") is **refuted** for this vocab file.

## 3. Current prompt format (code as-is, not modified)

- `rwkv_pipeline/infer_kateto.py:256` — `formatted = f"User: {prompt.strip()}\n\nAssistant:"`
- `rwkv_pipeline/prepare_datasets.py:29-32` — `format_rwkv_chat()` returns `f"User: {q}\n\nAssistant: {r}"`
- SPEC envelope (§2.4:248-262, §4.2:422-435) requires
  `<|im_user|>{q}\n<|im_start|>{voice}\n{r}\n<|im_end|>` + trailing open `<|im_start|>{voice}` —
  **not emittable as single tokens** with the current vocab.

## 4. Verdict: BLOCKED (unambiguous)

ChatML single-token path is **BLOCKED** — all 3 ids MISSING, `grep -c im_` = 0.

- Wave 1 data todos (5/6/9) MUST NOT assume ChatML ids exist; they emit via the
  converter path of todo 10, which honors this verdict (multi-char sequences or
  vocab extension decided there, not here).
- Todo 10 validity gate (`scripts/convert_to_kateto_format.py --check`) must treat
  `<|im_*|>` as plain-text delimiters (multi-token) until a vocab-extension todo exists.
- Loss-mask work (todo 8) proceeds on `User:/Assistant:` spans only.

## 5. Decision: C1 inference target

Build C1 inference fixes in `rwkv_pipeline/infer_kateto.py` **either way**
(plaintext `User:/Assistant:` envelope retained until todo 10 resolves the encoding).
No change to sibling `rwkv_rocm.py`; no Qwen-template reuse in Kateto rows.
