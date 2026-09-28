# Wave 0 — PiSSA wiring path + canonical model pin (todo 3)

Date: 2026-09-06. Read-only probe; no product file modified.

## 1. What `--pissa_config '{"pissa_r":32,"svd_niter":4}'` does today: PARSED, then SILENTLY IGNORED

**Parsed (accepts the flag).** `RWKV-PEFT/train.py:98` declares it with `type=json.loads`,
so argparse converts the string to a dict without error (replicated exact declaration):

```
parsed OK: {'pissa_r': 32, 'svd_niter': 4}
```

**Never consumed.** Repo-wide grep for `pissa|Pissa|PISSA` (`*.py,*.sh,*.md`, excl. `.omo/`, venvs)
returns only: the `train.py:80` comment (`# lora pissa DiSHA`), the `train.py:98` declaration,
the `rwkvt/args_type.py:63-64` dataclass default, README marketing lines, and `SPEC.md`
future-plan references. **Zero code reads `args.pissa_config`.**

**`--peft pissa` ERRORS.** `rwkvt/peft_loading.py:76-82` dispatches through an exact dict:

```python
peft_dict={
    "lora": LoraConfig,
    "miss": MissConfig,
    "adalora": AdaLoraConfig,
    "prefix": PrefixTuningConfig,
}
ConfigClass = peft_dict[args.peft]
```

Exact stderr with the repo's own classes (`venv-unsloth-qwen`, peft 0.20.0):

```
keys: ['adalora', 'lora', 'miss', 'prefix']
KeyError: 'pissa'
```

So: `--pissa_config` alone = silently ignored; `--peft pissa` = `KeyError: 'pissa'`.
Prior finding confirmed independently: `peft_dict` keys are exactly
**lora / miss / adalora / prefix** — no pissa branch, no state-vs-lora nuance beyond the
existing `state` env-var branch (`peft_loading.py:47-55`).

## 2. `peft` availability

- System `python3`: `ModuleNotFoundError: No module named 'peft'` (also no `torch`). Prior
  observation confirmed.
- `venv-unsloth-qwen/bin/python`: **peft 0.20.0**, torch 2.9.1+rocm6.3. Installed `peft`
  exposes `MissConfig`/`MissModel` but **no `PissaConfig`** (`dir(peft)` check). PiSSA cannot
  come from the installed dependency — todo 13 must implement it natively (or vendor it).

## 3. Canonical checkpoint pin for todos 7–18

**HF `aabbdev/RWKV7-2.3B-20260805`: NOT REACHABLE.** Exact result:

```
huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error.
Repository Not Found for url: https://huggingface.co/api/models/aabbdev/RWKV7-2.3B-20260805.
```

No download attempted; API metadata call only.

**PINNED (on-disk, verified by header read via mmap, no full load):**

| checkpoint | sha256 | n_layer | n_embd |
|---|---|---|---|
| `base/rwkv7-g1j-2.9b-20260831-ctx16384.pth` (**PINNED for todos 7–18**) | `966f3420f833532aae3fb1fd6326533b08d43d23b7b03eaa2f0694a30b64a239` | 32 | 2560 |
| `models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth` (fallback) | `947cb9b8013224e06b112b72204256bec65096cc935a7767ce63d8e3ddef83bb` | 24 | 1024 |

Dims derived from state-dict keys (`max(blocks.N)+1`, `ln_out.weight.shape[0]`); both match
the `rwkv_pipeline/train_lora_base.sh:19-40` arch table (2.9B→32L/2560, 0.4B→24L/1024).
Launch flags live at `train_lora_base.sh:58-74` (`--peft lora --peft_config '{"r":16,…}'`).

Decision: HF 2.3B 404s → **pin on-disk 2.9B (32L/2560, sha `966f3420…`)** for todos 7–18;
0.4B (24L/1024, sha `947cb9b8…`) stays the cheap-smoke fallback.

## 4. LoRA dry-run (trainable-param count): SKIPPED

`load_peft_model` instantiates `RWKVModel` twice and `torch.load`s the full 5.9 GB
checkpoint into RAM before `print_trainable_parameters()` — not cheap/safe as a probe on
this box. Skipped deliberately; param count will be observed naturally on the first real
todo-7+ training launch.

## 5. Consequence for todo 13

Native PiSSA dispatch required: add a `pissa` branch in `peft_loading.py` (SVD init,
`pissa_r`/`svd_niter` from the currently-dead `pissa_config`) — the LoRA-r16 fallback in
`train_lora_base.sh:72-73` stays valid until then.
