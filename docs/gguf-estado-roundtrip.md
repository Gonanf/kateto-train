# B1 — State round-trip over GGUF

**Question:** can a GGUF model in llama.cpp accept a state buffer produced elsewhere?
The repo claimed it could not (`export_gguf.sh` header, `docs/decision-gguf-vs-nativo.md:20`).

**Answer, measured on this machine: yes.** The state buffer of `seq_id 0` can be written out,
restored into `seq_id 1`, and the restored sequence then behaves identically to the original —
byte-identical payload, identical logits, identical greedy continuation.

## How to run

```bash
bash script/probar-gguf-estado.sh            # CPU, the default
BACKEND=vulkan bash script/probar-gguf-estado.sh   # GPU (not used for this result)
```

It compiles `tools/gguf_state_roundtrip.cpp` against the system llama.cpp, prints the llama.cpp
version, and runs the three tests. Exit code: 0 = PASS, 1 = FAIL, 2 = a llama.cpp API call failed.

## What it does NOT prove

- **Nothing about `.pth` states.** The buffer here was produced by llama.cpp itself. Whether a
  PyTorch RWKV state can be serialized into this layout is a separate experiment.
- **Nothing about ROSA.** ROSA was never involved in this test.
- **Nothing about speed.** The round-trip is 21 MB in and out. Whether swapping states is fast
  enough at inference time was not measured.
- **Nothing about the native path.** Only the llama.cpp/GGUF path was exercised.
- **Only one prompt.** 7 tokens, one sequence pair, greedy sampling, Q4_K_M quantization.

## Environment

- llama.cpp `0.1.2-dev (build 10502, commit 8ff87cc)`, distro package `llama.cpp-vulkan b10502-1`
- Model `out/kateto-rwkv29-Q4_K_M.gguf`, RWKV-7 (2.9B), 4 GB VRAM budget, `n_gpu_layers=0`
- Context: `n_ctx=128`, `n_batch=128`, `n_seq_max=2`, `n_threads=8`
- State size: **21,627,676 bytes** (`size_bytes`, ~20.6 MB)
- Backend: CPU only. The GPU was occupied by another job, so Vulkan was not used.

## Reading the output

| Line | Meaning |
|---|---|
| `size_bytes` | Bytes `llama_state_seq_get_size` reports for one sequence state. |
| `state_seq_set_data_bytes` | Bytes `llama_state_seq_set_data` consumed. Non-zero = the restore was accepted. |
| `header_seq_id_0` / `header_seq_id_1` | The `llama_seq_id` stored in the first 8 bytes of each dump. These differ by design: `set_data` reads the header but restores into the `seq_id` you pass as the last argument. |
| `test1_identidad` | Byte comparison of the payload after those 8 header bytes. `OK` = the state survived the round-trip bit-for-bit. |
| `max_abs_diff` | Largest absolute difference between the logits vector of the original and restored sequences, over the whole vocabulary. |
| `argmax_0` / `argmax_1` / `argmax_igual` | Which token each sequence considers most likely. |
| `test2_logits` | PASS when `max_abs_diff <= 1e-3` and the argmax agree. |
| `continuacion_seq0` / `continuacion_seq1` | 8 greedy tokens generated from each sequence. |
| `test3_continuacion` | PASS when the two strings match exactly. |
| `RESULTADO` | `PASS` only if all three tests pass. |

## Result

`RESULTADO: PASS`. All three tests pass:

- `test1_identidad: OK` — 21,627,668 payload bytes identical.
- `test2_logits: OK` — `max_abs_diff: 0`, same argmax token.
- `test3_continuacion: OK` — both sequences generate the same 8 tokens.

Two artifacts of the test harness, not of llama.cpp, are worth knowing when reading a future FAIL:
the 8-byte header carries the `seq_id` and therefore always differs between a seq-0 and a seq-1
dump, and `llama_get_logits_ith` returns a pointer into the context's internal buffer, so the logits
vectors must be copied before the next `llama_decode` overwrites them.