# B3: el `.pth` nativo de estado convertido en el byte-stream de llama.cpp

Base: `docs/gguf-estado-roundtrip.md` (B1) y `docs/gguf-estado-python.md` (B2). El
formato del buffer ya esta probado; lo que faltaba era que los tensores salieran de
un estado propio y no del dump de llama.cpp.

## El mapeo, campo por campo

El `.pth` de estado nativo (`train_state_voice.sh`, `peft: state`) trae una sola cosa
por capa:

| `.pth` | forma | -> | buffer de llama.cpp | quien lo prueba |
|---|---|---|---|---|
| `blocks.{i}.att.time_state` | `(n_head, head_size, head_size)` f4 | -> | tensor **s** de la capa `i`, una fila por celda de `n_embd_s` floats | `state_write_data` escribe `s_size_row = ggml_row_size(s_l->type, hparams.n_embd_s())` en `llama-memory-recurrent.cpp:948-949`; el `s_l` es el wkv state en `models/rwkv7-base.cpp:105,113-114` |
| (nada) | | -> | tensor **r** de la capa `i` (token shift, `token_shift_count * n_embd` floats) | `llama-memory-recurrent.cpp:915-916` + `llama-hparams.cpp:208-211`; `build_rwkv_token_shift_store` en `llama-graph.cpp:3566-3582`. **El `.pth` no lo trae**: queda en ceros |
| (nada) | | -> | tensor **p** | `llama-memory-recurrent.cpp:928-933`. rwkv7 no tiene p: `p_l[il] == nullptr` y no se escribe |

Las formas salen de la metadata del GGUF (`general.architecture = rwkv7`):

```
rwkv7.block_count       = 32    -> n_layer
rwkv7.embedding_length  = 2560  -> n_embd
rwkv7.wkv.head_size     = 64    -> wkv_head_size  -> n_head = 2560/64 = 40
token_shift_count = 2  (llama_hparams.h:143; el GGUF no lo escribe)
n_embd_r = token_shift_count * n_embd = 5120     (llama-hparams.cpp:208-211)
n_embd_s = n_embd * wkv_head_size      = 163840   (llama-hparams.cpp:236-239)
```

El orden de bytes del `s` es head-major en C: aplanar
`(n_head, head_size, head_size)` a `(n_embd, head_size)` es exactamente la fila que
`ggml_row_size` mide, asi que el reshape es gratis y no hay transposicion.

## Veredicto de shapes por modelo

`out/rwkv_states/seco/rwkv-3.pth`: **24 tensores** `blocks.{i}.att.time_state` de
**`(16, 64, 64)`** -> n_head=16, n_embd=1024. Viene de
`out/rwkv_kateto_base_plus_0.4b/rwkv-1.pth` (`n_layer: 24, n_embd: 1024,
head_size_a: 64` en `out/rwkv_states/seco/train_log.txt`).

| GGUF | n_layer | n_embd | head | `s` por celda | veredicto |
|---|---|---|---|---|---|
| `out/kateto-rwkv29-Q4_K_M.gguf` | 32 | 2560 | 64 | `(2560, 64)` | **no calza**: 24 vs 32 capas, n_embd 1024 vs 2560 (0/32 capas) |
| `out/base-rwkv29-f16.gguf` | 32 | 2560 | 64 | `(2560, 64)` | **no calza**: mismo modelo que el anterior |

Los dos GGUFs son el mismo RWKV-7 2.9B, asi que el veredicto es el mismo. El `.pth`
es de un 0.4B: no hay forma de que calce sin fabricar tensores, y un `.pth` de estado
de otro modelo no se arregla inventando tensores, se reporta. El camino de
fabricacion (`estado_a_bytes`) se niega con `ValueError` cuando no calza.

## Como se corre

```bash
bash script/probar-estado-pth.sh                    # report + veredicto
MODELOS="out/kateto-rwkv29-Q4_K_M.gguf" bash script/probar-estado-pth.sh
env -u PYTHONPATH -u VIRTUAL_ENV ./venv/bin/python -m rwkv_pipeline.state_from_pth \
    --pth out/rwkv_states/seco/rwkv-3.pth --modelo out/kateto-rwkv29-Q4_K_M.gguf --report
```

Termina en `RESULTADO B3: SHAPES_NO_CALZAN` (exit 3) con el `.pth` de voz actual. Con
un `.pth` de las formas correctas haria lo otro: arma el buffer, lo inyecta con
`tools/gguf_state_roundtrip --inject-state` y exige `state_seq_set_data_bytes != 0` y
`test1_identidad: OK` -> `RESULTADO B3: PASS`.

Tests: `env -u PYTHONPATH -u VIRTUAL_ENV ./venv/bin/python -m pytest
tests/test_state_from_pth.py -q` -> `8 passed`. Los del camino de fabricacion usan un
`.pth` sintetico con las formas del modelo y comparan contra el dump real de B1.

## Que NO prueba

- **La semantica.** Que el modelo con ese estado conteste como el camino nativo
  **no** esta medido. Que el layout calce y que `llama_state_seq_set_data` lo acepte
  y lo devuelva byte a byte dice nada sobre si el estado significa lo mismo. Medirlo
  exige comparar la generacion del `.pth` nativo contra la del GGUF con el estado
  inyectado, token a token.
- **El token shift.** El tensor `r` va en ceros porque el `.pth` no lo trae. Aun con
  las formas calzando, el estado inyectado no seria una captura del estado nativo.
- **La velocidad de swap.** No se midio. No hay numero de ms por intercambio de
  estado, ni CPU ni Vulkan.
- **Los otros dos GGUFs rwkv** (`base-rwkv29-Q4_K_M-local.gguf`) no se probaron; se
  anuncio solo el default y el f16.
- **GGUF != fuente del `.pth`.** Los pesos del GGUF (2.9B) y del `.pth` de estado
  (0.4B) son de modelos distintos: aunque las formas calzaran, el estado seria de
  otro modelo.
