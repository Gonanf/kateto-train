# B2 — El estado se construye desde Python

**Pregunta:** el buffer de estado que B1 demostro inyectable, se puede **fabricar en Python**
en vez de depender de que lo produzca llama.cpp?
**Respuesta, medida en esta maquina: si.** `rwkv_pipeline/state_bytes.py` re-serializa un dump
de 21.627.676 bytes a **0 diferencias**, y el buffer reconstruido se inyecta igual que el original.

## Como se corre

```bash
bash script/probar-estado-python.sh              # CPU, ~10 s en caliente
venv/bin/python -m pytest tests/test_state_bytes.py -q
```

El script no compila nada: reusa el binario de B1 (`tools/gguf_state_roundtrip`).
Hace dump -> rebuild en Python -> `cmp` -> inyeccion. Termina en `RESULTADO B2: PASS` o
`RESULTADO B2: FAIL` con el motivo.

## El formato, campo por campo

Deducido del codigo de `llama.cpp-master` @ `ce8caa6`, no de un doc. `read` y `write`
coinciden exactamente (`llama-memory-recurrent.cpp:847` vs `:897`).

| offset | size | campo | linea que lo prueba |
|---|---|---|---|
| 0 | 4 | `uint32 io_magic = 0xaf143cd8` | `llama-context.cpp:3093` (constexpr), escrito en `:3099` y `:3116` |
| 4 | 4 | `int32 seq_id` | `llama-context.cpp:3100` |
| 8 | 4 | `uint32 cell_count` | `llama-memory-recurrent.cpp:841` |
| 12 | 8 x `cell_count` | por celda: `int32 pos` + `uint32 n_seq_id` | `:885-886`; `n_seq_id` forzado a 0 si `seq_id != -1` en `:883` |
| — | 4 | `uint32 s_trans = 0` | `:901` (const en `:898`) |
| — | 4 | `uint32 n_layer` | `:902` |
| — | 32 x (12 + payload) | `int32 r_type = 0` (F32) | `:911-912` |
| | | `uint64 r_size_row` | `:915-916` (`ggml_row_size(F32, n_embd_r)`) |
| | | `cell_count * r_size_row` bytes de `r` | `:923` |
| | | el `p` de esa capa, pegado: `uint64 p_size_row` + payload | `:928-929`, `:933` |
| — | 32 x (12 + payload) | los `s` van **todos juntos al final**: `int32 s_type` | `:938` (bucle separado), `:944-945` |
| | | `uint64 s_size_row` + `cell_count * s_size_row` bytes | `:948-949`, `:956` |

Los payloads son `memcpy` crudo: `write_tensor` / `read_tensor` no agregan cabecera ni padding
(`llama-context.cpp:2628` y `:2868`). No hay alineacion entre tensores.

Los numeros de este modelo salen de la metadata del GGUF: `rwkv7.block_count = 32`,
`rwkv7.embedding_length = 2560`, `rwkv7.wkv.head_size = 64`, y `token_shift_count = 2`
(default en `llama-hparams.h:143`). Entonces `n_embd_r = 2*2560 = 5120` (fila de 20.480 bytes,
`llama-hparams.cpp:208-211`) y `n_embd_s = 2560*64 = 163.840` (fila de 655.360 bytes,
`llama-hparams.cpp:236-239`). La aritmetica cierra exacta:

```
8 + 4 + 8 + 8 + 32*12 + 32*20480 + 32*12 + 32*655360  =  21.627.676
```

`cell_count` es **1**, no 7: RWKV es recurrente, el estado es de tamano fijo y la cache
guarda solo la ultima celda (`pos 6`, el ultimo token del prompt).

## Trampa del formato

La tensor `p` **no tiene discriminante**: el escritor decide escribirla con
`p_l[il] != nullptr` del modelo, no con algo del stream. Un buffer con `p` es indistinguible
de uno sin `p` leyendo bytes. Por eso `parse_state_bytes` acepta `p_cols` (capa -> bytes) y
`build_state_bytes` no lo necesita (en la dict la `p` ya viene agrupada por capa). Este modelo
no tiene `p`, asi que la via normal no lo necesita.

## Que NO prueba

- **No mapea un `.pth` nativo.** Este buffer lo produjo llama.cpp y lo re-serializo Python: el
  parseo y el rebuild son el mismo formato, ida y vuelta. Traducir los tensores
  `blocks.{i}.att.time_state` de PyTorch a este layout es **el paso siguiente**, y no esta hecho.
- **No mide velocidad de swap.** El round-trip es 21 MB de I/O en CPU. Si un swap de voz por
  GGUF entra en el presupuesto de los 48-135 ms nativos, no se midio.
- **No prueba integridad de datos, solo fidelidad estructural.** Un round-trip es identidad por
  construccion sobre el payload: si un byte de un tensor esta corrupto, parse->build lo devuelve
  igual y el `cmp` pasa. Lo que si detecta es orden, offsets, tamanos y campos perdidos.
- **No toca ROSA**, ni el camino nativo, ni `repeat_penalty` (ver `docs/gguf-estado-roundtrip.md`).
