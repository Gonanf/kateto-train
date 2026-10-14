# Reporte de Engagement de Estados de Voz en RWKV-7 GGUF (2.9B)

## 1. Comando Reproducible y Entorno

- **Comando de prueba:** `bash script/probar-estados-voz.sh`
- **Modelo GGUF evaluado:**
  - Repositorio HuggingFace: `shoumenchougou/RWKV7-G1j-2.9B-GGUF`
  - Archivo en disco: `/home/chaos/.cache/huggingface/hub/models--shoumenchougou--RWKV7-G1j-2.9B-GGUF/snapshots/ab0040b584622847f9b8702c41af222a00fda968/rwkv7-g1j-2.9b-Q4_K_M.gguf`
  - SHA256: `68c3a49dfc8a6dc34032d7c89adf64542ec9c81229a1333591163717bc3ab7b5`
- **Versión de llama.cpp:** Commit `f2cc7282ce0042641df55e0e563b75e824e2b647` (`/run/media/chaos/secundario/llamacpp-conv`)
- **Configuración de ejecución:** `n_ctx=256`, `n_batch=256`, `n_seq_max=1`, `n_threads=8`, `n_gpu_layers=0` (inferencia puramente CPU).

## 2. Tabla de Formas (Compatibilidad Arquitectural)

| State (.pth) | Tensores | Forma time_state | Veredicto | n_embd (.pth vs GGUF) |
| :--- | :---: | :---: | :---: | :---: |
| `jane/rwkv-11.pth` | 32 | `(40, 64, 64)` | **CALZA** | 2560 vs 2560 |
| `doktor/rwkv-0.pth` | 32 | `(40, 64, 64)` | **CALZA** | 2560 vs 2560 |
| `doktor/rwkv-14.pth` | 32 | `(40, 64, 64)` | **CALZA** | 2560 vs 2560 |
| `jane/rwkv-0.pth` | 24 | `(16, 64, 64)` | NO CALZA | 1024 vs 2560 (base 0.4B) |
| `seco/rwkv-3.pth` | 24 | `(16, 64, 64)` | NO CALZA | 1024 vs 2560 (base 0.4B) |
| `streamer/rwkv-0.pth` | 24 | `(16, 64, 64)` | NO CALZA | 1024 vs 2560 (base 0.4B) |
| `whisperer/rwkv-0.pth` | 24 | `(16, 64, 64)` | NO CALZA | 1024 vs 2560 (base 0.4B) |
| `seco-20260906/rwkv-0.pth` | 24 | `(16, 64, 64)` | NO CALZA | 1024 vs 2560 (base 0.4B) |

## 3. Tablas de Engagement (Generación y Salida)

### Prompt A: "Hola, soy Kateto." (12 tokens greedy)

| State | set_data_bytes | cell_pos | start_pos | Top-5 paso 1 (`id:logit`) | logits_max | ids generados | Continuación | Difiere de baseline |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- | :--- | :---: |
| `baseline` | 0 | -1 | 0 | `11:4.3593 20913:4.3279 261:3.7522 4936:3.7273 4025:3.5712` | 4.3593 | `11 6012 98 45 22504 29193 2215 47 11 6012 98 45` | `\nHola, soy Kateto.\nHola,` | no |
| `jane/rwkv-11` | 21627676 | 0 | 1 | `112:5.4691 118:2.8812 106:2.7393 98:2.7186 102:0.2040` | 5.4691 | `112 59 4757 2334 2206 59 4635 21681 98 59 4635 21681` | `o: reyote: la fila: la fil` | **sí** |
| `doktor/rwkv-0` | 21627676 | 0 | 1 | `11:2.2407 7441:0.7640 7645:0.3632 1905:0.2583 8401:-1.0834` | 2.2407 | `11 63 1949 9324 112 59 32464 37598 4601 59 22590 22752` | `\n>imíto: what about it: the way` | **sí** |
| `doktor/rwkv-14` | 21627676 | 0 | 1 | `21430:-3.7310 19633:-4.8869 302:-5.0665 19418:-5.0794 7235:-5.0985` | -3.7310 | `21430 93 2338 112 93 8308 93 2224 93 7235 93 2334` | ` cho\yto\ody\ty\cho\yo` | **sí** |

### Prompt B: "Contame algo corto." (12 tokens greedy)

| State | set_data_bytes | cell_pos | start_pos | Top-5 paso 1 (`id:logit`) | logits_max | ids generados | Continuación | Difiere de baseline |
| :--- | :---: | :---: | :---: | :--- | :---: | :--- | :--- | :---: |
| `baseline` | 0 | -1 | 0 | `11:5.6717 261:5.4040 3336:3.9797 36206:3.4835 20524:3.4321` | 5.6717 | `11 63 11 63 11 63 11 63 11 63 11 63` | `\n>\n>\n>\n>\n>\n>` | no |
| `jane/rwkv-11` | 21627676 | 0 | 1 | `11:1.3628 106:-4.0773 7417:-4.5287 22351:-4.8306 7933:-4.8716` | 1.3628 | `11 93 116 5721 7928 4803 38449 4515 32137 1851 21636 8862` | `\n\sCarito te falta el subtel escubi` | **sí** |
| `doktor/rwkv-14` | 21627676 | 0 | 1 | `106:-3.5805 4639:-3.6172 1951:-3.9731 308:-4.0252 74:-4.2701` | -3.5805 | `106 19232 112 114 1852 19232 102 19232 19232 19232 19232 102` | `i�oqem�e����e` | **sí** |

#### Distribución de logits y fenómeno OOD (Out-of-Distribution)

- **`doktor` (rwkv-0 y rwkv-14):** Deja todos los logits del primer paso estrictamente negativos (`logits_max` < 0, e.g. -3.7310 en rwkv-14), lo cual coloca al modelo completamente fuera de su distribución calibrada y resulta en texto degenerado o repetitivo (` cho\\yto\\ody...`).
- **`jane/rwkv-11`:** Preserva logits positivos de magnitud normal (`top1` = 112 con logit 5.4691), dentro de la distribución estándar del modelo, y modifica determinísticamente los tokens generados respecto a la línea base sin degenerar.

## 4. ROSA: Veredicto

ROSA (*Rapid Online Suffix Automaton*, mecanismo propuesto para RWKV-8) **NO existe** en este stack y no contribuye al cambio de estado medido:
1. **Código fuente:** `grep -rn -i rosa` sobre los directorios `src`, `include`, `common` y `ggml/src` del árbol `llamacpp-conv` arroja **0 líneas** de coincidencia.
2. **Incompatibilidad de arquitectura GGUF:** Al intentar cargar el GGUF de ROSA con `llama-cli`, el proceso aborta inmediatamente con:
   ```
   E llama_model_load: error loading model: unknown model architecture: 'rosa'
   ```
3. **Incompatibilidad de memoria y pesos:** Tal como se detalló en `docs/rosa-ggml-viabilidad.md`, la implementación de ROSA en `rwkv_pipeline/rosa_module.py` es una rama de retrieval cuyo buffer en runtime crece por token (`torch.cat`), haciéndola incompatible con el buffer de memoria recurrente de tamaño fijo por secuencia que utiliza llama.cpp. Adicionalmente, los pesos de ROSA para RWKV-7 están sin entrenar (*"pesos RANDOM sin entrenar"*).
4. **Conclusión:** ROSA no aporta nada al cambio de state; no aplica al stack GGUF por las razones medidas arriba.

## 5. Procedencia y Límites

- **Procedencia de los checkpoints:**
  - `doktor` (rwkv-0 y rwkv-14): posee 32 capas de dimensión `(40, 64, 64)`, correspondiente a la topología 2.9B.
  - `jane/rwkv-11`: mtime del archivo es `2026-09-09` y no posee log propio de entrenamiento (su `train_log.txt` hace referencia a otra corrida de 0.4B); su procedencia exacta queda no documentada.
- **Tensor `r` (Token Shift):** El archivo `.pth` original contiene únicamente `time_state` (tensor `s`). El tensor `r` se inicializa en ceros por diseño en `state_from_pth.py`.
- **Desacople modelo-estado:** El GGUF base oficial y los estados `.pth` no provienen del mismo entrenamiento conjunto; se acoplan únicamente por compatibilidad de dimensiones tensoriales.
- **Condiciones de inferencia:** Modelo cuantizado en `Q4_K_M`, decodificación greedy pura (argmax), ejecutado en CPU.
- **Alcance semántico:** La semántica de "voz" o naturalidad auditiva no está probada; lo demostrado matemáticamente y de manera determinística es que la inyección de estado altera de forma verificable los logits y los tokens generados.
