# Kateto Finetune en Colab

## Scripts (nuevos, 2026-09)

### `kateto_colab_main.py` — plan principal (usar este)
RWKV7-G1j-2.9B en formato transformers (`fla-hub/RWKV7-G1j-2.9B-20260831`,
espejo de `RWKV/RWKV7-G1j-2.9B`) + LoRA con `peft`/`trl`. **No usa
Joluck/RWKV-PEFT ni .pth.** ~2.5-3h en T4:

1. deps (`transformers`, `peft`, `trl`, `flash-linear-attention`)
2. carga del modelo (plan B: `RWKV/RWKV7-G1j-2.9B`, plan C: `1.5B`)
3. subís `data/sft_out/sft_train.jsonl` (618 filas)
4. baja los 5 datasets de texto argentino de HF (uno roto no corta nada) +
   `bertin-project/alpaca-spanish` → pares q/a
5. todo a formato `messages` con el system prompt Kateto (el template del
   modelo ya trae tool-calling nativo)
6. Stage 1: adaptación de dominio sobre el corpus (LoRA, lr 2e-4)
7. Stage 2: SFT persona + tool calling, **último a propósito**
   (`assistant_only_loss`, 2 épocas)
8. test de inferencia (chiste + tool call)
9. export `kateto-lora.zip` + download

### `kateto_colab_rwkv8_small.py` — experimento ROSA (opcional, sesión aparte)
Tiny RWKV-7 L6-D512 **desde cero** (`RWKV-v7/train_temp`, binidx via
`RWKV-v5/make_data.py`) sobre corpus argentino + tarea de copia exacta de
argumentos de tool calls, y una capa **ROSA soft retrieval** entrenada en la
misma tarea. Mide exact-recall de ambas y compara. ~1.5-2h en T4.
Validado local en CPU (generación de datos + training + eval greedy).

## Legacy

El notebook viejo `Kateto_Finetune.ipynb` usaba `Joluck/RWKV-PEFT` (ex
JL-erhu) con el pth de BlinkDL. Queda de referencia pero está superseded por
`kateto_colab_main.py` (daba problemas y solo llega a v7).

## Qué necesitás (main)

- Colab con GPU (T4 alcanza; TPU no sirve).
- `data/sft_out/sft_train.jsonl` de tu PC (618 filas, query/response).
- Un rato: el modelo baja ~5.9GB y las dos etapas de entrenamiento son
  ~50-60min cada una.

## Si algo falla

- `CUDA out of memory`: reiniciá el entorno; probá el plan C (1.5B).
- Datasets HF caídos: la celda 4 sigue con los demás y lista los fallados.
- Desconexión de Colab: reejecutá 1-5 y seguí; cada stage guarda adapter.

