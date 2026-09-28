# Decisión: GGUF fuera del camino de servicio (ROSA + estados por agente)

Fecha: 2026-09-15.
Motivo: el diseño de Kateto necesita **emular varios agentes cambiando rápido el
estado inicial** (`S_0`) del mismo modelo base, y además **ROSA** (memoria
asociativa en runtime). El camino GGUF/llama.cpp no soporta ninguna de las dos
cosas. Este documento fija por qué y qué queda vigente.

## Requisitos que definen la decisión

1. Un modelo base cargado **una sola vez**, N identidades (voces/agentes) que se
   activan cambiando su tensor de estado. Cambio **durante la conversación**, con
   latencia de milisegundos.
2. **ROSA** activo: `rwkv_pipeline/rosa_module.py` — estructura de datos en
   runtime (indexado de K, ruteo de Q, re-inyección de V + hooks de read/write
   sobre el residual stream). No es un tensor; es maquinaria del forward.

## Por qué GGUF no sirve para eso

1. **No acepta estados externos.** Los estados de la Capa 2 (`state-tuning`) son
   tensores PyTorch guardados como `blocks.{i}.att.time_state` (6-20 MB según el
   tamaño del base). llama.cpp no tiene bandera ni API para **inyectar** un
   estado inicial externo: `--slot-save-path` guarda/restaura el estado que
   **el propio runtime generó** (cache de slot), no un tensor de afuera.
   Verificado sobre el build instalado (`llama-server` 0.1.2-dev, build 10502,
   commit 8ff87cc): las únicas opciones de estado son `--slot-save-path` y
   `--cache-reuse`.
2. **ROSA no se puede compilar a GGUF.** GGUF serializa tensores estáticos; ROSA
   es estructura de datos en runtime. Solo las tablas de embeddings Q/K/V son
   tensores —el resto requeriría un op `ggml` nuevo + kernels + serialización de
   estado: proyecto C++ grande. Ni `llama-imatrix` ni `llama-quantize` aplican.
3. **Cambio de voz rápido: no.** Medido en el camino nativo el 2026-09-15:

   | operación | tiempo |
   |---|---|
   | swap de voz (cargar estado + construir `S_0`), 0.4B | **48-135 ms** (promedio 47.6 ms sobre 15 swaps) |
   | prefill 20 tokens, 0.4B, RX 6500 XT | 2859 ms (≈7 tok/s) |

   En GGUF el equivalente sería recargar un modelo por voz (segundos) y no
   entran 5 voces en 4 GB de VRAM (2.9B Q8_0 ≈ 3.2 GB cada una).

## Consecuencia

- **El camino de servicio y de evaluación de Kateto es el nativo**:
  `rwkv_pipeline/infer_kateto.py` (+ `rosa_module.py`), con `.pth` + estados en
  `out/rwkv_states/<voz>/`. El proveedor de la app usa
  `backend = "rwkv"`, `enable_rosa = true` (hoy está comentado en
  `~/.config/kateto/config.toml` y se sirve HTTP en su lugar).
- **`export_gguf.sh` queda como bench/export opcional**, fuera del camino
  crítico. No se usa ni para servir ni para verificar calidad (una verificación
  por GGUF mide un artefacto que el producto no usa, y con `repeat_penalty`
  distinto al del runtime nativo).
- **Los estados entrenados en la Capa 2 son inutilizables en GGUF.** Cualquier
  ruta que los ignore (Q4_K_M, GGUF "propio") no puede emular ni una voz.

## Pendiente / caveat

La verificación de calidad nativa **no está resuelta**: al 2026-09-15 el modo
one-shot de `infer_kateto.py` degenera para *los tres* combos probados —
base vieja + estado viejo (el que se servía), base nueva sin estado, y base
nueva + estado. Los tres producen `<|>`/emojis/repeticiones. Hipótesis sin
confirmar: el sampler nativo (`alpha_presence 0.6` / `alpha_frequency 0.5`) es
más débil que el `repeat_penalty = 1.30` que sí domaba el texto por
llama.cpp/GGUF. Antes de firmar cualquier mejora de entrenamiento hay que
resolver esto: si el modelo no genera texto coherente por el camino que se va a
servir, la métrica que importa (la del GGUF) es la equivocada.
