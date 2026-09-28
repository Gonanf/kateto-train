# Knowledge: Kateto / RWKV — decisiones, ROSA, GGUF y RX 6500 XT

Fecha: 2026-09-03. Resumen de todo lo investigado y decidido para entrenar a
Kateto (asistente argentino, canchero, agente con tool calling) sobre RWKV.

---

## 1. Decisiones de arquitectura

| Tema | Decisión | Por qué |
|---|---|---|
| Base model | `RWKV7-G1j-2.9B-20260831` (formato transformers: `fla-hub/RWKV7-G1j-2.9B-20260831`, espejo `RWKV/RWKV7-G1j-2.9B`) | Multilingüe (es), chat-ready, **template con tool-calling nativo** (` ```json {"name":..,"arguments":..}``` `), safetensors ~5.9GB bf16. Fallbacks: `RWKV/RWKV7-G1j-1.5B` (T4 holgado) |
| Método | LoRA (peft/trl) sobre base congelado | Full FT de 2.9B no entra en T4; LoRA r16/α32 es suficiente para persona + formato |
| Tooling | `transformers` + `peft` + `trl` + `flash-linear-attention` (fla) | **NO** usar Joluck/RWKV-PEFT (ex JL-erhu): dio problemas y es un fork viejo. Con el modelo en formato HF, el stack estándar es más confiable |
| Datos | `sft_train.jsonl` (618 filas, persona argentina + Kateto tool calling, formato `query`/`response` y filas con `openai.messages`) + 5 datasets argentinos de HF (texto crudo) + `bertin-project/alpaca-spanish` (→ q/a) | Persona/cultura/chistes viene del sft; dominio argentino del corpus; general SFT del alpaca |
| Estrategia | Stage 1: domain adaptation texto crudo (lr 2e-4) → Stage 2: SFT persona + tools **al final** (assistant_only_loss, 2 épocas) | Que lo último que aprenda sea lo que más importa (persona + tool calling) |
| Reliability de tool calls | Es problema de **datos**, no de arquitectura: template consistente + más ejemplos (`scripts/build_toolcalling_dataset.py` con `--limit` alto) | El template nativo del G1j ya define el formato exacto |

Scripts: `colab/kateto_colab_main.py` (plan principal, ~2.5-3h en T4),
`colab/kateto_colab_rwkv8_small.py` (experimento ROSA).


## 2. RWKV-8 / ROSA — qué es y qué se puede hacer hoy

- **ROSA (QKV-ROSA)**: capa de *recuperación exacta* de RWKV-8. Proyecta el
  hidden stream a bits Q/K discretos; K se indexa estilo suffix-automaton,
  Q rutea queries y el payload V matcheado se re-inyecta al residual stream.
  Ataca el punto débil de RWKV: **copiar strings exactos** (IDs, args de
  tool calls) que el estado comprimido del RNN pierde.
- **No hay RWKV-8 multilingüe/chat pretrained**. `BlinkDL/rwkv-8-pile` existe
  pero es: Pile (inglés-only), "early testing versions and I will iterate"
  (palabras de BlinkDL), 73 downloads/mes. Inútil para Kateto directamente.
- **SÍ se puede graft ROSA sobre un RWKV-7 existente**: la capa es aditiva
  (lee el stream, escribe al residual), así que base congelado + Q/K/V
  embeddings entrenables (millones de params) = adapter tipo LoRA.
  - `zyaaa-ux/ROSA-Tuning` — exactamente esto (adapter tunable sobre modelo pretrained)
  - `aabbdev/rosa`, `wjie98/rosa_soft`, `johanwind/wind_rosa` — variantes diferenciables/soft (ruta probada)
  - `xiaoiecc/qkv-rosa-fast-exact-backward` — backward EXACTO del routing duro (research, no pipeline)
  - `Juste-Leo2/ROSA-GPU-RWKV8` — implementación GPU del forward
- **Plan de validación**: `colab/kateto_colab_rwkv8_small.py` entrena un tiny
  RWKV-7 L6-D512 desde cero + una capa ROSA soft en la misma tarea de "copiá
  el argumento exacto" y compara exact-recall. Si ROSA gana → ROSA-Tuning
  sobre el G1j-2.9B congelado con datos de `toolcalling_sft`.
- **Caveat honesto**: nadie publicó un G1j-2.9B + ROSA. Es experimento, no garantía.
- Otros links evaluados y descartados para este proyecto:
  `Tnhn07/modded-nanogpt` (recetas para transformers GPT, no RWKV),
  `Expertium/rwkv-anki-autoresearch` (inferencia/eval),
  `xiaol/Multi-state-RWKV-online-memory` y
  `xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1` (Gemma4 congelado +
  adapter RWKV-MS online-memory entrenado en agent traces; requiere runtime
  patched `deltamem`, tuning angosto a un agente de telecom — valida la idea
  de entrenar memoria recurrente sobre traces de agente, pero no es drop-in).


## 3. GGUF + imatrix (llama.cpp)

- **llama.cpp YA soporta RWKV-7**: inference merged (lineage `MollySophia/rwkv-mobile`),
  kernels CUDA fused WKV7 merged (ago 2026), más PRs de optimización en curso.
- Pipeline existente y probado: `export_gguf.sh` = merge LoRA → pth → GGUF
  (converter de rwkv-mobile) → `llama-imatrix` con `data/calib.txt` →
  `llama-quantize --imatrix ... Q4_K_M` → `llama-bench`.
- Pendiente: shim HF→pth (~20 líneas de rename de pesos) para que la salida
  de `kateto_colab_main.py` (safetensors HF) entre al converter.
- **ROSA NO se puede compilar a GGUF**: GGUF guarda tensores estáticos; ROSA
  es estructura de datos en runtime (suffix automaton + routing discreto +
  hooks de read/write). Solo las tablas de embeddings Q/K/V son tensores.
  Hacerlo requeriría un op ggml nuevo + kernels CPU/CUDA + serialización de
  estado = proyecto C++ grande. Ni `llama-imatrix` ni la quantization aplican
  a la capa (imatrix cuantiza pesos estáticos).
- **Workarounds**:
  1. *Destilación*: usar el modelo con ROSA para generar datos de training
     más duros (copias exactas) y hornear la habilidad en pesos que sí van a GGUF.
  2. *Router híbrido*: requests raros de exact-recall → runtime Python
     (PyTorch/fla) con ROSA; todo lo demás → GGUF en llama.cpp.


## 4. RX 6500 XT — teoría para usarla en fine tuning

Hardware: RDNA2 Navi 24 (**gfx1034**), **4GB VRAM**, bus 64-bit. Sin soporte
ROCm oficial (gfx1034 no está en la lista). Dos rutas para que PyTorch la vea
como "CUDA device" (`torch.cuda.*` es la API; el backend debajo es HIP):

### Ruta A: ROCm + override (recomendada para probar)

1. Linux + ROCm 6.x (`amdgpu-install`), usuario en grupos `video`/`render`.
2. `export HSA_OVERRIDE_GFX_VERSION=10.3.0.0` — el runtime trata la 6500 XT
   como gfx1030 (RX 6800). La mayoría de los kernels RDNA2 son compatibles;
   gfx1034 compila instrucciones gfx10 genéricas.
3. `pip install torch --index-url https://download.pytorch.org/whl/rocm6.x`
   → `torch.cuda.is_available()` da True (es HIP, pero la API es la de CUDA).
4. Sanity check: tensor op en `cuda`, después el script ROSA tiny.

**Límites reales (no negociables)**:
- **4GB VRAM**: el G1j-2.9B NO entra ni para LoRA (solo pesos bf16 = 5.9GB).
  Lo máximo: modelos ~0.1-0.5B con LoRA, o el **tiny RWKV-7 L6-D512 del
  experimento ROSA (~50M params: pesos + Adam ≈ 0.6-1GB — entra sobrado)**.
- gfx1034 sin instrucciones MFMA → throughput bajo; esperá 3-6x más lento que
  una T4. Suficiente para el experimento tiny (1-2h), no para el plan principal.

### Ruta B: ZLUDA (CUDA real sobre HIP)

ZLUDA traduce CUDA→HIP en runtime: permite correr wheels CUDA-only
(`bitsandbytes`, etc.). Estado: funciona en RDNA2 con parches
(repos community de ZLUDA + hipBLASLt), pero es frágil, y con 4GB el límite
de VRAM es el mismo. Usarlo solo si falta una lib CUDA-only que ROCm no tiene.

### Ruta C (bonus): inference, no training

Para **inferencia** la 6500 XT sí es útil: llama.cpp con backend **Vulkan**
soporta RDNA2 bien, y el Kateto Q4_K_M 2.9B (~1.8GB) entra entero en los
4GB → agente local barato mientras la GPU grande entrena.

### Conclusión práctica

| Dispositivo | Rol |
|---|---|
| Colab T4 | plan principal (LoRA 2.9B, 2 stages) |
| RX 6500 XT (ROCm + override) | experimento ROSA tiny (L6-D512), pruebas de código, inference GGUF Vulkan del Q4 |

## 5. qvac-fabric-llm.cpp — fine tuning con Vulkan (tetherto)

- **Qué es**: fork mantenido de `llama.cpp` (baseline upstream b7248, sincroniza
  regularmente) de QVAC Fabric. MIT. 122 stars, Linux/Windows/macOS/Android/iOS.
- **Feature exclusiva vs upstream**: **LoRA fine-tuning on-device** sobre
  **CPU, Vulkan y Metal** — SFT con checkpointing y LR scheduling. También
  TurboQuant KV cache, BitNet inferencia+training en Vulkan/Metal, carga de
  modelos desde memoria (edge/mobile focus).
- **Build**: `cmake -B build -DGGML_VULKAN=ON && cmake --build build --config Release`
- **Por qué es interesante acá**: Vulkan corre nativo en RDNA2 → la **RX 6500 XT
  podría fine-tunear LoRA sin ROCm ni ZLUDA**, y hasta en Windows.
- **Caveats (honestos)**:
  1. Como soporta "cualquier GGUF de llama.cpp" para *inferencia*, pero el
     training con backward pass probablemente esté implementado para las
     arquitecturas transformer clásicas — **hay que verificar si el grafo de
     RWKV-7 tiene backward en ggml** (upstream tiene RWKV-7 inference; el
     backward es otro cantar). Testear primero con un GGUF transformer chico.
  2. 4GB VRAM: LoRA sobre el Q4 2.9B (1.8GB pesos + activaciones + optimizer
     de adapters) está justo en el límite; arrancar por el 1.5B o el tiny.

## 6. Todos los links (todo lo que se evaluó en este proyecto)

### Datasets (HuggingFace) — usados en la celda 4 del plan principal
- https://huggingface.co/datasets/samuelandaudreymedianetwork/che-argentina-travel-article-corpus
- https://huggingface.co/datasets/Villaitech/argentina-news
- https://huggingface.co/datasets/finiteautomata/news-argentina
- https://huggingface.co/datasets/jmibarlucea/mistral-argentina-reddit
- https://huggingface.co/datasets/juanmoisesdelas/adolescentes-argentina-uso-de-redes-sociales-sueno-y-somnolencia-base-recodifica
- https://huggingface.co/datasets/bertin-project/alpaca-spanish (→ pares q/a SFT)

### Modelos (HuggingFace)
- https://huggingface.co/fla-hub/RWKV7-G1j-2.9B-20260831 — **base elegido** (formato transformers)
- https://huggingface.co/RWKV/RWKV7-G1j-2.9B-20260831 — espejo oficial (plan B)
- https://huggingface.co/RWKV/RWKV7-G1j-1.5B-20260831 — fallback liviano (plan C)
- https://huggingface.co/BlinkDL/rwkv-8-pile — RWKV-8 sobre Pile, inglés-only, "early testing"
- https://huggingface.co/xiaol/gemma-4-e4B-hybrid-rnn-mem-rwkv-fable5-gpt5.5-v1 — Gemma4 congelado + RWKV-MS online-memory (runtime deltamem, no drop-in)

### Repos de investigación / entrenamiento
- https://github.com/xiaoiecc/qkv-rosa-fast-exact-backward — backward EXACTO de QKV-ROSA (su README lista el ecosistema ROSA: aabbdev/rosa, wjie98/rosa_soft, johanwind/wind_rosa, zyaaa-ux/ROSA-Tuning, bcml-labs/rosa-plus, x-0D/RASP, KakaruHayate/RWKV8-ROSA-FPGA, Juste-Leo2/ROSA-GPU-RWKV8)
- https://github.com/tetherto/qvac-fabric-llm.cpp — fork de llama.cpp con LoRA fine-tuning en Vulkan/Metal/CPU (ver sección 5)
- https://github.com/fla-org/flash-linear-attention — kernels RWKV7+ para transformers, training-ready (5.7k stars)
- https://github.com/Joluck/RWKV-PEFT — PEFT oficial RWKV (ex JL-erhu); descartado por problemas, superseded por el stack HF
- https://github.com/Tnhn07/modded-nanogpt — recetas de velocidad para transformers GPT (descartado: no RWKV)
- https://github.com/Expertium/rwkv-anki-autoresearch — RWKV eficiente para Anki (inferencia/eval, descartado)
- https://github.com/xiaol/Multi-state-RWKV-online-memory — memoria online multi-estado RWKV (investigación)
- https://github.com/BlinkDL/RWKV-LM — este repo (tokenizer, make_data, RWKV-v5/v7/v8)

### Tooling propio
- `colab/kateto_colab_main.py` — plan principal Colab
- `colab/kateto_colab_rwkv8_small.py` — experimento ROSA tiny
- `export_gguf.sh` — merge → GGUF → imatrix (calib.txt) → Q4_K_M → bench
- `scripts/build_toolcalling_dataset.py` — genera toolcalling_sft.jsonl (usar --limit alto para más reliability)
- `to_sft.py` — jsonl → sft_train/sft_eval/calib.txt

  3. Madurez: 122 stars, fork joven — no reemplaza al pipeline Colab, es el
     experimento "edge training".
- **Plan de test propuesto**:
  1. Compilar con Vulkan, correr inferencia del Kateto Q4_K_M (valida Vulkan en la 6500 XT).
  2. LoRA SFT chico sobre un GGUF transformer pequeño (valida el entrenamiento).
  3. Si RWKV-7 backward existe en este fork → LoRA Kateto on-device. Si no →
     upstream issue / esperar, o usarlo solo para transformers.

| ZLUDA | solo si falta una lib CUDA-only |

