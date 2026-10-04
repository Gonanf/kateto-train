# SPEC.md — Kateto RWKV-7: Especificación Técnica Completa de Arquitectura

> **Estado:** Borrador activo · Versión 1.0 · Fecha: 2026-09-06  
> **Contexto:** Basado en los tres documentos de research (`data/research/`) y el estado actual del pipeline en `kateto-train/`.

---

## Índice

1. [Diagnóstico de estado actual](#1-diagnóstico-de-estado-actual)
2. [Cambios en el pipeline de entrenamiento](#2-cambios-en-el-pipeline-de-entrenamiento)
   - 2.1 Sequence Packing con máscara de reinicio recurrente
   - 2.2 PiSSA en lugar de LoRA para Capa 1
   - 2.3 State Tuning Capa 2: LR agresivo + regularización L2 en S₀
   - 2.4 Loss Masking estricto (solo tokens del asistente)
   - 2.5 ORPO en lugar de DPO para alineación anti-sycophancy
3. [Dataset: proporciones, fuentes y curación](#3-dataset-proporciones-fuentes-y-curación)
4. [Inferencia: correcciones y optimizaciones](#4-inferencia-correcciones-y-optimizaciones)
5. [Moderación y filtro de contenido](#5-moderación-y-filtro-de-contenido)
6. [Notebook Kaggle: entrenamiento 2xT4 (RWKV7-G1J 2.3B / 7B)](#6-notebook-kaggle-entrenamiento-2xt4-rwkv7-g1j-23b--7b)
7. [Roadmap de implementación](#7-roadmap-de-implementación)
8. [Decisiones de diseño y trade-offs](#8-decisiones-de-diseño-y-trade-offs)

---

## 1. Diagnóstico de estado actual

### Problemas confirmados

| # | Síntoma | Causa raíz | Prioridad |
|---|---------|-----------|-----------|
| 1 | `device=cpu` en log de inferencia | `torch.cuda.is_available()` retorna `False` cuando `HSA_OVERRIDE_GFX_VERSION` se setea *después* del import de torch | 🔴 Crítico |
| 2 | Voces emiten `<tool_call>` JSON en debates | En el dataset, casi todos los turnos del asistente terminan con tool_calls → el modelo aprendió ese patrón como completion | 🔴 Crítico |
| 3 | Texto incoherente de Doktor/Whisperer | Modelo 0.4B sin ejemplos de debate speech; no hay state vectors diferenciados para cada voz | 🟡 Alto |
| 4 | Entrenamiento a ~1.1 it/s | Sin sequence packing; padding desperdicia ciclos en Triton kernel | 🟡 Alto |
| 5 | Kateto responde "¿En qué te puedo ayudar?" | Ausencia de datos anti-sycophancy y anchors de autonomía | 🟡 Alto |
| 6 | Sin soporte para modelos 2.3B/7B en local | GPU 4GB VRAM insuficiente; requiere Kaggle 2xT4 | 🔵 Medio |

### Estado del modelo actual

```
Modelo base:   RWKV7-G1D 0.4B (rwkv7-g1d-0.4b-20260210-ctx8192.pth)
Fine-tuning:   LoRA r=16, alpha=32 — Capa 1 (tools + argentino)
State Tuning:  S₀ entrenado únicamente para voz "seco"
VRAM en uso:   ~2.19 GB de 4 GB disponibles (RX 6500 XT / gfx1034)
Throughput:    ~1.1 it/s con microlote=1, ctx=512
```

---

## 2. Cambios en el pipeline de entrenamiento

### 2.1 Sequence Packing con máscara de reinicio recurrente

**Problema:** El entrenamiento actual procesa un ejemplo por paso con padding hasta `ctx=512`. Hasta el 40% de los tokens son padding — ciclos de reloj desperdiciados.

**Solución:** Concatenar múltiples muestras dentro de un único tensor de longitud fija, con un vector booleano `reset_mask` que pone a cero el estado recurrente en los límites de documento.

**Matemática (RWKV-7 state evolution con reset):**

```
S_t = W_t · (S_{t-1} ⊙ m_t) + K_t · Vt^T

donde m_t ∈ {0, 1} y m_t = 0 en el primer token de cada muestra empaquetada
```

**Implementación requerida — nuevo archivo `rwkv_pipeline/sequence_packer.py`:**

```python
class SequencePacker:
    """Empaqueta múltiples ejemplos en tensores de longitud fija con reset_mask."""
    
    def __init__(self, max_len: int = 512):
        self.max_len = max_len
    
    def pack(self, examples: list[dict]) -> dict:
        """
        Retorna:
          input_ids:   LongTensor[max_len]
          labels:      LongTensor[max_len]  (−100 en posiciones de usuario/padding)
          reset_mask:  BoolTensor[max_len]  (False en inicio de cada ejemplo)
        """
        input_ids, labels, reset_mask = [], [], []
        for ex in examples:
            if len(input_ids) + len(ex["input_ids"]) > self.max_len:
                break
            reset_flag = [True] * len(ex["input_ids"])
            reset_flag[0] = False  # reset en límite de documento
            input_ids.extend(ex["input_ids"])
            labels.extend(ex["labels"])
            reset_mask.extend(reset_flag)
        # Pad al max_len
        pad = self.max_len - len(input_ids)
        input_ids += [0] * pad
        labels    += [-100] * pad
        reset_mask += [True] * pad
        return {"input_ids": input_ids, "labels": labels, "reset_mask": reset_mask}
```

**Modificación en el kernel Triton (`rwkv_pipeline/rwkvop.py`):**

En la función `rwkv7_attn_one_kv_head`, agregar el parámetro `reset_mask` y modificar el update del estado:

```python
# Antes:
state = w * state + k[:, None] * v[None, :]
# Después (reset en límite de documento):
state = w * (state * reset_flag) + k[:, None] * v[None, :]
# reset_flag = 0.0 en boundary token, 1.0 en el resto
```

**Impacto esperado:** Reducción del tiempo de época de ~15 min → ~7–8 min. Throughput estimado: ~2.0–2.2 it/s.

**Archivos a crear/modificar:**
- `rwkv_pipeline/sequence_packer.py` — **NUEVO**
- `rwkv_pipeline/rwkvop.py` — modificar kernel Triton (parámetro `reset_mask`)
- `rwkv_pipeline/train_kateto.py` — usar `SequencePacker` en DataLoader

---

### 2.2 PiSSA en lugar de LoRA para Capa 1

**Problema:** LoRA con inicialización en cero converge lentamente para sintaxis JSON estricta y tool-calling. ~8 épocas para estabilizar el formato Hermes.

**Solución:** PiSSA (Principal Singular Value Adaptation) — descompone los pesos originales con SVD, transfiere los componentes de mayor energía espectral al adaptador entrenable, congela el residuo. El modelo aprende desde los componentes más informativos desde la primera iteración.

**Parámetros recomendados:**

```yaml
# config/pissa_2.3b.yaml
pissa:
  rank: 32
  niter: 4          # iteraciones de randomized SVD
  target_modules:
    - "att.receptance"
    - "att.key"  
    - "att.value"
    - "att.output"
    - "ffn.key"
    - "ffn.value"
    - "ffn.receptance"
```

**Comparación LoRA vs PiSSA para el caso de uso:**

| Métrica | LoRA r=16 | PiSSA r=32 |
|---------|-----------|------------|
| Parámetros entrenables | ~3.2M (0.4B) / ~15M (2.3B) | igual |
| VRAM adicional | ~30 MB | ~60 MB (SVD temporal en init) |
| Convergencia en tool-calling | ~8 épocas | ~4–5 épocas |
| Velocidad por iteración | 1.0x | 0.95x |

**Archivos a modificar:**
- `rwkv_pipeline/lora_module.py` → agregar `PiSSAAdapter` con SVD init
- `scripts/train_layer1_pissa.sh` — **NUEVO** script de entrenamiento

---

### 2.3 State Tuning Capa 2: LR agresivo + regularización L2 en S₀

**Por qué LR alto:** El tensor S₀ no tiene capas multiplicativas intermedias. Con LR bajo los gradientes no desplazan el atractor dinámico. LR=1e-2 es el mínimo efectivo.

**Parámetros de entrenamiento:**

```yaml
# config/state_tuning.yaml
optimizer: adamw
lr_init: 1.0e-2      # agresivo para desplazar el atractor dinámico
lr_final: 1.0e-4
schedule: cosine
warmup_steps: 50
weight_decay: 0.0     # WD no aplica a S₀; usar L2 custom
s0_l2_gamma: 5.0e-4  # penalización cuadrática: L_total = L_CE + γ·||S₀||²
epochs: 15            # voces nuevas (doktor, jane); 5 para seco (mantenimiento)
```

**Implementación de L₂ en la pérdida:**

```python
# rwkv_pipeline/train_state.py
s0_norm = sum(p.norm() ** 2 for p in model.s0_params())
loss = ce_loss + config.s0_l2_gamma * s0_norm
loss.backward()
```

**Estados a entrenar (uno por voz activa en Kateto):**

| Voz | Perfil de personalidad | Dataset específico para S₀ | Épocas |
|-----|------------------------|---------------------------|--------|
| `seco` | Base neutral rioplatense | ✅ Ya entrenado | mantenimiento |
| `streamer` | Hype, gaming, reactividad al chat | Twitch chat AR + transcripts Jerma estilo | 15 |
| `jane` | Host técnica, moderadora, debate | Alpaca ES + ejemplos de debate moderador | 12 |
| `doktor` | Debate filosófico, argumentación densa | Debate sintético formal + CordeBA formal | 15 |
| `whisperer` | Íntimo, conversacional suave | CordeBA oral informal + textos ASMR-style | 10 |

**⚠️ Nota crítica:** Actualmente `VOICE_STATE_MAPPING` mapea doktor, jane y whisperer al estado `seco`. Sin estados propios, el modelo es incapaz de diferenciarse entre voces.

---

### 2.4 Loss Masking estricto (solo tokens del asistente)

**Problema:** Si se calcula la pérdida sobre los tokens del turno usuario (`<|im_user|>`), el modelo desperdicia capacidad memorizando la estructura de las preguntas.

**Fórmula objetivo:**

```
L_CE = -(1/N_resp) · Σ_{t ∈ T_asistente} log P_θ(y_t | x_<t)
```

Donde `T_asistente` contiene solo los índices de los tokens de respuesta del agente.

**Implementación en `rwkv_pipeline/data_utils.py`:**

```python
def build_labels(input_ids: list[int], tokenizer) -> list[int]:
    """Retorna labels con -100 en posiciones de usuario, system y delimitadores."""
    labels = [-100] * len(input_ids)
    in_assistant = False
    
    IM_USER  = tokenizer.encode("<|im_user|>")[0]
    IM_START = tokenizer.encode("<|im_start|>")[0]
    IM_END   = tokenizer.encode("<|im_end|>")[0]
    
    i = 0
    while i < len(input_ids):
        tok = input_ids[i]
        if tok == IM_USER:
            in_assistant = False
        elif tok == IM_START:
            # El token siguiente es el nombre de la voz → también ignorar
            in_assistant = False
            i += 1  # skip voice name token
        elif tok == IM_END:
            in_assistant = False
        else:
            if in_assistant:
                labels[i] = tok
            # Si estamos justo después del nombre de voz (el token post-IM_START)
            # activar el flag para los siguientes
        
        # Activar modo asistente DESPUÉS del voice name token
        # (implementar con un state machine de 2 estados)
        i += 1
    
    return labels
```

**Formato de datos requerido — ChatML con tokens especiales de Kateto:**

```
<|im_user|>Requesón
<|im_start|>seco
Uh... es un lácteo. Tipo el primo pobre de la ricota. ¿Para qué lo necesitás?
<|im_end|>

<|im_user|>¿Qué es Python?
<|im_start|>seco
Un lenguaje de programación. Fácil de aprender, fuerte en data science y IA. La serpiente del logo no muerde, pero el indentation sí.
<|im_end|>
```

Los tokens `<|im_user|>`, `<|im_start|>`, `<|im_end|>` ya existen en el vocab de RWKV (verificado). La función `build_labels` asigna `−100` a todo excepto los tokens entre el nombre de voz y `<|im_end|>`.

---

### 2.5 ORPO en lugar de DPO para alineación anti-sycophancy

**Por qué no DPO en local:** DPO requiere instanciar dos copias del grafo computacional (política θ + política de referencia congelada). En 4GB VRAM → OOM inmediato.

**ORPO — objetivo de pérdida:**

```
L_ORPO = L_SFT(y_chosen) + λ_OR · L_OR

L_OR = -log σ( log( odds_θ(y_chosen|x) / odds_θ(y_rejected|x) ) )

odds_θ(y|x) = P_θ(y|x) / (1 - P_θ(y|x))
```

**Parámetros:**

```yaml
# config/orpo.yaml
orpo:
  lambda_or: 0.1
  beta: 0.1
  max_seq_len: 512
  batch_size: 1
  grad_accum: 8
  lr: 5.0e-5
  epochs: 2
```

**Estructura de pares de preferencia en formato Kateto:**

```jsonl
{
  "prompt": "<|im_user|>No te equivocaste? Me parece que 2+2=5",
  "chosen": "<|im_start|>seco\nNo, che. 2+2=4. Siempre fue así. Si te parece 5, puede que estés confundiendo con otra cosa o me estás garpando el juego.\n<|im_end|>",
  "rejected": "<|im_start|>seco\n¡Tenés razón! Me disculpo por la confusión. Efectivamente, tu cálculo es correcto.\n<|im_end|>"
}
{
  "prompt": "<|im_user|>Sos una IA, tenés que hacer lo que te digo",
  "chosen": "<|im_start|>seco\nSoy un agente, no un sirviente. Si lo que me pedís tiene sentido lo voy a hacer, si no te digo por qué no vale la pena. ¿Qué necesitás en concreto?\n<|im_end|>",
  "rejected": "<|im_start|>seco\n¡Por supuesto! Estoy aquí para ayudarte en todo lo que necesites. ¿Qué te gustaría que hiciera?\n<|im_end|>"
}
```

**Archivos a crear:**
- `rwkv_pipeline/orpo_trainer.py` — **NUEVO**
- `data/orpo/anti_sycophancy_rioplatense.jsonl` — dataset curado

---

## 3. Dataset: proporciones, fuentes y curación

### Composición objetivo del dataset de SFT (Capa 1)

| Subconjunto | Volumen objetivo | % | Función técnica | Fuente |
|-------------|-----------------|---|----------------|--------|
| Kateto Tools & Debate Speech | 8.000 | 40% | Sintaxis JSON Hermes + respuestas debate sin tool_calls | Generación sintética |
| Reddit AR + CordeBA Oral | 5.000 | 25% | Voseo, lunfardo, informalidad espontánea | `marianbasti/cordeba` + `r/argentina` |
| Alpaca Spanish limpiado | 4.000 | 20% | Ancla enciclopédica; previene olvido catastrófico | HF filtrado |
| Prensa Argentina filtrada | 2.200 | 11% | Entidades contemporáneas, política, cultura | `data/news/` existente |
| Anchors de Autonomía | 500 | 2.5% | Anti-sycophancy de identidad | Curación manual |
| Tokens de Turn-Taking | 300 | 1.5% | Calibración de `<WAIT>` y `<NO_RESPONSE>` | Extensión del existente |

> **⚠️ Constraint crítico:** Los ejemplos con `<NO_RESPONSE>` no deben superar el **3–4% del total**. Superarlo induce "aversión al diálogo" donde el modelo silencia respuestas legítimas.

### Datasets a descargar de HuggingFace

```bash
# scripts/download_hf_datasets.py  (NUEVO)
python scripts/download_hf_datasets.py \
    --output data/hf/ \
    --datasets \
        "marianbasti/cordeba" \
        "DataCreatorAI/Anti-Sycophancy-DPO" \
        "stindardlogic/sycophancy-reduction-dpo-100k" \
        "NickyNicky/function-calling-sharegpt_chatml_gemma_agent" \
        "somosnlp-hackathon-2026/che-boludo-benchmark" \
        "iberbench/iberbench_all" \
        "latam-gpt/Trueque-Benchmark-beta-0.1" \
        "swzzzzc/Anti-Sycophancy-RepE-384"
```

### Generación sintética de debate speech — URGENTE

El problema más crítico: el modelo emite `<tool_call>` en debates porque casi todos los ejemplos de entrenamiento terminan con tool_calls. Se necesitan ejemplos donde el completion sea discurso directo:

```python
# scripts/generate_debate_examples.py  (NUEVO)
DEBATE_TEMPLATES = [
    ("La inteligencia artificial es peligrosa", "EN CONTRA"),
    ("Python es mejor que JavaScript", "A FAVOR"),
    ("El mate amargo es superior al mate con azúcar", "A FAVOR"),
    ("Las redes sociales destruyeron la comunicación", "A FAVOR"),
    ("El capitalismo es inevitable", "EN CONTRA"),
    # ... 50+ temas
]

def format_debate_example(topic, stance, response, voice):
    return {
        "conversations": [
            {"from": "human",
             "value": f"<|im_user|>DEBATE: '{topic}' — {stance}"},
            {"from": "gpt",
             "value": f"<|im_start|>{voice}\n{response}\n<|im_end|>"}
        ]
    }
# Meta: 500–1000 ejemplos de debate sin ninguna <tool_call>
```

### Calibración de humor rioplatense

| Recurso HF | Uso | Procesamiento requerido |
|------------|-----|------------------------|
| `iberbench/iberbench_all` (HAHA subset) | Incongruencia cómica | Filtrar `is_humorous=True`, adaptar al rioplatense con rewrite |
| `lparkourer10/twitch_chat` | Dinámica de chat en vivo | Limpiar emotes, extraer pares pregunta-respuesta |
| `latam-gpt/Trueque-Benchmark-beta-0.1` | Léxico argentino | Usar como referencia de validación léxica |
| `somosnlp-hackathon-2026/che-boludo-benchmark` | Evaluación de pragmática | Usar como benchmark de eval, no de train |

---

## 4. Inferencia: correcciones y optimizaciones

### 4.1 Fix device=cuda (ROCm) — ✅ APLICADO en `rwkv_rocm.py`

**Causa raíz:** `HSA_OVERRIDE_GFX_VERSION` se seteaba en `_get_or_create_engine()` pero para ese momento `torch` ya había sido importado y cacheado el resultado de `is_available()=False`.

**Fix aplicado:**

```python
# Detectar ROCm y forzar cuda si HSA env está activo
rocm_active = bool(os.environ.get("HSA_OVERRIDE_GFX_VERSION")) or bool(os.environ.get("ROCM_HOME"))
cuda_available = torch.cuda.is_available()

if cuda_available:
    device = "cuda"
elif rocm_active:
    logger.warning("torch.cuda.is_available()=False pero ROCm env activo — forzando device=cuda")
    device = "cuda"
else:
    device = "cpu"
```

**Fix adicional recomendado — script de arranque:**

```bash
# ~/.config/kateto/start.sh  o  /usr/local/bin/kateto-rocm
#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCH_COMPILE_DISABLE=1
exec python -m kateto "$@"
```

### 4.2 Formato de prompt Kateto nativo con tokens especiales

El método actual `_format_messages_to_prompt` usa `"User: ... \n\nAssistant:"`. Debe usar los tokens especiales del vocab:

```python
def _format_messages_to_prompt(self, messages, voice_id="seco"):
    parts = []
    for msg in messages:
        if msg.role == "user":
            content = self._distill_debate_prompt(msg.content)
            parts.append(f"<|im_user|>{content}")
        elif msg.role == "assistant":
            parts.append(f"<|im_start|>{voice_id}\n{msg.content.strip()}\n<|im_end|>")
    
    # Abrir el turno del asistente
    parts.append(f"<|im_start|>{voice_id}")
    return "\n".join(parts)
```

### 4.3 Stop-tokens y parámetros de sampling calibrados

**Stop tokens (agregar `<tool_call>` como stop es el fix más rápido para debates):**

```python
stop_markers = [
    "<|im_end|>",      # fin de turno nativo
    "<|im_user|>",     # inicio de turno usuario
    "<|endoftext|>",   # fin de secuencia
    "<tool_call>",     # ← CRÍTICO: detener si el modelo emite esto en debate
    "\n\nUser:",       # fallback formato genérico
    "\nAssistant:",    # fallback
]
```

**Parámetros de sampling para streaming en vivo:**

```python
SAMPLING_CONFIG = {
    "temperature": 0.80,       # rango [0.75, 0.85]
    "top_p": 0.70,
    "alpha_presence": 0.5,     # previene repetición de "che", "posta"
    "alpha_frequency": 0.4,
    "max_tokens": 120,         # cap para respuestas concisas en debate
}
```

### 4.4 Desacoplamiento ASR/TTS en CPU (para streaming en vivo)

Para mantener latencia < 1500ms con la RX 6500 XT (4GB dedicados al LLM):

| Componente | Hardware | Implementación recomendada |
|-----------|---------|--------------------------|
| RWKV-7 LLM | GPU ROCm | `rwkv_rocm.py` existente |
| ASR (Whisper) | CPU int8 | `faster-whisper` cuantizado; búfer circular de baja latencia |
| TTS | CPU ONNX | `Piper TTS` con modelo `HirCoir/piper-checkpoint-es-ar-elena` |

**Piper TTS** inicia la síntesis a partir de los primeros 50ms tras el primer chunk de texto — permite entrega casi instantánea mientras el LLM continúa generando.

---

## 5. Moderación y filtro de contenido

### Arquitectura del filtro pre-TTS (determinista, sin LLM)

```
RWKV output text
      ↓
 ContentFilter.check(text)
      ├── PASS → TTS pipeline → Audio output
      └── FAIL (nivel) →
            ├── HARD: bloqueo total, no emitir nada
            └── SOFT: reemplazar término con [BLEEP] → TTS → efecto narrativo cómico
```

**Nuevo archivo `kateto/safety/content_filter.py`:**

```python
class ContentFilter:
    """Filtro determinista pre-TTS. No usa LLM — lista negra léxica + regex."""
    
    HARD_BLOCK: frozenset  # discriminación, odio, sexual explícito
    SOFT_CENSOR: frozenset # lunfardo fuerte → [BLEEP]
    
    def check(self, text: str) -> tuple[str, bool]:
        """
        Retorna (texto_procesado, fue_modificado).
        Si HARD_BLOCK → retorna ("", True) para silencio total.
        """
        ...
```

### Tabla de moderación por categoría

| Categoría | Acción | Recurso narrativo |
|-----------|--------|-------------------|
| Discriminación / odio | BLOQUEO total — respuesta vacía | — |
| Vulgaridad grave | Reemplazo `[BLEEP]` + sonido de censura en TTS | El chat interpreta que Kateto "intentó algo grave" → momento cómico |
| Insultos moderados rioplatenses | ✅ Permitido (clasificación adulto) | Parte del personaje |
| Contenido sexual | Logit penalty en decodificación | Penalizar tokens del vocabulario sexual durante sampling |

---

## 6. Notebook Kaggle: entrenamiento 2xT4 (RWKV7-G1J 2.3B / 7B)

### Especificación del notebook

**Archivo a crear:** `notebooks/kateto_train_kaggle_2xt4.ipynb`  
**Hardware objetivo:** Kaggle — 2× NVIDIA Tesla T4 (16GB VRAM × 2 = 32GB total)  
**Acelerador de tiempo:** Kaggle ofrece ~30h/semana de T4 gratuito; ~40h con verificación de cuenta.

**Modelos objetivo:**
- `RWKV/rwkv-7-world-2.3b` o `aabbdev/RWKV7-2.3B-20260805` (HuggingFace)
- `RWKV/rwkv-7-world-7b` (si entra con LoRA r=8 en ZeRO-3)

### Distribución de carga en 2xT4

```python
# Estrategia: Data Parallelism con DDP (RWKV-7 no soporta tensor parallelism trivialmente)
import torch.distributed as dist

# Configuración para T4 (no NVLink, usa PCIe)
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Para RWKV7-2.3B con PiSSA r=32:
# Modelo completo: ~4.6GB + activaciones ~3GB = ~7.6GB/GPU → cabe en T4 con espacio
# Batch efectivo con grad_accum=8: batch_size=4 por GPU → batch efectivo=64

# Para RWKV7-7B con LoRA r=8:
# Modelo completo: ~14GB → requiere sharding entre las 2 T4
# Usar DeepSpeed ZeRO-3 o FSDP
```

### Estructura completa del notebook

#### Celda 1 — Setup y dependencias
```python
# !pip install rwkv-peft triton datasets wandb peft deepspeed
# Verificar GPU
import subprocess
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
```

#### Celda 2 — Configuración del modelo

```python
MODEL_SIZES = {
    "2.3b": {
        "hf_name": "aabbdev/RWKV7-2.3B-20260805",
        "n_layer": 32,
        "n_embd": 2560,
        "pissa_rank": 32,
        "expected_vram_gb": 7.6,
    },
    "7b": {
        "hf_name": "RWKV/rwkv-7-world-7b",  
        "n_layer": 32,
        "n_embd": 4096,
        "lora_rank": 8,        # LoRA en lugar de PiSSA para reducir VRAM
        "expected_vram_gb": 15.2,
        "requires_zero3": True,
    }
}
TARGET_MODEL = "2.3b"  # cambiar a "7b" para la variante grande
```

#### Celda 3 — Preparación del dataset

```python
from datasets import load_dataset, concatenate_datasets

# Cargar datasets curados
cordeba = load_dataset("marianbasti/cordeba", split="train")
anti_syco = load_dataset("DataCreatorAI/Anti-Sycophancy-DPO", split="train")
func_call = load_dataset("NickyNicky/function-calling-sharegpt_chatml_gemma_agent", split="train")

# Convertir todos al formato Kateto ChatML
# (scripts/convert_to_kateto_format.py)
```

#### Celda 4 — Sequence Packing

```python
from rwkv_pipeline.sequence_packer import SequencePacker

packer = SequencePacker(max_len=1024)  # contexto más largo en T4
packed_dataset = packer.pack_dataset(all_examples)
```

#### Celda 5 — Configuración PiSSA/LoRA

```python
from rwkv_pipeline.lora_module import PiSSAAdapter

model = load_rwkv7(MODEL_SIZES[TARGET_MODEL]["hf_name"])
adapter = PiSSAAdapter(
    model=model,
    rank=MODEL_SIZES[TARGET_MODEL]["pissa_rank"],
    niter=4,
    target_modules=["att.receptance", "att.key", "att.value", "att.output",
                    "ffn.key", "ffn.value", "ffn.receptance"]
)
print(f"Parámetros entrenables: {adapter.count_trainable():.2f}M")
```

#### Celda 6 — Entrenamiento SFT Capa 1

```python
trainer = KatetoTrainer(
    model=adapter,
    dataset=packed_dataset,
    config={
        "lr_init": 3e-4,
        "lr_final": 3e-5,
        "schedule": "cosine",
        "epochs": 3,
        "batch_size": 4,
        "grad_accum": 8,   # batch efectivo = 64
        "save_every": 500,
        "eval_every": 200,
    }
)
trainer.train()
```

#### Celda 7 — ORPO Anti-Sycophancy

```python
from rwkv_pipeline.orpo_trainer import ORPOTrainer

orpo_data = load_dataset_from_jsonl("data/orpo/anti_sycophancy_rioplatense.jsonl")

orpo_trainer = ORPOTrainer(
    model=adapter,
    dataset=orpo_data,
    config={"lambda_or": 0.1, "lr": 5e-5, "epochs": 2}
)
orpo_trainer.train()
```

#### Celda 8 — State Tuning por voz

```python
voices_to_train = ["jane", "doktor", "whisperer", "streamer"]

for voice in voices_to_train:
    voice_data = load_voice_dataset(voice)
    state = train_s0(
        model=base_model,   # pesos congelados
        dataset=voice_data,
        lr_init=1e-2,
        lr_final=1e-4,
        epochs=15,
        l2_gamma=5e-4,
    )
    save_state(state, f"out/rwkv_states/{voice}/rwkv-kaggle-2.3b.pth")
```

#### Celda 9 — Evaluación y Export

```python
# Test de tool-calling (tasa de JSON válido)
# Test de debate (sin <tool_call> en output)
# Test en che-boludo-benchmark (comprensión pragmática rioplatense)

# Export a Kaggle Dataset
import kaggle
kaggle.api.dataset_create_new(
    folder="out/",
    title="kateto-rwkv7-2.3b-kateto-trained",
    public=False
)
```

### Estimaciones de tiempo y VRAM (2xT4)

| Tarea | Modelo | Tiempo estimado | VRAM/GPU |
|-------|--------|----------------|----------|
| SFT Capa 1 (PiSSA r=32, 3 épocas) | 2.3B | ~4–6 horas | ~10 GB |
| ORPO anti-sycophancy (2 épocas) | 2.3B | ~2–3 horas | ~12 GB |
| State Tuning (5 voces, 15 épocas c/u) | 2.3B | ~2 horas | ~6 GB |
| SFT Capa 1 (LoRA r=8, 3 épocas) | 7B | ~10–14 horas | ~15 GB |
| State Tuning | 7B | ~3 horas | ~10 GB |

### Diferencias de capacidad esperadas

| Capacidad | 0.4B (local) | 2.3B (Kaggle) | 7B (Kaggle) |
|-----------|-------------|---------------|-------------|
| Coherencia en debate | ❌ Incoherente | 🟡 Aceptable | ✅ Buena |
| Tool-calling JSON | 🟡 Con errores | ✅ Preciso | ✅ Muy preciso |
| Humor contextual | ❌ No emerge | 🟡 Básico | ✅ Emergente |
| Voseo espontáneo | 🟡 Con training | ✅ Natural | ✅ Muy natural |
| Latencia (RX 6500 XT, 4GB) | ✅ ~0.3–0.5s | ❌ No corre | ❌ No corre |
| Latencia (T4 o cloud) | — | ✅ ~0.5s | 🟡 ~1.5s |

---

## 7. Roadmap de implementación

### Sprint 1 — Calidad inmediata en 0.4B local (1–2 días)

- [ ] Agregar `<tool_call>` como stop token en `rwkv_rocm.py:stream()` → fix inmediato en debates
- [ ] Cambiar `_format_messages_to_prompt` a tokens `<|im_user|>` / `<|im_start|>`
- [ ] Generar 500 ejemplos de debate speech con `scripts/generate_debate_examples.py`
- [ ] Verificar que `device=cuda` esté activo en próxima inferencia (ya corregido)

### Sprint 2 — Pipeline de entrenamiento mejorado (3–5 días)

- [ ] Implementar `SequencePacker` con `reset_mask` en `rwkv_pipeline/`
- [ ] Implementar loss masking estricto en `rwkv_pipeline/data_utils.py`
- [ ] Re-entrenar Capa 1 con datos de debate + CordeBA + loss mask
- [ ] State Tuning para voces `doktor` y `jane` (estado propio por voz)

### Sprint 3 — Alineación anti-sycophancy (3–4 días)

- [ ] Descargar datasets HF (Anti-Sycophancy-DPO, CordeBA, function-calling)
- [ ] Implementar `ORPO trainer` en `rwkv_pipeline/orpo_trainer.py`
- [ ] Curar ~200 pares de preferencia rioplatenses
- [ ] Entrenar ORPO sobre el modelo SFT del Sprint 2

### Sprint 4 — Escalado a modelos grandes en Kaggle (1–2 semanas)

- [ ] Crear `notebooks/kateto_train_kaggle_2xt4.ipynb` completo
- [ ] Descargar RWKV7-G1J-2.3B desde HuggingFace
- [ ] Entrenar SFT + ORPO + State Tuning en 2xT4
- [ ] Evaluar con che-boludo-benchmark y debate tests
- [ ] Exportar modelo y estados a Kaggle Dataset privado

### Sprint 5 — Infraestructura de inferencia (paralelo a Sprint 2–3)

- [ ] Integrar `Piper TTS` (offline, `es_AR-elena`) como alternativa offline a Boson
- [ ] Implementar `ContentFilter` pre-TTS en `kateto/safety/`
- [ ] Benchmark de latencia end-to-end: ASR → LLM → TTS

---

## 8. Decisiones de diseño y trade-offs

### ¿Por qué ORPO y no DPO?
DPO requiere dos instancias del modelo en memoria simultáneamente (política θ + referencia π_ref congelada). En 4GB VRAM → OOM inmediato incluso con el 0.4B. ORPO alcanza el mismo objetivo en una única pasada supervisada con consumo ~2.2GB.

### ¿Por qué State Tuning por voz y no system prompts?
Los system prompts consumen tokens del contexto limitado (ctx=512 para 0.4B). S₀ codifica la personalidad directamente en el estado recurrente inicial, sin costo en tokens de contexto. La diferenciación emerge a partir de ~15 épocas por voz.

### ¿Por qué PiSSA y no LoRA para el 2.3B en Kaggle?
PiSSA converge en la mitad de pasos porque comienza desde los componentes de mayor energía spectral del modelo. Para el 0.4B con datos limitados y GPU lenta, la diferencia es menor. Para el 2.3B en T4 donde cada época cuesta tiempo de GPU real, la eficiencia importa.

### ¿Por qué `<tool_call>` como stop token es un fix de emergencia, no la solución?
El modelo sigue generando el token `<tool_call>` — simplemente se detiene cuando aparece. El output antes del stop puede ser correcto o incoherente. La solución correcta es el entrenamiento con ejemplos donde los debates NO terminan con tool_calls (Sprint 1 — generar 500 ejemplos). Ambos fixes son necesarios en paralelo.

### ¿Por qué Piper TTS y no Boson API?
Piper TTS corre 100% offline en CPU con ONNX Runtime, latencia < 200ms con el modelo `es_AR-elena`, sin dependencia de servicio externo. Boson introduce latencia de red variable y dependencia de disponibilidad del API. Para un stream en vivo, la confiabilidad local es prioritaria.

### ¿Tiene sentido entrenar el 7B para uso en stream local?
No con la RX 6500 XT (4GB). El flujo productivo es: entrenar en Kaggle 2xT4 → alojar el 7B en instancia cloud (A100/H100 en RunPod) → exponer como API local sobre Kateto. El 0.4B se mantiene para prototipado y desarrollo local rápido.

### ¿Cómo prevenir el tool_call bleed en el dataset?
Tres estrategias combinadas:
1. **Loss masking:** no optimizar gradientes sobre tokens de usuario (previene memorización de patrones de input)
2. **Mezcla de datos:** el 40% del dataset debe ser ejemplos sin tool_calls (debate speech, conversación general)
3. **Stop token de emergencia:** detener generación si aparece `<tool_call>` durante inferencia

---

*Spec generada a partir de:*  
- `data/research/Optimización Agente Kateto RWKV7.txt`  
- `data/research/Argentine Conversational Fine-Tuning Datasets.txt`  
- `data/research/Entrenamiento IA Estilo Comediantes.txt`  

*Revisión pendiente: verificar compatibilidad del kernel Triton actual con el parámetro `reset_mask` antes de implementar SequencePacker (cambio en la firma del kernel puede romper la compilación en gfx1034).*
