# Diagnóstico y Solución: Falla Silenciosa de Merge LoRA en Qwen 3.5 (Unsloth / PEFT)

**Fecha:** 2026-09-05  
**Modelo:** `Qwen/Qwen3.5-0.8B`  
**Entorno:** AMD Radeon RX 6500 XT (Navi 24 / gfx1034), ROCm 6.3/7.2 (`HSA_OVERRIDE_GFX_VERSION=10.3.0`), Unsloth, PyTorch 2.9, llama.cpp.

---

## 1. Síntoma Reportado

Tras completar el entrenamiento de 2 etapas (Stage 1: adaptación a corpus argentino; Stage 2: SFT de Kateto con dialecto y tool calling) y exportar a formato GGUF (`out/qwen35-0.8b-kateto-Q4_K_M.gguf`), las respuestas del modelo en inferencia eran idénticas a las del modelo base de Alibaba, sin adoptar modismos argentinos ni la estructura de razonamiento (`[Start thinking]`, `<correct_response>`, `<answer>`).

> **Observación del usuario:** *"parece modelo base, verifica con un diff que sean modelos distintos"*.

---

## 2. Causa Raíz: Mapeo de Claves Incompatibles en PEFT

La exportación estándar recomendada para modelos PEFT es:

```python
# CÓDIGO PROBLEMÁTICO:
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, ...)
peft_m = PeftModel.from_pretrained(base, stage2_dir)
merged = peft_m.merge_and_unload()
merged.save_pretrained(merged_dir)
```

### ¿Por qué falló silenciosamente?

1. **Arquitectura en Unsloth:** Al entrenar Qwen 3.5 con Unsloth (`FastLanguageModel`), el modelo interno se encapsula como `Qwen3_5ForConditionalGeneration`. Las matrices LoRA generadas en `adapter_model.safetensors` tienen nombres con doble anidamiento de `model`:
   ```text
   base_model.model.model.language_model.layers.{i}.self_attn.q_proj.lora_A.weight
   base_model.model.model.language_model.layers.{i}.mlp.gate_proj.lora_A.weight
   ```

2. **Arquitectura al cargar con `AutoModelForCausalLM`:** Transformers carga la clase `Qwen3_5ForCausalLM`, cuyos pesos base tienen un solo `model`:
   ```text
   model.language_model.layers.{i}.self_attn.q_proj.weight
   model.language_model.layers.{i}.mlp.gate_proj.weight
   ```

3. **Comportamiento silencioso de PEFT:** `PeftModel.merge_and_unload()` itera sobre los módulos que reconoce. Al no coincidir los prefijos (`model.model.` vs `model.`), **no fusionó ni una sola de las 96 proyecciones entrenadas**. PEFT no arrojó ninguna advertencia ni error; simplemente descartó el adapter y guardó el 100% de los pesos originales del modelo base.

---

## 3. Solución: Fusión Matemática Directa de Tensores

Para garantizar la aplicación exacta e incondicional de los pesos entrenados, se implementó una rutina de fusión directa a nivel de tensores en `scripts/merge_and_diff_qwen.py` e integrada en `train_qwen_2stage.py`:

Para cada capa lineal proyectada ($W \in \mathbb{R}^{d_{out} \times d_{in}}$), la actualización de LoRA es:
$$\Delta W = (B \cdot A) \times \left(\frac{\alpha}{r}\right)$$
$$W_{\text{merged}} = W_{\text{base}} + \Delta W$$

Donde para nuestro modelo:
- $r = 16$
- $\alpha = 16$
- $\text{scaling} = \frac{\alpha}{r} = 1.0$

### Implementación del Merge Directo

```python
for k in sorted(adapter.keys()):
    if k.endswith(".lora_A.weight"):
        # Normalizar prefijo de clave eliminando el doble 'model.'
        clean_k = k.replace("base_model.model.", "")
        base_k = clean_k.replace(".lora_A.weight", "") + ".weight"
        
        lora_A = adapter[k].float()
        lora_B = adapter[k.replace(".lora_A.weight", ".lora_B.weight")].float()
        delta_W = (lora_B @ lora_A) * (alpha / r)
        
        orig_W = base_tensors[base_k]
        merged_tensors[base_k] = (orig_W.float() + delta_W).to(orig_W.dtype)
```

---

## 4. Protocolo de Verificación con Tensor Diff

Se compararon los tensores de `model.safetensors` del modelo base original (`Qwen/Qwen3.5-0.8B`) contra los tensores del modelo fusionado:

- **Total de tensores en el modelo:** 488 (incluye capas visuales y de lenguaje).
- **Total de tensores con pesos diferentes:** **96** (exactamente las proyecciones LoRA entrenadas: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` en las 24 capas del language model).
- **Tensores base no modificados:** 392 (embeddings, layernorms, vocab head, vision encoder congelado).

### Top 15 Tensores con Mayor Desviación ($\|\Delta W\|_2$)

| Tensor | Shape | Max $|\Delta W|$ | Mean $|\Delta W|$ | Norma $L_2$ |
| :--- | :--- | :--- | :--- | :--- |
| `model.language_model.layers.23.mlp.gate_proj.weight` | `[3584, 1024]` | 0.002747 | 0.000116 | **0.3276** |
| `model.language_model.layers.22.mlp.gate_proj.weight` | `[3584, 1024]` | 0.001709 | 0.000107 | **0.2888** |
| `model.language_model.layers.23.mlp.up_proj.weight` | `[3584, 1024]` | 0.002686 | 0.000107 | **0.2860** |
| `model.language_model.layers.23.self_attn.q_proj.weight` | `[4096, 1024]` | 0.001434 | 0.000098 | **0.2782** |
| `model.language_model.layers.17.mlp.gate_proj.weight` | `[3584, 1024]` | 0.001221 | 0.000105 | **0.2738** |
| `model.language_model.layers.19.self_attn.q_proj.weight` | `[4096, 1024]` | 0.001373 | 0.000097 | **0.2736** |
| `model.language_model.layers.21.mlp.gate_proj.weight` | `[3584, 1024]` | 0.001373 | 0.000103 | **0.2731** |
| `model.language_model.layers.20.mlp.gate_proj.weight` | `[3584, 1024]` | 0.001461 | 0.000102 | **0.2701** |
| `model.language_model.layers.18.mlp.gate_proj.weight` | `[3584, 1024]` | 0.001282 | 0.000102 | **0.2672** |
| `model.language_model.layers.22.mlp.up_proj.weight` | `[3584, 1024]` | 0.002743 | 0.000102 | **0.2656** |
| `model.language_model.layers.16.mlp.gate_proj.weight` | `[3584, 1024]` | 0.001290 | 0.000103 | **0.2648** |
| `model.language_model.layers.12.mlp.gate_proj.weight` | `[3584, 1024]` | 0.001038 | 0.000102 | **0.2608** |
| `model.language_model.layers.15.mlp.gate_proj.weight` | `[3584, 1024]` | 0.001221 | 0.000100 | **0.2578** |
| `model.language_model.layers.15.self_attn.q_proj.weight` | `[4096, 1024]` | 0.001099 | 0.000093 | **0.2575** |
| `model.language_model.layers.21.mlp.up_proj.weight` | `[3584, 1024]` | 0.001183 | 0.000100 | **0.2559** |

Las capas más altas (capas 16 a 23) presentan las mayores modificaciones, lo cual es coherente con el aprendizaje de estilo y alineación de diálogo en Stage 2.

---

## 5. Pipeline de Conversión y Cuantización

1. **Conversión a GGUF F16:**
   ```bash
   python /run/media/chaos/terciario/Backup/Proyectos/kateto/llama.cpp/convert_hf_to_gguf.py \
     out/qwen35-0.8b-kateto-merged \
     --outfile out/qwen35-0.8b-kateto-f16.gguf \
     --outtype f16
   ```
   *Tamaño:* 1,516 MB (~1.48 GB).

2. **Matriz de Importancia (`imatrix`):**
   ```bash
   llama-imatrix -m out/qwen35-0.8b-kateto-f16.gguf \
     -f data/sft_out/calib.txt \
     -o out/qwen35-0.8b-kateto.imatrix.dat \
     --chunks 64
   ```

3. **Cuantización Q4_K_M asistida por imatrix:**
   ```bash
   llama-quantize --imatrix out/qwen35-0.8b-kateto.imatrix.dat \
     out/qwen35-0.8b-kateto-f16.gguf \
     out/qwen35-0.8b-kateto-Q4_K_M.gguf \
     Q4_K_M
   ```
   *Tamaño final:* 529 MB (~504.8 MiB).

---

## 6. Verificación en Inferencia (`llama-cli`)

Para probar la inferencia por terminal sin que `llama-cli` quede esperando interacción en `stdin`, se deben pasar los flags `--single-turn` y `-r "<|im_end|>"`:

```bash
llama-cli \
  -m out/qwen35-0.8b-kateto-Q4_K_M.gguf \
  -p "<|im_start|>user\nChe boludo, quién sos y qué hacés acá?<|im_end|>\n<|im_start|>assistant\n" \
  -n 128 \
  --temp 0.6 \
  --single-turn \
  -r "<|im_end|>"
```

### Salida Obtenida
```text
[Start thinking]

<correct_response>
<answer>El boludo que te pide es el de dos personas...
[ Prompt: 172,4 t/s | Generation: 41,4 t/s ]
```

El modelo responde ahora con la estructura XML aprendida del dataset Kateto v2 (`[Start thinking]`, `<correct_response>`) y vocabulario rioplatense, confirmando que los pesos de LoRA están activos en el binario final GGUF.

---

## 7. Error Resuelto: `wrong number of tensors; expected 335, got 320`

### Causa
- El snapshot original de Hugging Face (`Qwen/Qwen3.5-0.8B`) incluye 24 capas de lenguaje (320 tensores) más 1 capa especulativa de Multi-Token Prediction (MTP / `mtp.layers.0.*`, 15 tensores adicionales), totalizando 335 tensores con `architectures: ["Qwen3_5ForConditionalGeneration"]`.
- Al convertir el snapshot base directamente a GGUF, se escribían 335 tensores con `block_count = 25`.
- Si se parcheaba manualmente el header binario a `block_count = 24`, `llama.cpp` solo cargaba las 24 capas principales (320 tensores) e ignoraba los 15 tensores MTP sobrantes. Al terminar de leer, comparaba contra el total declarado en la cabecera:
  ```text
  llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 335, got 320
  ```

### Solución
- Filtrar los 320 tensores puros del `model.language_model.*` y configurar la arquitectura como `Qwen3_5ForCausalLM` con `num_hidden_layers = 24`.
- `convert_hf_to_gguf.py` exporta limpiamente el archivo GGUF con `n_tensors = 320`, permitiendo que tanto el modelo base (`qwen35-0.8b-base-f16.gguf`) como los modelos Kateto se ejecuten sin incompatibilidades en `llama-cli`.

