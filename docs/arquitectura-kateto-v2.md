# Arquitectura Kateto v2 — RWKV7 + ROSA + OP/ORPO

**OK arquitectura v2**

## Decisiones finales

### Checkpoint base
**Usar:** `out/rwkv_kateto_base_plus_1.5b/rwkv-0.pth`
Justificación: El base_plus 0.4B queda corto para podcast + ROSA retrieval. El 1.5B full ya está entrenado en out/rwkv_kateto_base_plus_1.5b, cabe en Kaggle 2×T4 ~16s/paso con ctx512, y mantiene compatibilidad con los state vectors de Capa 2. No se usa el 0.4B.

### ROSA retrieval_dim
**256** (actual 128)
Justificación: Capacidad asociativa para nombres de funciones, tokens exactos y JSON. 128 funciona en smoke pero satura en multiturno largo. 256 duplica parámetros de out_head (~16.8M vs 8.4M) y sigue <100MB. Gradientes verificados en smoke 10 pasos con rosa_grad_norm >0.

### OP teacher
**Mantener space-bunny-free (opencode) para v2**
Justificación: El pipeline OP actual depende del teacher remoto para generar pares de preferencia. Migrar a teacher local implica re-entrenar el oráculo y recalcular pares; queda como tarea post-v2. Se mantiene teacher actual con límite de tasa vía freellmapi.

### ORPO β
**0.1 (default)**
Justificación: Valor SPEC §2.5. Smoke ORPO muestra pérdida finita y orden chosen>rejected. Tunear β sin ablación aumenta riesgo de colapso de SFT. Se congela para v2.

### Context length
**512**
Justificación: 512 es el punto de equilibrio Kaggle 2×T4 ~16s/paso. 1024/2048 duplican VRAM y tiempo, y los podcasts se chunken en 9-12s (~150 tokens). Se mantiene ctx512 para consistencia con datos existentes.

### Vocab
**65536**
Justificación: Vocab de RWKV-7. Cobertura rioplatense suficiente con subword. Expandir requiere re-tokenizar todo el corpus y romper compatibilidad con checkpoints. No.

### LoRA / Fine-tune
**PiSSA rank 32, target modules att.*, ffn.***
Justificación: SPEC §2.2. LoRA r16 converge lento en tool-calling JSON. PiSSA inicia desde componentes principales y alcanza convergencia en 4-5 épocas vs 8. Full fine-tune no cabe en 4GB local; PiSSA r32 es el compromiso.

### Scheduler
**Cosine + warmup**
Justificación: Ya validado en Capa 1. No se cambia.

## Configuración unificada
Ver `config/kateto_v2.yaml`.

## Pipeline de entrenamiento
`scripts/entrenar-kateto-v2.sh` orquesta:
1. Carga base 1.5B + PiSSA r32
2. SFT con SequencePacker + loss-mask QA solo asistente
3. ROSA head retrieval_dim 256 entrenable, mezcla logits + gate*rosa_logits
4. ORPO alignment λ_or 0.1 sobre pares anti-sycophancy

## Gates verificados
- Tests ROSA: `pytest tests/test_rosa_*.py` → 29 passed
- Smoke ROSA 10 pasos: rosa_grad_norm >0, loss decreciente 6.995 → 6.920, VERDICT OK
- Doc firmada: OK arquitectura v2

## Riesgos
- step_infer preallocación: cat por token → O(T²). Arreglar antes de inferencia de producción.
- ROSA no serializa memoria de runtime. Solución en backlog.
- Teacher remoto: dependencia externa. Plan de migración a local post-v2.

Firmado: Kateto
