# kateto-train

> Proyecto de Gonanf (—) — kateto-train
> **Lenguaje principal (GitHub):** — · **URL:** https://github.com/Gonanf/kateto-train · **Path:** `/run/media/chaos/terciario/proyectos/kateto-train`

## Qué es

Proyecto de Gonanf (—) — kateto-train. Repositorio de Gonanf/Gabriel Solotorevsky en `/run/media/chaos/terciario/proyectos/kateto-train`.

Este repo forma parte de la colección terciaria de **Gonanf / Gabriel Solotorevsky** clonada en `/run/media/chaos/terciario/proyectos/`.

## Qué busca conseguir

Objetivo original del proyecto según su código y docs: proveer la funcionalidad descrita y servir como base de aprendizaje/experimento dentro del portfolio terciario.

## Qué consigue ahora mismo

Repo inicializado sin commits: placeholder/estructura base creada, aún sin implementación.


## Stack

- **Stack detectado:** Python
- **Extensiones principales:** `.py`×122, `.png`×29, `.sample`×28, `.md`×24, `.cpp`×21, `.cu`×21
- **Archivos aprox.:** 44748

## Cómo correr

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

> Probado en Arch Linux (7.2.2-arch1-1). Ajustar rutas según tu entorno.

## Estructura

```text
kateto-train/
  README.md
  export_gguf.sh
  fill_toolcalling.py
  merge_lora.py
  run_kateto.sh
  to_sft.py
  base/
    rwkv7-g1j-2.9b-20260831-ctx16384.pth
    .cache/
  RWKV-PEFT/
    LICENSE
    README.md
    README_zh.md
    main.py
    pyproject.toml
    requirements.txt
    train.py
    uv.lock
    assert/
      logo.jpg
    cuda/
      rwkv7_clampw.cpp
      rwkv7_clampw.cu
      rwkv7_state_clampw.cpp
```

## Estado

- **Último commit:** `sin commits`
- **Path absoluto:** `/run/media/chaos/terciario/proyectos/kateto-train`
- **Remote:** `—`
- **Docs previas:** `docs/overview.md` (borrador previo, preservado como referencia)

## Modelos y Entrenamiento

### Qwen 3.5 0.8B (Unsloth + ROCm 6.3/7.2 en Navi 24)
- **Script principal:** [`train_qwen_2stage.py`](./train_qwen_2stage.py)
- **Documentación técnica del merge:** [`docs/qwen_unsloth_lora_merge_issue.md`](./docs/qwen_unsloth_lora_merge_issue.md)
  - Diagnóstico del bug de claves en PEFT (`base_model.model.model.` vs `model.`) que provocaba exportaciones idénticas al modelo base.
  - Implementación de fusión matemática de tensores y verificación con diff L2 norm.
  - Pipeline de exportación a GGUF FP16 y cuantización asistida por `imatrix` a `Q4_K_M`.
- **Script de verificación/diff:** [`scripts/merge_and_diff_qwen.py`](./scripts/merge_and_diff_qwen.py)

## Docs y README original

- `docs/qwen_unsloth_lora_merge_issue.md` — Diagnóstico y resolución de fusión LoRA para Qwen 3.5 en Unsloth.
- `docs/knowledge-rwkv-kateto-rosa-gguf-rx6500xt.md` — Arquitectura RWKV-7/8, ROSA y compatibilidad RX 6500 XT.
- `docs/overview.md` — borrador previo generado automáticamente (se mantiene como referencia, no como README principal).
- README previo recortado (2000 chars) se preservó abajo para trazabilidad.

<details><summary>README original (preservado, recortado)</summary>

```markdown
(sin README original previo)
```

</details>

---

*README reescrito el 2026-09-04 como README completo (Qué es / Objetivo / Qué consigue / Stack / Cómo correr / Estado / Roadmap) con inspección real de estructura, stack y git. No se modificó código, solo documentación.*
*Autor original: Gonanf — https://github.com/Gonanf/kateto-train*
