# MoE-LoRA (RWKV-LM-RLHF) sobre RWKV-7 1.5B en Kaggle

## Estructura de la carpeta
- `build_sft_text.py`: Convierte dataset de preferencias v8 a formato SFT jsonl para h5.
- `param_math.py`: Aritmética de parámetros entrenables y control honesto de presupuesto.
- `kernel/kateto-moe-lora-1xt4.ipynb`: Notebook Kaggle autocontenido (6 celdas código + 1 md).
- `kernel/kernel-metadata.json`: Metadata Kaggle (GPU T4, datasets `kateto-v7-sft` y `kateto-orpo-v8`).

## Pin del repositorio
`https://github.com/OpenMOSE/RWKV-LM-RLHF` en commit `f58064b059c0dd4d29e7723efccab1598d66a6e7`.

## Despliegue en Kaggle
```bash
kaggle kernels push -p moe_rlhf/kernel
```

## Artefactos esperados
- `moe_verdict.json`: métricas (`rc`, `steps_logueados`, `wall_s`, `s_por_paso_real`, `n_loss_nan`).
- `moe-out/*.pth`: checkpoints con adaptadores LoRA y router MoE.
- `train_moe.log`: log completo stdout/stderr del proceso de entrenamiento.

## Presupuesto de parámetros (L=24, D=2048, FFN=8192)
```
Configuración                      | LoRA Params    | Router   | Total Params
-----------------------------------+----------------+----------+-------------
LoRA dense r=16 (baseline v7)      | 7,864,320      | 0        | 7,864,320
LoRA dense r=32                    | 15,728,640     | 0        | 15,728,640
LoRA dense r=64                    | 31,457,280     | 0        | 31,457,280
LoRA dense r=128 (control honesto) | 62,914,560     | 0        | 62,914,560
MoE E=4 active=2 r=16              | 31,457,280     | 147,456  | 31,604,736
MoE E=4 active=2 r=32              | 62,914,560     | 147,456  | 63,062,016
```
Control honesto para MoE E=4 r=32: LoRA dense r=128 (router agrega 147,456 params, +0.23%).
