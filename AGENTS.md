# AGENTS.md — kateto-train

Repo de **entrenamiento y evaluación** de Korat (modelo) y las voces de Kateto.
No es un repo de aplicación: acá viven los datasets, el pipeline de entrenamiento,
la receta de Kaggle, y los scripts de medición.

---

## FUENTES DE LA VERDAD (4, y solo 4)

Cada cosa tiene UN lugar canónico. Si un dato vive en dos lados, uno de los dos es basura.
Nunca duplicar: o se enlaza, o se borra.

| # | Fuente | Qué contiene | Regla dura |
|---|---|---|---|
| 1 | **HF `Gonanf/Korat`** | Safetensors + configs del modelo | **UN solo archivo de pesos (`model.safetensors`). NO versiones ni alternos.** Múltiples archivos OK (config.json, tokenizer, etc.) pero solo 1 modelo. Al actualizar: commit nuevo que reemplaza, no suma. |
| 2 | **HF `Gonanf/Korat-gguf`** | GGUF (F16, Q8_0, Q6_K) | Mismo esquema que #1. 1 modelo, commits que reemplazan. |
| 3 | **HF `Gonanf/Datasets`** | 6 categorías separadas | **Nunca** datos de clientes (workmatch/sherut). Revisar nombres/claves privadas antes de subir. Commit reemplaza, no acumula. |
| 4 | **GitHub `Korat-Training`** (este repo) | Scripts de dataset, docs, receta Kaggle, tests | Ver estructura obligatoria abajo. |

---

## CATEGORÍAS DE DATASET (fuente de verdad para Datasets #3)

Exactamente estas seis, y se suben por separado:

1. `chistes`
2. `hipotenusa`
3. `conversacion`
4. `roleplaying`
5. `conocimiento`
6. `streaming`

No existe "dataset único" como fuente: el unificado se **reconstruye** desde los seis
(combinación reproducible, mismo sha256). Un turno nuevo entra a una categoría; nunca
a un archivo suelto.

**Gates obligatorios antes de subir** (aplican a los seis por igual, no se relajan):
- formato canónico ChatML Kateto (`rwkv_pipeline/chat_template.py`)
- cero marcas comerciales
- cero datos de clientes
- cero nombres de archivos, rutas del SO o claves privadas

---

## PROMPTS

**TODOS los prompts van en `.omo/` (nunca en otro lado).** `.omo/` está en `.gitignore`.
No hay prompts en `scripts/`, `docs/`, ni en el código.

---

## ESTRUCTURA OBLIGATORIA DEL REPO (fuente #4)

```
kateto-train/
├── AGENTS.md              # este archivo
├── README.md              # apunta a las 4 fuentes; no repite contenido
├── SPEC.md                # contrato de comportamiento
├── docs/                  # documentación de decisiones (1 solo lugar)
├── scripts/               # 1 sola carpeta de scripts
├── tests/                 # 1 sola carpeta de tests
├── kaggle/
│   └── <UN solo recipe>   # 1 receta de Kaggle, con su kernel-metadata.json
├── data/                  # datasets de trabajo (nunca versionados los .jsonl grandes)
└── out/                   # artefactos de corrida (nunca versionados)
```

**PROHIBIDO en este repo** (si existe, se quita o se mueve al lugar único):
- más de una carpeta de scripts (`script/`, `tools/`, `evals/` como carpetas hermanas)
- más de una carpeta de tests
- más de un notebook/recipe de Kaggle
- más de una carpeta de docs (`notes/` vs `docs/`)
- más de una carpeta de pipelines (`pipelines/` vs `rwkv_pipeline/` vs `script/`)
- código de terceros vendorado suelto (`RWKV-LM/`, `RWKV-PEFT/`, `venv*/`) como carpeta
  versionable — el entrenamiento lo clona/apunta por variable de entorno, no lo copia

---

## ARQUITECTURA SEGÚN EUREKA (no el pipeline viejo)

| Componente | Estado real (medido) |
|---|---|
| **Modelo base** | RWKV-7 1.5B World (embed 2048). El "V7" real entrenado es 1.5B (`kateto-15b-Q4_0.gguf`), **NO** 2.9B. |
| **Entrenamiento V7 real** | `out/kateto-15b-Q4_0.gguf` = checkpoint `orpo-v7-c256-1.5b.pth` (PiSSA r=16 sobre SFT, λ=0.05, ctx 256, 1 época de 305 pasos). |
| **Serving** | 1.5B en llama.cpp + template ChatML (`korat-chat/kateto_rwkv.jinja`, `repeat_penalty 1.30`). |
| **State-Tuning (Capa 2)** | Style tuning: entrena SOLO S₀, 100% pesos congelados. 5 voces (`seco`, `streamer`, `jane`, `doktor`, `whisperer` + `none`). Config en `config/state_tuning.yaml`. |
| **Métodos siguientes (Eureka cards)** | MoE-LoRA (t_a7558e58) · RWKV-Infer+MRSS pseudo-MoE por gating (t_e4af7cc1) · Meta-State (t_6a54b99d) · Arquitectura propia RWKV-8 (t_e2e0c575). |
| **NO se usa** | RL/RLVR (no hay reward model), DPO puro (ORPO lo reemplaza), cuantización como parte del entrenamiento. |

---

## VERIFICAR ANTES DE TOCAR

- **`kateto-rwkv29-Q4_K_M-v7.gguf` es FALSO**: mismo md5 que el base 2.9B sin entrenar. El modelo entrenado de verdad es `kateto-15b-Q4_0.gguf`. No volver a servir el 2.9B creyendo que es el entrenado.
- **Dataset 2.9B está contaminado**: 7.5% `<tool_call>` del agente Hermes, 7.2% `<tool_call>`, 4.7% rutas del SO. Es fallo del dataset, no del sampler. No reentrenar antes de arreglarlo.
- **Veredicto coherencia 1.5B ORPO v7 = NO CUMPLE** (0/3 escenarios, colapso a `<|wait|>`, repetición literal). Ver `out/coherencia-v7/VEREDICTO-t_b704afcf.md`.

---

## ANTI-PATTERNS (ESTE REPO)

- No subir a HF un archivo nuevo sin reemplazar el anterior (prohibido `korat.pth` + `korat_v2.pth`).
- No subir datos de clientes (workmatch/sherut) a `Datasets`.
- No tener más de una carpeta de scripts / tests / notebooks.
- No vendorar `RWKV-PEFT` / `RWKV-LM` / `venv*` como carpetas versionables.
- No servir el 2.9B creyendo que es el entrenado.
- No relajar los gates de formato para que entre una categoría nueva.
- No reentrenar sin mirar el veredicto de coherencia del modelo anterior.