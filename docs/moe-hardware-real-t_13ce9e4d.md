# MoE en el hardware real: veredicto, costo y prototipo de ruteo

## 1. Veredicto en 5 líneas (¿hay MoE en RWKV-PEFT? ¿entra un MoE acá?)

RWKV-PEFT **no tiene MoE** (T1: cero expertos, router, gate, ni args de experto en `train.py`). El 1.5B Q4 **entra completo en la dGPU** (943 MiB sobre 1.037 MiB libres) — la premisa "un modelo grande no cierra" es falsa al cuantizar. Partir capas a CPU **multiplica TTFT 5,7×–14×** (T3): no es pendiente, es precipicio. No hay disco escribible para un MoE real (≥4 GB) — espacio libre total 5,9 GB repartido en dos discos (T4). **Recomendación única: P3 (ruteo por turno entre 1.5B dGPU / 2.9B iGPU) para medir el ruteo + P4 (atacar TTS) como ataque real a latencia**. Evidencia que cambiaría esto: un MoE ≤2 GB en GGUF + corpus que no pierda coherencia.

## 2. La verificación pedida: RWKV-PEFT contra MoE (T1, con anclas archivo:línea)

- `grep -rniE 'moe|mixture|expert|routing|gate_net' rwkvt/ train.py main.py` → **0 coincidencias** (rc=1).
- Bloque denso puro: `rwkvt/rwkv7/block.py:18-19` → `self.att = RWKV_Tmix_v7(...)`, `self.ffn = RWKV_Cmix_v7(...)`.
- FFN es **un solo** par `key`/`value`: `rwkvt/rwkv7/ffn.py:35-36` → `self.key = nn.Linear(args.n_embd, args.n_embd*4)`, `self.value = nn.Linear(args.n_embd*4, args.n_embd)`. Sin dimensión de experto, top-k, ni balance loss.
- `train.py`: args son `--peft`, `--lora_config`, `--pissa_config`, `--miss_config`, `--train_parts`, etc. **Cero** args de experto/router.
- Módulos existentes: `rwkvt/{rwkv5,rwkv6,rwkv7}/{att,block,ffn,model}.py`, `infctx_module.py`, `dataset/`, `operator/rwkvop.py`, `peft_loading.py`, `trick/lrs.py`. No hay `moe.py` ni similar.

## 3. La premisa del dueño contra la medición (T3 + T4: tabla y las 3 lecturas)

### Tabla medida (llama-bench, 07:08–07:09, :8092 y :8200 sirviendo)

| dispositivo | ngl | pp128 t/s | TTFT128 (ms) | tg32 t/s |
|---|---|---|---|---|
| Vulkan0 = RX 6500 XT (dGPU) | 99 | 1547,73 | **83** | 84,31 |
| Vulkan0 | 24 | 1251,61 | 102 | 69,35 |
| Vulkan0 | 12 | 270,01 | 474 | 28,87 |
| Vulkan1 = Intel Iris Xe (iGPU) | 99 | 377,20 | 339 | 26,70 |
| CPU (ngl 0) | 0 | 109,06 | 1173 | 10,35 |

Modelo `out/kateto-rwkv29-Q4_K_M.gguf` = **rwkv7 2.9B** Q4_K_M, 1,75 GiB, en iGPU (Vulkan1, ngl 99): pp128 **190,48 t/s → TTFT128 672 ms**, tg32 19,07 t/s.

### Lecturas obligatorias

1. El 1.5B **entra completo** en la dGPU con los otros dos servidores ya ocupando 3,19 GB: 943 MiB sobre 1037 MiB libres. La premisa ("un solo modelo grande no cierra la cuenta") es falsa **al cuantizar**: lo que no entra es el 1.5B en f16 (3,07 GB).
2. Partir capas a CPU es un precipicio, no una pendiente: bajar de ngl 99 a ngl 12 multiplica el TTFT **5,7×** (83 → 474 ms); a CPU pura, **14×** (1173 ms). "1.5B en GPU + resto en CPU" no baja latencia: la sube.
3. La iGPU (4024 MiB, 3621 libres) es el único lugar donde hoy vive algo más grande que el 1.5B Q4: el 2.9B a 672 ms de TTFT.

### Holgura real del host (T4, 07:08)

- VRAM dGPU: 4080 MiB total, **1037 MiB libres** (ocupan gemma-4-E2B en :8092 y el 2.9B en :8200).
- VRAM iGPU: 4024 MiB total, **3621 MiB libres**.
- RAM: 31,9 GB total, **24,8 GB usados**, ~7 GB disponibles (1 GB libre + 6,5 GB cache).
- Disco: `/` 4,0 GB libres; `/run/media/chaos/secundario` 1,9 GB libres; `/run/media/chaos/terciario` **read-only** (86 GB pero no escribible).
- Consecuencia dura: **no hay ningún MoE en disco y no entra uno**. El MoE Q4 más chico usable (OLMoE-1B-7B, Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite) pesa entre ~4 y ~10 GB; el espacio libre total escribible es 5,9 GB y repartido en dos discos. Medir un MoE de verdad hoy es imposible sin liberar disco.

## 4. Los cuatro caminos y su costo estimado

| camino | qué es | toca pesos? | costo estimado | qué mide el gate de coherencia | riesgo |
|---|---|---|---|---|---|
| **P1 — MoE nativo en RWKV-PEFT** | Agregar `RWKV_CMix_MoE` en `rwkvt/rwkv7/ffn.py` (N expertos + router lineal + top-k + balance loss), elegir bloque en `block.py`, adaptar `infctx_module.py` (estado por experto), `RWKVConfig`/`to_dict`, export GGUF (op ggml nuevo) | Sí (arquitectura + pesos) | 2–4 sem de ingeniería + entrenamiento desde cero (router nace sin señal; fine-tune no alcanza) | Coherencia-larga contra baseline 30,0% **no comparable** (cambia arquitectura) | Alto: métrica de la card no se puede comparar; export GGUF inexistente |
| **P2 — MoE de terceros en llama.cpp `-ncmoe`** | Usar `-cmoe`/`-ncmoe` de llama.cpp 0.6.0-dev (T2) con modelos MoE GGUF | No (solo runtime) | 4–10 GB disco que **no hay** (T4); expertos en RAM, prefill atado a ancho de banda memoria | Gate coherencia-larga **pierde la persona Kateto** (mide otro modelo) | Medio, **bloqueado por disco** |
| **P3 — Ruteo por turno entre modelos en disco (prototipo)** | Política determinista: barato=1.5B Q4 dGPU (83 ms TTFT), caro=2.9B Q4 iGPU (672 ms TTFT). Umbral chars + marcadores razonamiento | No | Horas, no semanas | Lo que mide el script §2: `t_decision`, `ttft`, `total`, tokens, guardas | Bajo: mide costo real de decisión + p50 punta a punta |
| **P4 — No hacer esto y atacar T5** | TTS `piper` tarda 4,02 s (frío) / 5,53 s (caliente) por frase corta; RTF 0,17–0,22. Modelo aporta 83 ms de 4–5 s (~4%) | No (infra TTS) | El de la card de TTS | Latencia real de primer audio, no TTFT de LLM | Ninguno: ataca el cuello de botella real |

## 5. El prototipo de ruteo: qué mide y cómo (T5 + script §2)

El script `scripts/router_hardware.py` implementa:

- **Config JSON** (escribe el runner): tiers `barato` (1.5B dGPU, costo_rel=1.0) y `caro` (2.9B iGPU, costo_rel=8.0), prompts, `umbral_chars=120`.
- **Política `route(prompt, umbral_chars)`** determinista: "caro" si `len(prompt) > umbral_chars` **o** hay marcador razonamiento (`explica`, `por que`, `por qué`, `compara`, `analiza`, `demostra`, `codigo` con/sin tilde); si no, "barato". Cero aleatoriedad, cero llamada a LLM.
- **`--bench`**: mide con `time.perf_counter()` + `urllib.request` contra `/v1/chat/completions` stream=true. Registra: `t_decision_ms`, `ttft_ms` (primer delta con contenido; chunk vacío/rol no cuenta), `total_ms`, `n_tokens`, `tier`, `motivo`. POST estándar sin `tools` ni `seed`. Agrega p50/p95 por tier y global.
- **`--mock`**: mismo camino de código, transporte falso en proceso (8 chunks SSE sintéticos, `sleep(0.001)`), para agregar sin servidores.
- **`--selfcheck`** (sin red/modelos): 6 casos — (1) corto→barato; (2) 500 chars→caro; (3) "che, explicame por qué..."→caro por marcador; (4) determinismo; (5) percentiles exactos (p50 de [10,20,30,40,50]=30); (6) guarda falla con `n_tokens=0`. Imprime `selfcheck OK: 6/6` y exit 0.

## 6. Lo que NO se pudo medir y qué lo destraba

- **(a) Ningún MoE corrió en este hardware** — no hay GGUF MoE ni disco para bajarlo (T4: 5,9 GB libres totales, MoE Q4 mínimo ~4 GB, pero repartido en dos discos; `/terciario` read-only).
- **(b) Coherencia-larga no se midió** — no hubo cambio de pesos, así que las "3 seeds" y el "anti-hack de −5%" de la card **no aplican**; hay que decirlo explícito.
- **(c) Qué destraba cada cosa**: liberar **≥6 GB en un solo disco escribible** → bajar OLMoE-1B-7B Q4 y medir `-ncmoe` real; o conseguir MoE ≤2 GB GGUF que quepa en iGPU (3,6 GiB libres).

## 7. No verificado / límites

- Número de capas del 1.5B y cuántas entran por GiB (no medimos capa por capa).
- TTFT medido con `-p 128` **no** es el de un turno de voz real (trae audio + historial).
- Host cargado: 8 núcleos con otros 3 workers y 2 llama-server; números de CPU y iGPU son **pesimistas**.
- El script `--bench` real necesita servidores vivos en las `base_url` del config; `--mock` valida la lógica sin ellos.
- No se midió `t_decision_ms` bajo carga real de CPU (host saturado).

## 8. Gates corridos (comandos literales + salida real)

```bash
# Selfcheck (sin red, sin modelos)
python3 scripts/router_hardware.py --selfcheck
# selfcheck OK: 6/6

# Mock bench (valida agregado y guardas sin servidores)
python3 scripts/router_hardware.py --bench --config /tmp/router-t_13ce9e4d.json --mock --n 3 \
  --out out/moe-t_13ce9e4d/banco-mock.json --md out/moe-t_13ce9e4d/banco-mock.md
# banco-mock.json con "n": 3 y "guardas": {"ok": true}

# Tests pytest
python3 -m pytest tests/test_router_hardware.py -q
# 5 passed en <1s
```