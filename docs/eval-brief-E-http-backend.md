# Brief Eval-E — Backend HTTP (llama.cpp) para medir el 2.9B

> **Repo:** `/run/media/chaos/terciario/proyectos/kateto-train`
> **Salida:** `scripts/eval_http_backend.py` + `scripts/eval_generation.py` con `--backend http`

## Por qué existe este brief (no lo re-litigues)

El **2.9B no se puede medir con el path local**. Hechos medidos:
- VRAM total de la maquina: **3.98 GiB**. Un 2.9B en bf16 son **5.9 GB** -> no entra.
- El operador RWKV7 de RWKV-PEFT (`RWKV-PEFT/rwkvt/operator/rwkvop.py`) tiene **tres ramas**: `fla`,
  `triton`, y un `else` que compila CUDA por JIT. **No hay path de CPU.**
- `scripts/eval_generation.py` tiene `load_student(..., device="cuda")` hardcodeado.

La salida: los **GGUF servidos por el router llama.cpp en `http://127.0.0.1:11434`**, que corren en CPU.
Par ya verificado, **ambos en Q4_K_M** (mismo quant: no se confunde cuantizacion con fine-tuning):

- BASE      -> preset `RWKV-2.9B`            (`zhiyuan8/RWKV-v7-2.9B-G1-GGUF:Q4_K_M`)
- CANDIDATO -> preset `RWKV7-Kateto-Retrain` (`out/kateto-rwkv29-Q4_K_M.gguf`)

**§2.1 (CBB, preferencia por PMI) NO es corrible por esta via** y no hay que intentarlo: el router no
puede scorear una continuacion arbitraria (`echo:true`, `max_tokens:0` y `n_probs` ignoran el prompt).
Solo se miden las metricas de **generacion**: §2.2, §2.3, §2.4, §2.5, §2.8. Declaralo en `notes`.

## REGLA 0 — PROHIBIDO EXPLORAR

**No lances subagentes. No hagas reconocimiento. No escribas un plan en prosa.** Escribí los archivos y
corré el smoke test. Tres corridas anteriores se quemaron explorando y entregaron cero archivos.

## Qué construir

### 1. `scripts/eval_http_backend.py`

```python
def generate_http(base_url: str, model: str, prompt_text: str, *,
                  max_len: int = 64, temperature: float = 0.80,
                  top_p: float = 0.70, seed: int = 1337,
                  stop: list[str] | None = None, timeout_s: int = 900) -> dict:
    """Devuelve {"text": str, "n_tokens": int, "stopping_word": str|None,
                 "stopped": bool, "model": str, "raw": {...}}"""
```

- POST a `{base_url}/completion` (endpoint **nativo** de llama.cpp, no el de chat: necesitamos mandar
  el prompt YA formateado).
- Payload minimo: `{"prompt": prompt_text, "n_predict": max_len, "temperature": ...,
  "top_p": ..., "seed": ..., "stream": false, "cache_prompt": false}` mas `"stop": [...]` si viene.
  **`cache_prompt: false` es obligatorio**: con el cache activo una generacion puede reusar el estado de
  la anterior y contaminar la medicion.
- Leé `content` (texto), `tokens_predicted` (`n_tokens`), `stopping_word`, `stop_type`.
- **Timeout generoso por default**: los modelos arrancan en frio y la primera llamada tarda minutos.
  Una llamada que tarda 90 s no es un fallo.
- **Sin `try/except` que se coma el error**: si la respuesta no es 200, levantar con el body incluido.

### 2. Cablear en `scripts/eval_generation.py`

Agregá `--backend {local,http}` (default `local`, para no romper lo ya corrido), `--base-url`
(default `http://127.0.0.1:11434`) y `--model`. Con `--backend http`:
- no cargues el .pth ni el tokenizer; generá con `generate_http`;
- pasá como `stop` los marcadores del formato que ya tenés en `STOP_MARKERS` (llama.cpp corta nativo,
  mejor que el truncado a posteriori que igual queda como red de seguridad);
- el resto del pipeline (métricas §2.2/§2.3/§2.4/§2.5/§2.8, `raw_responses`, truncado) **no se toca**.
- `n_tokens_truncated` con backend http: usá `len(text_cut)` en caracteres si no hay tokenizer, y
  registralo distinto (`n_chars_truncated`) para no mezclar unidades. Decilo en `notes`.

## Verificación obligatoria antes de terminar

1. **Smoke de 4 items con `--backend http --model RWKV-2.9B`**: pegá el comando y el resumen. Confirmá
   que `raw_responses` tiene 4 entradas con **texto no vacío** y `n_tokens > 0`.
2. **Smoke de 4 items con `--model RWKV7-Kateto-Retrain`**, ídem.
3. **Determinismo**: corré el smoke 2 veces con el mismo `--seed` y `--model`; `n_tokens` por item debe
   coincidir. Si no, declaralo en `notes` (el sampler de llama.cpp puede no ser determinista entre
   corridas por batching — si pasa, decilo, no lo escondas).
4. **Los dos modelos tienen que cargar.** Si la primera llamada devuelve error de modelo, esperá a que
   el router lo cargue (mirá `GET {base_url}/v1/models` y el campo `status.value`) y reintentá. **No hay
   endpoint de unload**: cargar el segundo puede desalojar el primero (`--models-max`), asi que corré los
   dos smokes **en secuencia, no en paralelo**.

## Restricciones de entorno (ya sufridas)

- **No** hace falta `HSA_OVERRIDE_GFX_VERSION` ni `systemd-run` para el path HTTP: el router ya corre
  aparte y sirve en CPU. Sí mantené `systemd-run --user --scope -p MemoryMax=infinity --unit=<n>` para
  el proceso python, por consistencia con el resto.
- No toques `out/eval/prompts_v1.json` ni `config/behavior_patterns.yaml`: están congelados.
- No toques el `.pth` local ni el camino `local`: sigue funcionando y ya dio resultados.

## NO hagas

- No intentes §2.1 (CBB/PMI) por HTTP: no se puede, está explicado arriba.
- No corras los 80 items todavía: este brief es el backend + el smoke. El run completo lo dispara el
  orquestador despues de revisar los smokes.
- No midas §2.6 (separacion de voces) ni §2.7 acá: van en briefs aparte.
