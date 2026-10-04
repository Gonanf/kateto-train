# Eval HTTP Backend — Notas

## §2.1 (CBB/PMI) no es corrible por HTTP

El router llama.cpp no puede scorear una continuación arbitraria. Los flags
`echo:true`, `max_tokens:0` y `n_probs` son ignorados por el endpoint
`/completion`. Solo se miden las métricas de generación: §2.2, §2.3, §2.4,
§2.5, §2.8.

## n_tokens_truncated vs n_chars_truncated

Con backend http no hay tokenizer disponible server-side. Se usa `len(text_cut)`
en caracteres para `n_chars_truncated`. Las métricas §2.4 (length distribution)
usan `n_tokens` del conteo de tokens de llama.cpp como fallback, no chars.
Las unidades están mezcladas en el JSON de salida — declarado acá.

## Determinismo

Confirmado: con `--seed 1337` idéntico, los dos smokes (RWKV-2.9B) produjeron
`n_tokens` idénticos y texto idéntico en las 4 items. El sampler de llama.cpp
es determinista con el mismo seed para el mismo modelo en secuencial.

## Model loading

El router carga modelos bajo demanda (first completion request triggers load).
La primera llamada puede tardar minutos. `wait_for_model()` verifica que el
preset existe, no que esté loaded — el load ocurre en `generate_http()`.

## Archivos generados

- `scripts/eval_http_backend.py` — módulo HTTP backend + smoke CLI
- `scripts/eval_generation.py` — modificado con `--backend {local,http}`
- `out/eval/run1/gen_http_rwkvb_smoke.json` — RWKV-2.9B smoke (4 items)
- `out/eval/run1/gen_http_kateto_smoke.json` — RWKV7-Kateto-Retrain smoke (4 items)
- `out/eval/run1/gen_http_determinism_{1,2}.json` — determinism check
