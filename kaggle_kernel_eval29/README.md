# kaggle_kernel_eval29

Suite de evaluación de RWKV 2.9B merged en Kaggle (2x T4).

## Push a Kaggle
```bash
kaggle kernels push -p kaggle_kernel_eval29
```

## Salidas (`out-eval29/`)
- `manifest.json`: entorno, hashes y estado de stages.
- `pmi29_{chatml,world}.json` y `*.jsonl`: CBB PMI por formato.
- `behavior29/probes.json`: sondas A/B probes y cbb.
- `gen29_rows.jsonl`: generaciones in-process con TTFAB.
- `gen29_metrics_{chatml,world}.json`, `SUMMARY.md`: métricas y resumen.
