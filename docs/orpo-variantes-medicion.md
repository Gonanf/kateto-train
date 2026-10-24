# Medición de Variantes ORPO: RWKV-PEFT vs RWKV-LM-RLHF vs Baseline SFT

## Tabla de Diseño

El banco evalúa comparativamente el impacto en el gradiente y la preferencia entre la formulación canónica de RWKV-PEFT y la de RWKV-LM-RLHF frente a un baseline SFT sin término de preferencia, manteniendo el cálculo de Cross-Entropy (CE) idéntico sobre los tokens de respuesta del `chosen`.

| Brazo | Función de Pérdida ($L$) | Término de Preferencia ($L_{\text{OR}}$) | Coeficiente y Default | Archivos y Líneas de Origen |
|---|---|---|---|---|
| **`sft`** | $L = \text{CE}_w$ | N/A ($0.0$) | N/A | Baseline de control sin término de preferencia |
| **`ours`** | $L = \text{CE}_w + c_l \cdot (s_l - s_w)$ | $c_l \cdot (s_l - s_w)$, con $c_l = \text{clamp}(\lambda \cdot \sigma(s_l - s_w), 0, \lambda)$ detached | $\lambda = 0.1$ (aditivo) | `RWKV-PEFT/rwkvt/lightning_train/light_rwkv.py:80-89` (`_logp_medio`), `:92-99` (`_a_log_odds`), `:288-370` (`_orpo_step`), `:400` (`on_after_backward`); default en `RWKV-PEFT/train.py:123` |
| **`theirs`** | $L = \text{CE}_w \cdot (1 - \alpha) - \alpha \cdot \text{logsigmoid}(\text{ratio})$ | $-\alpha \cdot \text{logsigmoid}(\text{ratio})$, con $\text{ratio} = s_w - s_l$ sobre suma sin normalizar | $\alpha = 0.01$ (convexo) | `main/src/trainers/orpo.py:14-33` (`compute_logps_simple_mask`, div por longitud comentada en `:31`), `:130-138` (mode 0); default en `main/train.py:145` |

### Especificaciones de Implementación y Control
- **Cross-Entropy común**: calculada estrictamente sobre las posiciones de respuesta del `chosen` (`reduction='mean'`), idéntica en los tres brazos.
- **Eficiencia**: cálculo del proyector final de vocabulario (65536) acotado a los tokens de respuesta (`h[mask] @ head.weight.T`), evitando proyectar toda la secuencia del prompt.
- **Inicialización idéntica**: controlada por semilla (`torch.manual_seed(seed)` previo a instanciar `MiniRWKV`), verificada mediante hash SHA256 de `emb.weight`.
- **Hilos de CPU**: fijados en 2 (`torch.set_num_threads(2)`) para evitar contención de recursos en el host.

---

## Smoke (números reales, 40 pasos, seed 0)

Resultados medidos en CPU sobre `kateto_orpo_v8.jsonl` con vocabulario oficial RWKV (65536):

| Brazo | `ce_at` 40 | `grad_pref_norm_median` | `pref_gap_median` | `pref_acc_ema` | Ritmo (`s/paso`) |
|---|---|---|---|---|---|
| **`sft`** | 10.0704 | 0.0000 | -0.028 | 0.4917 | 0.0989 |
| **`ours`** | 10.0913 | 0.1178 | -0.003 | 0.5305 | 0.1464 |
| **`theirs`** | 10.2133 | 0.9613 | -103.700 | 0.1556 | 0.2600 |

### Observaciones de los Datos
1. **Gradiente de preferencia**: en `sft` es exactamente `0.0000`. En `ours` inyecta un gradiente moderado y balanceado de mediana `0.1178`. En `theirs` inyecta un gradiente elevado (`0.9613`) debido a la saturación del término de preferencia acumulado sin normalizar por longitud.
2. **Gap de Log-Odds**: el gap de `theirs` ($\text{mediana} = -103.700$) es órdenes de magnitud mayor que el de `ours` ($\text{mediana} = -0.003$), confirmando que la omisión de la división por longitud distorsiona severamente la escala de los log-odds.
3. **Costo de CE y Precisión**: `ours` mantiene una pérdida CE muy cercana al baseline SFT (10.0913 vs 10.0704) mientras mejora el `pref_acc_ema` (0.5305 vs 0.4917). Por el contrario, `theirs` degrada la CE a 10.2133 y obtiene un `pref_acc_ema` bajo (0.1556).

---

## Ritmo (900 pasos × 3 seeds)

**lo corre el orquestador**
