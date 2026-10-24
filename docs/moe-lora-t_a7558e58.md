# Evidencia MoE-LoRA en RWKV-7 (OpenMOSE RWKV-LM-RLHF) — Brief t_a7558e58

## 1. Qué hay en código
- `main/src/models/rwkv7.py:967`: clase `Router` (`nn.Linear(input_dim, num_experts, bias=False)`).
- `main/src/models/rwkv7.py:980`: clase `RWKV_CMix_x070_LoRAExperts` (key/value con `make_linear_ffn_experts`).
- `main/src/models/rwkv7.py:1083` (y 992): clase `RWKV_CMix_x070_MoLE` (expert_0 compartido + top-k gated, `moe_info`, load_balance_loss).
- `main/src/linears.py:467`: clase `Shared_LoraLinear` (adaptador LoRA por experto sobre peso base congelado).
- `main/src/linears.py:313`: función `LinearForward_Experts` (despacho de adaptadores por experto).
- `main/src/model.py:90-91`: instanciación `if os.environ["CustomModel"] == 'MoE': self.ffn = RWKV_CMix_x070_MoLE(...)`.
- `main/src/model.py:108-119`: forward del bloque con retorno de `moe_router_loss`.
- `main/src/model.py:979-991`: acumulación de `moe_total_loss += (moe_router_loss + 0.001) / n_layer`.

## 2. Cómo se activa
- `main/train.py:56-60`: flags CLI `--moe 1`, `--moe_experts 4`, `--moe_active 2`, `--moe_shared 1`, `--moe_balance_alpha 0.01`.
- `main/train.py:268`: `--moe 1` define variable de entorno `CustomModel="MoE"`.
- `main/layerprofile/24_TEST_peft_moe.csv`: capas 0-15 en LoRA rank 32 (alpha 64), 16-23 en bone rank 32.

## 3. Qué NO hay
- Merge de MoE: `main/merge.py:1-134` no contiene referencias a MoE ni router; no puede fusionar múltiples expertos a peso denso.
- Inferencia MoE en engine: `main/rwkvengine/` carece de soporte MoE/MoLE (solo coincidencias léxicas en el vocabulario).
- Inferencia head2 rota: `main/src/model.py:1031` `forward_head2` llama `block(x, v_first, passthrough)` sin `input_ids`, fallando con MoE.

## 4. Presupuesto de parámetros (L=24, D=2048, FFN=8192)
| Configuración | LoRA Params | Router | Total Params |
| :--- | :--- | :--- | :--- |
| LoRA dense r=16 (baseline v7) | 7,864,320 | 0 | 7,864,320 |
| LoRA dense r=32 | 15,728,640 | 0 | 15,728,640 |
| LoRA dense r=64 | 31,457,280 | 0 | 31,457,280 |
| LoRA dense r=128 (control honesto) | 62,914,560 | 0 | 62,914,560 |
| MoE E=4 active=2 r=16 | 31,457,280 | 147,456 | 31,604,736 |
| MoE E=4 active=2 r=32 | 62,914,560 | 147,456 | 63,062,016 |
Control honesto de MoE E=4 r=32 es LoRA dense r=128 (router agrega 147,456 params, +0.23% sobrecosto).

## 5. Qué queda por medir
- Coherencia larga y cierre de turno (`<|im_end|>`): requieren generación con el propio `RWKV` de `main/src/model.py`.
- No es viable evaluar vía llama.cpp/GGUF ni RWKV-PEFT estándar sin exportador MoE dedicado.
