# Diagnostico de ROSA y el hook exacto

**Fecha:** 2026-10-01 · **Tipo:** diagnostico, no implementa. `rosa_module.py` intacto.
**Arboles:** repo `kateto-train`, mas `/run/media/chaos/terciario/proyectos/llama.cpp-master` @ `ce8caa6` (solo lectura).

---

## 1. Que hace `rwkv_pipeline/rosa_module.py`, linea por linea

| Linea | Que hace | Nota |
|---|---|---|
| `rosa_module.py:12` | `class RosaAssociativeMemory(nn.Module)` | sin `forward` |
| `rosa_module.py:16` | `__init__(vocab_size=65536, hidden_dim=1024, retrieval_dim=128)` | defaults coinciden exacto con el ckpt 0.4B |
| `rosa_module.py:22-24` | `proj_q/proj_k/proj_v`: `Linear(1024->128, bias=False)` | 3 x 131k = 0.393 M params |
| `rosa_module.py:25` | `out_head`: `Linear(128->65536, bias=False)` | **8.389 M params, el 95% del modulo** |
| `rosa_module.py:26` | `gate`: `Linear(1024->1)` con bias | un escalar por token |
| `rosa_module.py:33-36` | proyecta `hidden_states` a q/k/v | |
| `rosa_module.py:39-40` | `scores = einsum("bid,bjd->bij", q, k) * scale` | |
| `rosa_module.py:43-44` | `mask = torch.tril(...)`; `masked_fill(mask==0, -1e9)` | diagonalincluded |
| `rosa_module.py:45` | `attn = F.softmax(scores, -1)` | |
| `rosa_module.py:48-49` | `retrieved = attn@v`; `rosa_logits = out_head(retrieved)` | sale a **vocab**, no a hidden |
| `rosa_module.py:50` | `gate_weight = sigmoid(gate(hidden_states))` | |
| `rosa_module.py:52` | devuelve `(rosa_logits, gate_weight)` | el llamador debe mezclar |
| `rosa_module.py:54` | `step_infer(current_hidden, memory_k, memory_v)` | |
| `rosa_module.py:63-68` | `if memory_k is None: ... else: torch.cat(...)` | **la causa del O(T^2)** |
| `rosa_module.py:71-73` | `einsum` sobre `memory_k` completo, `softmax`, `attn@v` | |
| `rosa_module.py:75-76` | mismos dos heads | |
| `rosa_module.py:78` | devuelve `(rosa_logits, gate_weight, memory_k, memory_v)` | el buffer se re pasa |

**Total: 8.783 M params** (medido). Los 8.389 M son `out_head`; el gate es 1025.

Medido con un smoke CPU (`/tmp/opencode/rosa_probe*.py`, dummy 256/64/16 y dims reales):

- `m(x)` **levanta** `NotImplementedError: Module [RosaAssociativeMemory] is missing the required "forward" function`. No se puede usar como `nn.Module`; hay que llamar `forward_train` / `step_infer` a mano.
- `forward_train` con hidden `bfloat16` (el dtype real del ckpt) **levanta** `RuntimeError: expected m1 and m2 to have the same dtype, but got: c10::BFloat16 != float`. Los params nacen `float32` (`rosa_module.py:22-26`, sin dtype) y no hay cast. Hay que `.to(bfloat16)` explicito.
- El `masked_fill(-1e9)` es seguro: la diagonal queda valida, ninguna fila queda toda `-1e9` (medido: `softmax` finito).

## 2. Por que esta huerfano

`grep -rn "RosaAssociativeMemory" --include="*.py" .` (excluyendo `venv/`) no da **ningun** import ni
instanciacion: solo la definicion en `rosa_module.py:12`. La unica otra mencion del repo es
documentacion: `HANDOFF.md:274` ("Inicializa `RosaAssociativeMemory`"), que **no se sostiene contra el
codigo**. La inferencia real es `rwkv_pipeline/infer_kateto.py` y su sampling
(`infer_kateto.py:248`, `infer_kateto.py:399`), que nunca lo tocan.

## 3. Donde se engancharia en el forward de RWKV-7

Hay **dos** forwards de RWKV-7 en el repo, y no son el mismo. Importa cual:

**A. Inference (la que corre hoy, y la que usa el eval multiturno).**
`rwkv_pipeline/infer_kateto.py`, RNN por token, `RWKV_RNN.forward` en `infer_kateto.py:217`.
Loop de capas en `infer_kateto.py:223`, `time_mixing` en `infer_kateto.py:229`, `channel_mixing` en
`infer_kateto.py:241`, y el head: **`infer_kateto.py:245` `logits = z['head.weight'] @ x`**.
Ese es el hook: ROSA es una rama paralela al `head`, `(rosa_logits, gate)`, y la mezcla ocurre en la
misma linea. El buffer `memory_k/memory_v` tiene que vivir en el `state` que ya se arrastra entre
turnos (`infer_kateto.py:381`), que es justo lo que `eval_multiturn.py --state-policy carry` mide.
Es el unico punto donde un cambio se ve en un numero existente.

**B. Entrenamiento (RWKV-PEFT, vectorizado, T completa).**
`RWKV-PEFT/rwkvt/rwkv7/model.py:61` `forward_normal`: `ln_out` en `model.py:81`, `head` en
`model.py:82`. El equivalente para infctx es `model.py:111-112`. El time_mix por capa esta en
`RWKV-PEFT/rwkvt/rwkv7/att.py:147` (`forward`) y `:292` (`FullState`). Ahi `forward_train` encaja
natural: es paralelo sobre T, y la mascara causal ya la hace (`:43`).

**C. GGML (solo si se.porta; hoy no corre).**
`src/models/rwkv7.cpp`: `att_norm` en `rwkv7.cpp:154`, `time_mix` en `rwkv7.cpp:161`, `channel_mix`
en `rwkv7.cpp:190`, head en `rwkv7.cpp:205`. **Hook del residuo: `rwkv7.cpp:190`** (el `cur` que entra
a `build_rwkv7_channel_mix`, `rwkv7-base.cpp:9`) — ahi `ffn_norm` ya esta normalizado y con reshape
2d (`rwkv7.cpp:182`). El `gate` se mezcla en el head, `rwkv7.cpp:205`. La op no es nueva: es
`ggml_soft_max_ext` (`ggml/include/ggml.h:1813`) con `kq_mask` + `kq_scale`, el mismo patron que ya
usa attention en `src/llama-graph.cpp:2694`. La variante attention de RWKV-7 esta en
`src/models/arwkv7.cpp:153` / `arwkv7.cpp:169-179`.

**Nota de criterio:** el hook de la auditoria previa (`docs/ggml-rwkv7-rosa-gap.md:181`) nombra
`rwkv7.cpp:190`, y coincide. Lo que **no** dice la auditoria es que el camino que de verdad se puede
medir hoy es A, no C.

## 4. El costo O(T^2) de `step_infer`, con las lineas que lo causan

El docstring de `rosa_module.py:56` dice "recurrente O(1)". **Es falso.** Las causas, textas:

| Linea | Codigo | Costo |
|---|---|---|
| `rosa_module.py:67` | `memory_k = torch.cat([memory_k, k], dim=1)` | re-copia el buffer **entero** cada token |
| `rosa_module.py:68` | `memory_v = torch.cat([memory_v, v], dim=1)` | idem para V |
| `rosa_module.py:71` | `einsum("bid,bjd->bij", q, memory_k)` | O(T) por token, inevitable en atencion |
| `rosa_module.py:73` | `einsum("bij,bjd->bid", attn, memory_v)` | idem |

`torch.cat` es la que rompi el O(1): no es append in-place, es una reasignacion de un tensor nuevo de
tamano T. Medido (dims reales 1024/128, CPU): el costo por token **se duplica** al duplicar T —
`cumulative_s` 0.021 / 0.043 / 0.086 s en T=128 / 256 / 512, y el ratio de las ventanas
128->256 contra 256->512 da **2.02**. O sea O(T) por token, O(T^2) por contexto. La atencion de las
lineas 71/73 es O(T) por token pero con factor chico y sin re-copia; el `cat` es el que multiplica.

A T=512 el buffer KV son 512 KiB en fp32 / 256 KiB en bf16 (medido, `retrieval_dim=128`). Chico en
memoria, caro en CPU por el `cat`. En GGML el equivalente natural es el KV-cache paginado que ya
existe, no un buffer que crece: por eso hay que arreglarlo **antes** de portar, no despues.

Bug ademas sin verificar en el alcance de este item: `step_infer` **no** aplica mascara causal ni
excluye el token actual de la memoria, mientras que `forward_train` si la aplica (`:43-44`). Con el
`cat` de `:67` el token `t` entra en su propia softmax. No lo verifique mas alla de la lectura del
codigo.

## 5. Plan de cableado y criterio de aceptacion

Elegido: **A (PyTorch, `infer_kateto.py`), no C (GGML)**. Razon: C necesita ~3 semanas de estado
inyectable antes de poder medir nada, y el gap que ROSA ataca (`HANDOFF.md:274`: nombres de
funciones, tokens exactos) se ve en el eval multiturno que ya existe.

Orden (un item por vez, cada uno medible):
1. **Arreglar `step_infer`**: preallocar el buffer con indice de escritura en vez de `cat`, y aplicar
   la mascara que falta. Sin esto, el resto no se puede medir. Gate: el ratio de costo por token entre
   T y 2T baja de 2.02 a ~1.0.
2. **Una sola instancia de ROSA, en el head**: instanciar con `vocab_size=65536, hidden_dim=1024`
   (los defaults ya son los correctos para el 0.4B: `n_embd`=1024 y vocab 65536, leidos del ckpt), y
   mezclarla en `infer_kateto.py:245`. **No una por capa**: 8.783 M x 24 = **210.8 M** params, medido,
   contra 400 M del modelo base. El gate lo hace `sigmoid(gate(h))` por token, asi que la mezcla es
   `logits + gate * rosa_logits`.
3. **Cargar pesos entrenados**: `out_head` son 8.389 M — es un `head` paralelo, o se entrena con el
   resto o no sirve de nada.
4. **Arrastrar el buffer entre turnos**: vive en el `state` (`infer_kateto.py:381`), que es lo que
   `carry` vs `reset` mide.

**Criterio de aceptacion (el unico numero que importa):** `tasa_utilizable` de
`scripts/eval_multiturn.py` con `--state-policy carry` **supera el 23.44%** de la linea base
`/home/chaos/harness-run/kateto-multiturno/out/04b-carry.json` (15/64), con el resto de los flags
identicos (`--turns 8`, temp 0.8, top_p 0.7, seed 1337). Si `carry` no mejora sobre su propio `reset`
de ese mismo run, ROSA no esta aportando memoria: es decoracion.

Instrumento ya listo: `scripts/eval_multiturn.py` con `--turns`, `--state-policy carry|reset`, `--tag`,
`--model`, `--out-dir`. Los smokes en CPU; en background necesitan `systemd-run --user --scope
-p MemoryMax=infinity` (el cgroup de 4 GiB mata el ckpt cargado con exit 137).

## Registro de lo no verificado

- Que ROSA **mejore** el `tasa_utilizable`: **no verificado**. Este item no entrena ni corre nada de
  ROSA contra el modelo; el analisis es de lectura + un smoke dedims de `rosa_module.py` en CPU.
- La mezcla exacta `logits + gate * rosa_logits`: **no verificado**. El modulo devuelve el gate pero
  no lo aplica (`:52`, `:78`); el criterio de mezcla es una decision mia, no del codigo.
- La mascara faltante en `step_infer`: **no verificado** en comportamiento, solo leida.
- Si `out_head` debe ser un `head` compartido con `z['head.weight']` o uno nuevo: **no verificado**,
  no hay diseno previo en el repo.
- Compilacion de kernels, VRAM, y el GGUF de llama.cpp: **fuera de alcance**, ya marcadas como no
  verificadas en `docs/ggml-rwkv7-rosa-gap.md:293-300`.
- `rwkv_pipeline/kaggle_rwkv7_g1j.py` (48 lineas) y las copias del modulo en `kaggle_v2/` /
  `kaggle_upload/` que menciona la auditoria: **no verificado** en esta corrida.
