# Style tune (K3) — adaptar el tensor de estilo sin tocar los pesos

## Qué es

El style tune mueve **un tensor**: el estado inicial de la recurrencia,
`blocks.{i}.att.time_state`, un `(n_head, head_size, head_size)` float32 por capa.
Se crea en `RWKV-PEFT/rwkvt/rwkv7/att.py:191` y con `--peft state` el 100% de los
pesos queda congelado (`rwkvt/peft_loading.py:56-64`). Lo que se entrena es el
arranque del modelo — de ahí salen ritmo, tics y voseo; el conocimiento no se toca.

Que se llame "un tensor" y sean 24 no es contradicción: es una familia de tensores
por capa, y `S_0` es el nombre que ya usa el repo para el conjunto
(`HANDOFF.md:56`, `docs/decision-gguf-vs-nativo.md`).

## Por qué NO se mergea

No existe merge. `RWKV-PEFT/merge/merge_state.py` esta verificado y documentado:
`w[k] = w_state[k]` escribe claves que el modelo base no tiene (base: 1062 claves,
ninguna con `time_state`; state: 24 claves, todas `time_state`; intersección 0), o
sea un checkpoint con 24 tensores muertos que `strict=False` ignora en silencio.
`time_state` es estado de RUNTIME, no una matriz de pesos.

Lo que si funciona — y lo que hace `scripts/style_tune_kateto.py` — es **agregar**
el tensor de estilo a un checkpoint. Para eso el camino nativo ya esta listo:
`rwkv_pipeline/infer_kateto.py:408` busca exactamente `blocks.{i}.att.time_state`.

## Los tres comandos

```bash
# que tensor de estilo tiene (o que no tiene)
python3 scripts/style_tune_kateto.py info  <ckpt.pth>

# inicializarlo en ceros sobre un ckpt de solo pesos
python3 scripts/style_tune_kateto.py init  <ckpt.pth> --out <salida.pth>

# aplicar un head de estilo (el .pth que deja --peft state) y guardar el .pth nuevo
python3 scripts/style_tune_kateto.py apply <ckpt.pth> --head <head.pth> --out <salida.pth>
```

`info` sobre pesos solos sale con **rc 3** y un mensaje que dice la clave que
espera y como se arregla — nunca un `KeyError`:

```
error: este checkpoint no tiene tensor de estilo (ninguna clave '.att.time_state');
son solo pesos. Para agregarlo: 'init <ckpt> --out <salida.pth>'
```

`init` sobre un RWKV-7 1.5B (24 capas, n_embd 2048):

```
tensores: 24
forma: (32, 64, 64)
params: 3145728
dtype: torch.float32
bytes: 12582912
```

Es decir **3.1M parametros, 12.6 MB**. Para el 0.4B (n_embd 1024) da 1.57M, el
"1.6M" de `HANDOFF.md:56`. El costo de guardar el estilo entra en el presupuesto de
disco sin discusión.

`n_layer` y `n_embd` se deducen del ckpt (claves `blocks.{i}.*` y `emb.weight`);
`head_size` es hyperparametro del modelo y va por `--head-size` (default 64, el
`head_size_a` de `rwkvt/args_type.py:48`). `apply` valida el head contra esa
geometria y se niega entero —sin dejar un `.pth` a medias— si una capa no calza.

## Cómo se engancha a RWKV-PEFT

La cadena ya existe, no hubo que construirla:

1. **Entrena** — `RWKV-PEFT/train.py --peft state` (o `--train_type state`;
   `train.py:193-196` pone `RWKV_TRAIN_TYPE=state`, `att.py:22-27` devuelve
   `RWKV_Tmix_x070_State`). Con ese flag `train_callback` guarda **solo** las claves
   con `state` en el nombre (`rwkvt/lightning_train/trainer.py:170-180`): el `.pth`
   que sale es un head slim de N tensores, sin pesos.
2. **Aplica** — `apply <base.pth> --head <ese head slim>` cose el estilo sobre el
   checkpoint de Capa 1 y escribe un unico `.pth` que el camino nativo carga.
   Los pesos base salen identicos bit a bit; eso es lo que verifica el test.
3. **Sirve** — `rwkv_pipeline/infer_kateto.py` (backend `rwkv`, `enable_rosa = true`).

`init` existe para el caso de querer el tensor antes de entrenar (probar que el
runtime lo carga). `init` sobre un ckpt que ya tiene estilo se niega salvo
`--force`: sobreescribirlo tira el estilo entrenado.

## El comando exacto de Kaggle (2xT4)

Es el mismo que ya corre en `kaggle_kernel_v3/kateto_train_v3_2xt4.ipynb` (CELL 7),
que es a donde va este doc. Estado de voz sobre el 1.5B ya SFTeado:

```bash
# en Kaggle, /kaggle/working/kateto-train
# (env identico al CELL 7 del notebook: el state tuning va en 1 GPU)
N_LAYER=24 N_EMBD=2048 MICRO_BSZ=4 GRAD_CP=0 DEVICES=1 STRATEGY=auto \
PROJECT=/kaggle/working/kateto-train \
  bash rwkv_pipeline/train_state_voice.sh seco \
    /kaggle/working/out/rwkv_kateto_base/rwkv-final.pth
# -> /kaggle/working/kateto-train/out/rwkv_states/seco/rwkv-{0..4}.pth
```

Ojo con dos cosas que no se deducen leyendo `train.py`:

- `--op fla` es obligatorio. `RUN_RWKV7_STATE` solo existe para `WKV='fla'` o
  `'triton'` (`rwkvt/operator/rwkvop.py:28,90,289`); con `cuda` el state tuning
  no tiene kernel.
- el LR es **1e-2 → 1e-4**, no el `2e-4 → 2e-5` del SFT: son 3M parametros, no 1.5B.

`DEVICES=1` a proposito: son 3.1M parametros y el DDP no compra nada — es lo que
deja puesto el CELL 7 del notebook. El SFT de Capa 1 si va con las 2 T4.

Luego, el paso que agrega este K3:

```bash
# head slim -> checkpoint unico con el estilo adentro
python3 scripts/style_tune_kateto.py info  /kaggle/working/kateto-train/out/rwkv_states/seco/rwkv-4.pth
python3 scripts/style_tune_kateto.py apply \
  /kaggle/working/out/rwkv_kateto_base/rwkv-final.pth \
  --head  /kaggle/working/kateto-train/out/rwkv_states/seco/rwkv-4.pth \
  --out   /kaggle/working/kateto-ckpt/kateto-1.5b-seco.pth
```

Esto **no** corre todavia. El `.pth` final se exporta en el CELL 9.

## Tests

```bash
python3 -m pytest tests/test_style_tune.py -q     # 4 passed
```

Sin GPU, sin red, sin modelo real: los fixtures son tensores chicos (32 de
embedding, 2 capas) generados en el test. El test no importa torch (mismo truco que
`scripts/test_fix126.py`): lo que toca tensores corre en un subproceso con
`venv/bin/python`, asi el comando del gate verifica de verdad con cualquiera de los
dos interpreters en vez de dar un verde de skips.

Lo que cubren: `info` sin estilo falla legible (rc 3, sin traceback, sin `KeyError`);
`init` e `info` reportan la misma forma y los mismos params; **`apply` deja los
pesos base identicos bit a bit** y solo cambia el tensor de estilo; un head que no
calza se rechaza sin escribir nada.

Dos bugs reales los cazaron los tests mientras se escribian, por si aparecen en
otro lado: `n_embd` es `emb.weight.shape[1]` (no `shape[0]`: `emb` es
`nn.Embedding(vocab_size, n_embd)`, `rwkv7/model.py:22`), y validar `apply` solo
contra las claves que ya estan dejaba pasar un head de otra forma.

## Lo que este doc NO toca

`RWKV-PEFT/` quedo **sin modificar**. Ni `rwkvt/peft_loading.py`, ni `train.py`, ni
`merge/`: el PiSSA y el merge ya estaban y el state tuning ya estaba soportado. Lo
unico que se agrego es `scripts/style_tune_kateto.py` al lado, que no importa
`rwkvt` (lee y escribe `.pth` con torch pelado) y por eso no necesita ni
`lightning` ni `peft` para correr.