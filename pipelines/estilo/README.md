# R7 — Style tune de un solo tensor

La tecnica es de [Gryphe](https://huggingface.co/Gryphe/Gemma-4-26B-A4B-StyleTune-V2):
congelar TODO el modelo y entrenar **un solo tensor**, la proyeccion de salida
(`head.weight`), 1 epoch. El resultado publicado: 52% menos cliches y 19,9% de
trigramas compartidos con la base. Su V1 (mas de un epoch) le arruino la
estabilidad en MoE; la V2 es 1 epoch.

Se adapta a Kateto porque el problema medido es exactamente ese: el modelo sabe
hablar y no sabe sonar como Kateto.

Acá hay dos cosas, y conviene no confundirlas:

| | que es | estado |
|---|---|---|
| `metricas_estilo.py` | el **instrumento**: mide la brecha de voz | anda, verificado |
| `style_tune_head.py` | la **prueba** de que el congelado deja 1 tensor | dry-run pasa, exit 0 |

Lo que NO hay acá es el style tune entrenado: es un instrumento y una prueba, no
el run final.

---

## 1. El instrumento

Mide sobre dos corpus (acepta archivos o globs):

1. **densidad de cliches por 100 palabras** — 31 tells: 10 marcadores de
   asistente, 13 genericos, y 8 tomados de `prompts/comedia.md` (M6 linea 56,
   prosodia escrita; M7 linea 64, explicar el chiste).
2. **solapamiento de trigramas** entre los dos corpus (Jaccard, mas el
   porcentaje del corpus B que aparece en A) y **top-N de cliches con su
   conteo**, mas el diff de los cliches de B que no aparecen en A.

Sin dependencias (solo stdlib). Matching accent- y case-insensitive.
`--solo-voz` saca el turno del usuario y el nombre de la voz, y mide solo las
respuestas: los cliches viven del lado de la voz.

```bash
env -u PYTHONPATH -u VIRTUAL_ENV \
  /run/media/chaos/terciario/proyectos/kateto-train/venv/bin/python \
  pipelines/estilo/metricas_estilo.py --help
```

## 2. Una correccion al brief: cual es el corpus generado

El brief pide comparar contra `out/dataset/shards/parcial-w*.jsonl` como si fuera
el corpus generado. **No lo es.** Es la *entrada* de la purga, no su salida:

- `pipelines/kateto_purge.py:35-36` — `SHARD_GLOB = "parcial-w*.jsonl"`,
  `SHARD_DIR = REPO/"out"/"dataset"/"shards"`. El pipeline los **lee**.
- `_stats-shards.json` — `entrada: 39996`, `mantenidas: 21617`. Los 39996 docs de
  `parcial-w*` son exactamente la entrada, y las 21617 que sobreviven son
  exactamente `shards.purged.jsonl`.
- Verificado por hash md5 doc a doc: los **21617 de 21617 (100,00 %)** docs
  purgados aparecen byte-identicos dentro de `parcial-w*`.

O sea: `parcial-w*` es la **voz humana** (streams/videos), no generations del
modelo. Comparar humano-purgado contra `parcial-w*` mide el efecto de la purga,
no la voz del modelo.

Las generaciones reales que hay en la maquina son muchas y chiquitas. Se
poolearon las que tienen la firma `{text,scenario,cat,voice}` que escribe
`kateto_gen.py`:

```bash
# arma /tmp/generado.jsonl: 111 docs de generations reales
env -u PYTHONPATH -u VIRTUAL_ENV \
  /run/media/chaos/terciario/proyectos/kateto-train/venv/bin/python -c "
import json, glob
frm = ['pipelines/smoke/*/muestras.jsonl','pipelines/smoke/tc_r2.jsonl',
       '/run/media/chaos/terciario/proyectos/kateto-train/out/dataset/*.jsonl']
seen, out = set(), []
for pat in frm:
    for f in sorted(glob.glob(pat)):
        try: lineas = open(f, encoding='utf-8')
        except OSError: continue
        for l in lineas:
            l = l.strip()
            if not l: continue
            try: o = json.loads(l)
            except json.JSONDecodeError: continue
            if isinstance(o, dict) and {'text','scenario','cat','voice'} <= set(o) and o['text'].strip():
                if o['text'] in seen: continue
                seen.add(o['text']); out.append(o)
with open('/tmp/generado.jsonl','w',encoding='utf-8') as fh:
    for o in out: fh.write(json.dumps({'text':o['text']},ensure_ascii=False)+'\n')
print('docs en el pool:', len(out))
"
```

| docs | archivo |
|---|---|
| 5 | `pipelines/smoke/registro-off/muestras.jsonl` (worktree) |
| 3 | `pipelines/smoke/registro-on/muestras.jsonl` (worktree) |
| 3 | `pipelines/smoke/tc_r2.jsonl` (worktree) |
| 16 | `out/dataset/medicion2.jsonl` |
| 20 | `out/dataset/medicion3.jsonl` |
| 20 | `out/dataset/medicion4.jsonl` |
| 9 | `out/dataset/medicion_cmd.pre_measurement.jsonl` |
| 15 | `out/dataset/mimo_v1.jsonl` |
| 6 | `out/dataset/ola2_smoke.jsonl` |
| 4 | `out/dataset/probe-fix118.jsonl` |
| 4 | `out/dataset/smoke_opencode.jsonl` |
| 3 | `out/dataset/smoke_ola3.jsonl` |
| 2 | `out/dataset/smoke_http_gptoss.jsonl` |
| 2 | `out/dataset/tc_test.jsonl` |
| 1 | `out/dataset/smoke_cmd.jsonl` |
| **111** | **despues de deduplicar por `text` (113 archivo-doc, 2 repetidos)** |

**111 es poco.** Los numeros de abajo son de esa muestra y hay que leerlos como
tal.

## 3. La brecha medida

```bash
env -u PYTHONPATH -u VIRTUAL_ENV \
  /run/media/chaos/terciario/proyectos/kateto-train/venv/bin/python \
  pipelines/estilo/metricas_estilo.py \
    --a /home/chaos/harness-run/kateto-purge/shards.purged.jsonl \
    --b /tmp/generado.jsonl --solo-voz --top 30
```

| | humano (purgado) | generado (pool) |
|---|---|---|
| documentos | 21617 | 111 |
| palabras (solo voz) | 2 144 452 | 10 572 |
| cliches (31 tells) | 227 (11 distintos) | **1** (1 distinto) |
| **cliches por 100 palabras** | **0,0106** | **0,0095** |
| trigramas de B que estan en A | — | 3474 (**40,41 %**) |
| Jaccard de trigramas | — | 0,003886 |

### Top-10 de cliches del generado que NO aparecen en el humano

**Ninguno.** El unico cliche del lado generado es `por otro lado` (1), y tambien
aparece en el humano (20 veces). La lista de la brecha sale vacia.

### Contra que se lee el 40,41 %

Sin referencia, 40,41 % no dice nada. Estas son las comparaciones que dan escala
(todas con el mismo pool de 111 docs, mismo criterio de solo-voz):

| comparacion | trigramas de B en A | que es |
|---|---|---|
| humano vs humano (shards vs train_set) | **63,60 %** | dos voces humanas del mismo origen |
| generado vs humano descartado | 45,74 % | contra la voz que la purga rechazo |
| **generado vs humano purgado** | **40,41 %** | lo que nos importa |
| humano descartado vs humano purgado | 17,35 % | otra voz humana, otra fuente |

El generado cae **entre** los dos extremos humanos: por debajo del piso de "dos
corpus de la misma voz" (63,60 %) y muy por arriba del piso de "otra voz"
(17,35 %). O sea: elige algo de vocabulario propio, pero no tanto como para
parecer otra persona.

## 4. Que dice y que no dice estos numeros

- **La densidad de cliches NO es una brecha medible aca, y sale al reves.** El
  generado tiene **1** ocurrencia en 10 572 palabras: 0,0095 contra 0,0106 del
  humano. El lado generado esta *debajo* del humano, y sobre 1 hit eso no es una
  diferencia: es ruido. Haria falta un orden de magnitud mas de generations para
  que ese numero signifique algo. Si estabas buscando "el modelo clichea mas que
  la voz real", esta medicion **no lo confirma**.
- **El instrumento detecta la brecha cuando existe.** Con los 18379 docs humanos
  que la purga descarto encuentra tells que no estan en el corpus purgado:
  `por un momento` (5) y `(risa)` (3). La lista de la brecha se llena sola. Las
  generaciones reales no muestran eso, no porque el instrumento no lo vea, sino
  porque en 111 docs no los hay.
- **El 40,41 % no es comparable al 19,9 % de Gryphe.** El de el es el head de
  Gemma (vocab ~262k) contra el corpus base de Gemma; el de aca es el head de
  RWKV (vocab 65536) contra un corpus de voz. Que los dos sean "pocos trigramas
  compartidos" no los pone en la misma escala.

## 5. El style tune (la parte probada)

`style_tune_head.py` congela todo y entrena **solo `head.weight`**.

El atajo matematico: con el tronco congelado sus hidden states son constantes, asi
que no hace falta backpropagar a traves de el — se calculan una vez por tanda
bajo `no_grad` y se entrena el head sobre esas features cacheadas. El gradiente
que llega a `head.weight` es **exactamente** el mismo que en un fine-tune con todo
el modelo delante, y la senal de gradiente baja de **450 834 432** parametros a
**67 108 864**.

Los operadores RWKV-7 **no estan reimplementados**: se importan de
`rwkv_pipeline/infer_kateto.py`, que es la unica fuente de verdad de la
matematica del modelo. Reimplementarlos seria duplicar el forward y arriesgarse
a que se desincronice.

### Cuanto del modelo real es el head

Medido sobre `out/rwkv_kateto_0.4b_ctx256k/rwkv-1.pth` (24 capas, vocab 65536,
n_embd 1024), leyendo solo las formas con `mmap`:

| | params | del total |
|---|---|---|
| head (`65536 x 1024`) | 67 108 864 | **14,89 %** |
| total | 450 834 432 | 100 % |

O sea: en el modelo real el head es **1 tensor de 798** (el mismo conteo que
reporta Gryphe para Gemma) pero el **14,89 %** de los parametros. Gryphe reporta
"1 de 659 tensores"; en RWKV el numero equivalente es 1 de 798, y el que importa
para decidir si el freeze es barato es el 14,89 %.

### Dry-run (lo que se puede correr aca)

```bash
env -u PYTHONPATH -u VIRTUAL_ENV \
  /run/media/chaos/terciario/proyectos/kateto-train/venv/bin/python \
  pipelines/estilo/style_tune_head.py --dry-run
```

Sale 0. RSS pico medido con `/usr/bin/time -v`: **738 MB** (756056 KB;
presupuesto 1,5 GB).

```
=== PARAMETROS ===
  tensores entrenables : 1 de 73
  params entrenables   : 8,388,608
  params congelados     : 8,824,320
  params totales        : 17,212,928
  fraccion entrenable   : 48.73%
  tensor entrenable     : head.weight  (proyeccion de salida)

=== INVARIANTES ===
  ok  el tronco congelado dio hidden states IDENTICOS antes y despues
  ok  el head SI cambio (el unico tensor que se entrena)
  ok  el unico .grad es head.weight (norma 1.270e+00)

[params entrenables] 8,388,608 (1 tensor, 48.73% del modelo)
```

Eso no es decorativo, son asserts: si aparece un segundo tensor con
`requires_grad`, o si el tronco cambia un solo bit, el script aborta. El ultimo
assert es la frase de Gryphe ("The model hasn't changed. Only the voice has.")
convertida en comprobacion.

**Ojo con el 48,73 %**: es del RWKV en miniatura (vocab 65536 x n_embd 128), no
del modelo real, donde el head es 14,89 % (arriba). La miniatura prueba el
mecanismo, no cuantifica la fraccion.

### Sobre el checkpoint

El brief pedia reportar si faltaba. **No falta.** Los cinco que busca el script
estan todos:

```
/run/media/chaos/terciario/proyectos/kateto-train/out/rwkv_kateto_0.4b_ctx256k/rwkv-1.pth  0.90 GB
/run/media/chaos/terciario/proyectos/kateto-train/out/rwkv_kateto_base/rwkv-1.pth
/run/media/chaos/terciario/proyectos/kateto-train/out/rwkv_kateto_base/rwkv-0.pth
/run/media/chaos/terciario/proyectos/kateto-train/base/rwkv7-g1j-2.9b-20260831-ctx16384.pth
/run/media/chaos/terciario/proyectos/kateto-train/out/kateto-rwkv29-merged.pth
```

No son los unicos `.pth` de RWKV en la maquina: tambien estan
`models/rwkv7-0.4b/rwkv7-g1d-0.4b-20260210-ctx8192.pth` (0.90 GB) y
`models/rwkv7-1.5b/rwkv7-g1j-1.5b-20260831-ctx16384.pth` (3.0 GB), que no estan
en la lista de candidatos. La lista prioriza el 0.4B, que es el que se puede
probar en CPU.

El dry-run **no los carga**. El mas chico de los cinco son 0,90 GB en disco, y
pasarlo a fp32 lo deja en ~1,8 GB: solo, ya se pasa del tope de ~3 GB del cgroup
que mato un run con exit -15. Busca las rutas, las reporta, y sigue con la
miniatura.

### El run real

```bash
... pipelines/estilo/style_tune_head.py --entrenar --ckpt <ruta> --lote 8 --ctx 512
```

1 epoch, como la V2 de Gryphe. No se corrio aca: es GPU. Ojo que `--docs` (2000
por default) acota el corpus, asi que un `--entrenar` sin `--docs` grande no es el
epoch completo.

Guarda el head en `/home/chaos/harness-run/kateto-estilo/`: `head-estilo.pth` +
`head-estilo.json` (config, informe, losses).

> **El `head-estilo.pth` que hay ahora NO es un head usable.** Es el artefacto del
> dry-run: `__meta__.base = "mini(rand)"`, `n_embd 128` contra los 1024 del modelo
> real, 2 pasos sobre pesos al azar. Sirve como prueba de que el congelado anda;
> no se puede cargar en ningun checkpoint. El head de verdad sale del `--entrenar`.

## 6. Tests

`pipelines/tests/test_estilo.py` (12 tests) corre en el gate y protege lo que
hace que los numeros de este README sean ciertos: que `fold` saque tildes, que
el conteo y los trigramas den bien, y que `--solo-voz` saque el turno del usuario
y el nombre de la voz. Los 2 tests del congelado (que importan `torch`) se
**saltan solos** si el interprete no lo tiene, para que el gate falle por el
codigo y no por el interprete.

```bash
env -u PYTHONPATH -u VIRTUAL_ENV \
  /run/media/chaos/terciario/proyectos/kateto-train/venv/bin/python -m pytest pipelines/tests -q
```

Sin torch (por ejemplo `/usr/bin/python3`), los 10 del instrumento corren y los
2 del congelado saltan con motivo:

```
10 passed, 2 skipped
```

La prueba real del congelado es el `--dry-run`, no el test.

Nota de setup: el venv no traia `distilabel`, que `kateto_purge.py` importa a
nivel de modulo; sin el, 2 tests preexistentes no colectaban. Se instalo
(1.5.3). Es dependencia real del repo, no un parche para los tests.
