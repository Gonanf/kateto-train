# ROSA medida con el head entrenado (R9)

Pregunta: con pesos que pasaron por entrenamiento, ROSA ayuda en el dialogo
multiturno de 8x8? Los numeros, no la intuicion.

## Veredicto

**No se puede decidir si ROSA ayuda: con el head de 300 steps el delta es
exactamente 0 (utilizables ON 15/64 vs OFF 15/64, textos identicos 64/64) y el
propio entrenamiento no muestra aprendizaje (loss plano oscilando 2-4.6 con
spikes de 87.5/55.0, gradiente a 0 desde ~step 93, norma ROSA +1.1%), asi que
el brazo ON sigue sin poner a prueba el recall de ROSA.**

## Gate (a): el head — constancia, no hubo que esperar

Al arrancar este item `out/rosa_head_corto/` YA tenia el head (entrenamiento
terminado 15:45, `VERDICT: OK` en su log). No se espero: la condicion de los
30 minutos no llego a aplicar.

```
$ stat -c '%y %s %n' out/rosa_head_corto/rosa_head.pth out/rosa_head_corto/train.log out/rosa_head_corto/metrics.jsonl
2026-10-01 15:45:39.302316864 -0300 17568813 out/rosa_head_corto/rosa_head.pth
2026-10-01 15:45:41.896288955 -0300 33676 out/rosa_head_corto/train.log
2026-10-01 15:45:39.285317046 -0300 27232 out/rosa_head_corto/metrics.jsonl
$ md5sum out/rosa_head_corto/rosa_head.pth
f5de418e078055a5d9fef8a604866aa7  out/rosa_head_corto/rosa_head.pth
```

Metadata del head (leida del `.pth` en CPU): `n_embd=1024, vocab_size=65536,
retrieval_dim=128, step=300`, 8,782,849 params. El motor aborta con `SystemExit`
si el head es de otro tamano de modelo (`infer_kateto.py:343-348`); los brazos
ON cargaron sin abortar, asi que el head es del tamano del 0.4B que se mide.
No aplica parar por mismatch.

Senal de aprendizaje del entrenamiento (de su propio `train.log`):

```
[rosa] base 1024d x24l vocab=65536 device=cuda:0 dtype=torch.bfloat16
[rosa] head=RosaHead +8,782,849 params trainables (retrieval_dim=128)
[rosa] subconjunto: 22/64 filas usables (ctx_len=512, T real=128->128 con pad)
[rosa] step 77/300 {"step": 77, "loss": 87.5, "label_tokens": 60, "rosa_grad_norm": 5542.048847, ...}
[rosa] step 165/300 {"step": 165, "loss": 55.0, "label_tokens": 60, "rosa_grad_norm": 2074.01573, ...}
[rosa] guardado /run/media/chaos/terciario/proyectos/kateto-train/out/rosa_head_corto/rosa_head.pth
[rosa] rosa_norm 148.224789 -> 149.910396 (delta 1.685607)
VERDICT: OK
```

Loss sin descenso (oscila 2-4.6 toda la corrida, con spikes), `rosa_grad_norm`
en 0.0 desde ~step 93 hasta el 300, y la norma de ROSA se mueve +1.69 sobre
148.22 (+1.1%). Entreno nominal 300 steps, aprendizaje efectivo casi nulo.

## Gate (b): los dos brazos

Procedencia honesta: los dos brazos los corrio el intento-1 del mismo loop
(archivos de 16:39 y 16:47, GPU `cuda`, secuenciales en diseno). Este item NO
los re-corrio: al arrancar, la GPU estaba ocupada por el re-run secuencial del
brazo ON (`r9-on-entrenado`, mismo seed/head, iniciado 16:57) y el brief
prohibe los dos brazos en paralelo con 4 GB de VRAM; lanzar otra corrida
encima hubiese puesto en riesgo la del loop por contienda de VRAM. Lo que si
hizo este item: verificar los dos `.jsonl` fila por fila por cuenta propia
(misma grilla 8x8 en el mismo orden, mismos flags salvo `rosa`, textos
comparados uno por uno) y contra la linea base externa.

Comandos (mismo que R5 + `--rosa-head`; `--out-dir` fuera del repo):

```
# ON (rosa-on-entrenado.jsonl, 16:39, wall 982.9 s)
scripts/eval_multiturn.py --model out/rwkv_kateto_base_plus_0.4b/hist/rwkv-1-20260928-1739.pth \
  --turns 8 --state-policy carry --device cuda \
  --rosa --rosa-head out/rosa_head_corto/rosa_head.pth \
  --tag rosa-on-entrenado --out-dir /home/chaos/harness-run/kateto-multiturno/out
# OFF (r9-off.jsonl, 16:46, wall 955.5 s): el mismo sin --rosa ni --rosa-head
```

Prueba de que el brazo ON cargo pesos ENTRENADOS y no random (salida cruda del
re-run en vuelo, mismo head md5 `f5de418e...`, misma linea que emite el motor
en ambas corridas):

```
[Kateto] ROSA encendida con PESOS ENTRENADOS de '.../out/rosa_head_corto/rosa_head.pth'
(step 300, +8,782,849 params): la rama suma gate*rosa_logits sobre los logits del head.
[mt] device efectivo: cuda  (pedido: cuda)
```

Controles de identidad verificados por este item:

- ON vs OFF: 64 filas, misma grilla `(conversacion, turno)` en el mismo orden,
  `policy=carry` y `ctx_truncate=512` en las 128, `rosa=true+head` en ON,
  `rosa=false` sin head en OFF. **Textos identicos 64/64.**
- OFF vs linea base externa `/home/chaos/harness-run/kateto-multiturno/out/04b-carry.jsonl`:
  **textos identicos 64/64.** El brazo OFF es la linea base, igual que en R5.
- Caveat: el ON original (16:39, wall 982.9 s) y el inicio del OFF (16:46
  menos wall 955.5 s = 16:30) solapan ~9 min, o sea que el intento-1 no fue
  estrictamente secuencial. La contienda enlentece pero no cambia la
  matematica del muestreo determinista (seed 1337); el re-run secuencial
  `r9-on-entrenado` (iniciado 16:57, despues del fin del OFF) debe
  reproducirlo bit a bit. Al cierre de este item seguia corriendo.

## Tabla comparada (rosa ON entrenado vs rosa OFF)

`util` = filas no degeneradas de 8. `rep8` = promedio sobre utilizables.
`eco`/`vacia` sobre las 8 del turno.

| turno | util ON | util OFF | rep8 ON | rep8 OFF | eco ON | eco OFF | vacia ON | vacia OFF |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | 0.000 | 0.000 | 0/8 | 0/8 | 5/8 | 5/8 |
| 2 | 1 | 1 | 0.000 | 0.000 | 1/8 | 1/8 | 6/8 | 6/8 |
| 3 | 2 | 2 | 0.000 | 0.000 | 2/8 | 2/8 | 4/8 | 4/8 |
| 4 | 2 | 2 | 0.000 | 0.000 | 2/8 | 2/8 | 4/8 | 4/8 |
| 5 | 2 | 2 | 0.000 | 0.000 | 2/8 | 2/8 | 4/8 | 4/8 |
| 6 | 3 | 3 | 0.000 | 0.000 | 1/8 | 1/8 | 4/8 | 4/8 |
| 7 | 2 | 2 | 0.000 | 0.000 | 2/8 | 2/8 | 4/8 | 4/8 |
| 8 | 0 | 0 | n/a | n/a | 4/8 | 4/8 | 4/8 | 4/8 |
| **total** | **15/64** | **15/64** | 0.000 | 0.000 | 14/64 | 14/64 | 35/64 | 35/64 |

Modo de falla (igual en los dos brazos, fila por fila): texto real 29,
`<|no_response|>` 14, `<|wait|>` 4, vacia 17, `rep8 > 0` en 0/64.
Tasa de utilizables **0.2344** vs **0.2344** (delta 0). Longitud mediana 63
chars en los dos. La perturbacion del head entrenado no voltea ni un solo
token muestreado en 64 turnos con seed 1337.

## Gate (c): por que "no se puede decidir" y no "no ayuda"

El numero medido (delta 0 con textos identicos) dice que ESTE head no cambia
nada, pero el `train.log` dice que este head casi no aprendio (gradiente
muerto media corrida, norma +1.1%). Igual que en R5 con pesos random: un A/B
contra una rama que no aporta nada mide el ruido de la rama, no el recall de
ROSA. Contestar "ROSA no ayuda" generalizando desde un head que no entreno
seria estirarse mas alla de los numeros. Lo que falta sigue siendo lo mismo
que pedia R5: un head con curva de loss que baje de verdad, y recien ahi el
mismo A/B contesta la pregunta.
