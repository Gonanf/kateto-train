# ATAQUE PROMETHEUS — Kateto Train

Fecha: 2026-10-02. Repo: kateto-train @ working tree. Solo lectura; nada se entreno ni se modifico fuera de este archivo.

Metodologia: cada numero lleva el comando y su salida cruda. Cuando no pude correrlo, dice "no verificado".

---

## A. DATOS

**A1. El 10,4% de centinela `<|no_response|>` contamina la voz "seco".**
- Evidencia (medido):
```
$ wc -l data/rwkv_kateto_base_train_plus.jsonl
44351 data/rwkv_kateto_base_train_plus.jsonl
$ python3 (conteo) -> total 44351 no_response 4632 10.4%
lens: p50 688 chars, p90 2133, max 6394
turns per sample: 1 turno: 22295, 2: 2463, 3: 19593
speaker seco: 43711 muestras, 4632 con no_response (10,6% de sus muestras)
speaker doktor/streamer/jane/whisperer: 160 cada uno, 0 con no_response
```
- El no_response vive 100% en "seco" (4632/4632). Es la voz principal y la que evalua el multiturno.
- Experimento barato (<30 min): fine-tune 0.4B con el mismo pipeline quitando las 4632 filas con `<|no_response|>` y re-correr `eval_multiturn.py` carry (mismo seed 1337). Si la tasa de `stop_reason=id_end` desde turno 1 baja, el centinela ensueza el turno temprano; si no, la explicacion del silencio esta en otro lado.
- Impacto: alto si explica el id_end temprano / esfuerzo: medio (una corrida de fine-tune corta + eval de 64 filas).

**A2. Desbalance de speakers brutal: 98,6% "seco".**
- Evidencia: contadores anteriores -> 43711 seco vs 160 x 4 voces. Eso es 273:1.
- El "state-tuning por voz" (config/state_tuning.yaml: `epochs_per_voice doktor 15, jane 12, whisperer 10, streamer 15, seco 5`) asume 5 voces entrenables, pero el dato base es casi todo de una. Un state-tune con 160 muestras por voz sobre 10-15 epochs es memorizable, no una voz.
- Experimento: contar n-gramas unicos por voz y correr perplejidad del 0.4B sobre las 160 de doktor vs 160 random de seco (script de 20 lineas sobre el ckpt). Si la PPL de doktor es ~1.0, el modelo ya las memorizo -> las voces "perdido" no son voces generalizables.
- Impacto: alto sobre la narrativa de "5 voces" / esfuerzo: bajo.

**A3. Largos: la mayoria es 1 turno, pero el 44% es 3 turnos sin control de largo.**
- Evidencia: 22295 de 1 turno, 19593 de 3, 2463 de 2. Mediana 688 chars (~170 tokens aprox), p90 2133 chars (~500 tokens). Con ctx 512 de entrenamiento se trunca la cola del p90.
- Experimento: histograma de tokens por muestra con el tokenizer real (no chars) y % que excede 512. Si >20% trunca, cada sample 3-turnos perdio su ultimo intercambio y el modelo nunca vio cierres.
- Impacto: medio / esfuerzo: bajo.

## B. ENTRENAMIENTO

**B1. Hiperparametros copiados, no medidos.**
- Evidencia: `config/state_tuning.yaml`: lr_init 1e-2 "agresivo para desplazar el atractor dinamico" (comentario, no medicion), warmup 50, s0_l2_gamma 5e-4, epochs por voz 5-15. `config/pissa_2.3b.yaml`: lr 2e-4, LoRA r16 alpha32. `config/orpo.yaml`: lr 5e-5, batch 1. Ningun archivo del repo guardo una tabla de lr sweep ni curva de validacion para justificar esos valores. grep de `sweep|lr_sweep|curva` en docs/ y out/*.log no aparecio nada que justifique 1e-2 vs 1e-3.
- Experimento: 3 state-tunes de seco a lr 1e-3/1e-2/1e-1 con el mismo seed y 500 pasos, comparar loss final y PPL en `data/kateto_v2_combined_eval.jsonl` (9044... no: ese es train; usar eval jsonl del split). 30 min si corren en la misma GPU con ctx chico.
- Impacto: alto (el state-tune con lr 1e-2 puede estar destruyendo S0 en vez de desplazarlo) / esfuerzo: medio.

**B2. LoRA vs state-tuning: sin A/B en el mismo checkpoint.**
- Evidencia: LoRA esta en pissa_2.3b.yaml (fallback r16) pero el state-tuning es la via de voz; no hay documento que compare downstream usables% de LoRA-merge vs state-tune sobre el MISMO base.
- Experimento: tomar el 0.4B, correr un LoRA r16 corto y el state-tune equivalente, mismo seed/datos, medir los dos en `eval_multiturn.py --state-policy carry`.
- Impacto: medio / esfuerzo: medio.

**B3. ctx 512 de train vs ctx 8192 del checkpoint.**
- Evidencia: SPEC.md:42 declara base `ctx8192`; `config/pissa_2.3b.yaml:15` entrena con ctx_len 512; eval multiturno usa `--ctx-truncate 512`. Entrenar en 512 y evaluar degradacion multiturno a 512 puede estar midiendo el truncado, no el modelo (ver D).
- Experimento: re-correr eval con `--ctx-truncate 2048` sobre el mismo checkpoint con historial truncado a 2048. Si `utilizables` sube mucho, el cuello era ctx, no el estado.
- Impacto: alto / esfuerzo: bajo.

## C. COHERENCIA MULTITURNO

**C1. El modo de falla dominante es silencio por emision del centinela, no repeticion.**
- Evidencia medida sobre los jsonl del harness (`/home/chaos/harness-run/kateto-multiturno/out/`):
```
04b-carry: 64 filas, stop_reason: id_end 26, marker 26, max_tokens 12; vacia 35, eco 14, rep8_max 0
04b-reset: id_end 33, marker 12, max_tokens 19; vacia 20, eco 15, rep8_max 0
15b-carry: id_end 57, max_tokens 4, marker 3; vacia 14, eco 21, rep8_max 0.4706
15b-reset: id_end 46, max_tokens 13, marker 5; vacia 13, eco 26, rep8_max 0.1081
base-carry: marker 13, max_tokens 32, id_end 19; vacia 1, eco 57, rep8_max 0
rosa-on-full: max_tokens 63, repetition 1; vacia 0, eco 54, rep8_max 0.6667
```
- Los numeros del usuario (15/64, 29/64, 25/64, 6/64, 9/64) matchean: 04b-carry 15, 04b-reset 29, 15b-carry 29, 15b-reset 25, base-carry 6, 04b-carry-nopen 9. rep8=0 en todo salvo un outlier en 15b-carry.
- Conclusion alternativa mejor: el modelo base (RWKV7-G1j) habla ChatML con `<|im_end|>` (IM_END) como terminador; si el fine-tune lo entreno para cerrar turnos con `<|im_end|>` INMEDIATAMENTE cuando no hay nada que decir (por el 10,4% de no_response de A1), entonces el "silencio desde turno 1" es el centinela de no-respuesta aprendido en entrenamiento. El eco creciente con el historial (eco 14->26 de carry a reset, base-carry eco 57/64) es la segunda via: el modelo copia el prompt porque nunca aprendio a responder sobre historial largo.
- Que falta medir: (1) tasa de `vacia`/`stop_reason=id_end` en el PRImER turno vs el 8, aislado por conversacion; (2) si las filas `vacia` contienen literal `<|no_response|>` o cadena vacia; (3) eco vs longitud de historial (scatter); (4) same eval sobre el base SIN fine-tune pero con el mismo prompt template: si el base tambien cierra con id_end, la culpa es del template/ctx, no del fine-tune; (5) logprob del token `<|no_response|>` en el primer token del turno 1 — si domina, es decision aprendida.
- Experimento barato: correr el eval 8x8 sobre el base con temperatura 0.8 seed 1337 y contar stop_reasons (ya existe `base-carry`: id_end 19, max_tokens 32 -> el base NO cierra con id_end tanto como el 1.5B; eso apunta al fine-tune). El experimento que falta: ablar los 64 textos de 04b-carry y contar cuantos empiezan con `<|no_response|>` literal.
- Impacto: alto / esfuerzo: bajo (contar strings en los jsonl ya escritos).

**C2. `vacia` incluye centinela Y cadena vacia: la metrica mezcla dos fallas distintas.**
- Evidencia: `scripts/eval_multiturn.py:132` `es_vacia` devuelve True para "", whitespace y `<|wait|>`/centinela. Un modelo que emite `<|no_response|>` (decision de no-hablar, correcta en voz) puntua igual que uno que devuelve `""`.
- Experimento: separar `vacia_centinela` de `vacia_estricta` en el jsonl y recomputar utilizables.
- Impacto: medio / esfuerzo: bajo.

## D. EVALUACION

**D1. El set 8x8 es sintetico del harness, no conversaciones reales.**
- Evidencia: `data/multiturno_set.jsonl` tiene 8 lineas; el doc `docs/eval-multiturno.md` dice que el problema que mide es "el 1.5B responde bien con 1 mensaje pero degrada rapido con mas", y el instrumento recorre ese set. No hay ningun set anotado de conversaciones reales del usuario (grep de `conversacion` en data/ no muestra otro multiturno real).
- Consecuencia: las tablas 15/64 vs 29/64 miden desempeno sobre 8 conversaciones inventadas; la varianza entre conversaciones puede ser enorme (turno_max_sano=1 en todas las corridas -> la cola es perfecta pero la media es 0).
- Experimento: muestrear 20 conversaciones reales de uso (logs de chat_loop si existen) y correr la misma metrica. Comparar distribucion de `utilizables` por turno.
- Impacto: alto (las tablas pueden no generalizar) / esfuerzo: medio.

**D2. n=64 de un solo seed: las diferencias de ~10 puntos no se separan del ruido.**
- Evidencia: `docs/rosa-medicion.md:99` admite "21/64 vs 15/64: diferencia +0.094, error estandar 0.080, z=1.18". Con un seed, lo unico que se puede afirmar es "con este seed y este harness".
- Experimento: correr 3 seeds (1337, 1338, 1339) x los 4 brazos y reportar media + IQR.
- Impacto: alto sobre toda conclusion A/B / esfuerzo: medio (4 corridas x ~15 min).

## E. ROSA

**E1. El head entrenado 20k pasos SI bajo loss (2,67 -> 0,64 medido) pero el A/B es de una sola corrida.**
- Evidencia medida:
```
$ head/tail out/rosa_head_lr3e4/metrics.jsonl
{"step": 1, "loss": 3.8125, ...}
{"step": 2, "loss": 4.375, ...}
...
{"step": 20000, "loss": 0.63672, "rosa_grad_norm": 3.898912}
$ wc -l out/rosa_head_lr3e4/metrics.jsonl
20000
A/B nativo del harness: rosa-off-full 6/64 vs rosa-on-full 10/64 utilizables (turno_max_sano 1->2)
r9-on-entrenado 15/64 = r9-off 15/64 (exacto empate, del brief de R9)
```
- El claim "loss 2,67 -> 1,93" del usuario no se verifica contra el metrics.jsonl crudo: el loss final medido es 0,637. El primer step es 3,81. El 2,67->1,93 quiza sea una media movil suavizada; no esta persistido como tal. "No verificado" para esa forma exacta.
- Experimento: 3 seeds del A/B nativo ROSA on/off con el head lr3e4; si on>off se sostiene en los 3, hay algo; si no, era seed.
- Impacto: alto / esfuerzo: medio.

**E2. step_infer O(T) y buffer que crece: defecto conocido, no arreglado.**
- Evidencia: `docs/rosa-diagnostico.md` tabla de rosa_module.py:54-78: `step_infer` concatena memory_k/memory_v en cada paso (lineas 63-68) y proyecta sobre todo el buffer -> O(T) por token, O(T^2) por generacion. `docs/ggml-rwkv7-rosa-gap.md:186` lo llama "defecto del modulo".
- Experimento: medir wall-time por token con --rosa en ctx 512 vs 4096 sobre el 0.4B; si el costo/token crece lineal, confirmado en el nativo tambien.
- Impacto: alto a ctx largo / esfuerzo: bajo (medir).

**E3. En GGML, matematica igual a PyTorch pero el -10% tokens/s y la memoria +2MiB/2048 no estan verificados en el repo.**
- Evidencia: `docs/rosa-ggml-viabilidad.md` reporta `max_abs_diff: 0` para el round-trip de ESTADO (no para el port de ROSA); "No midio velocidad de swap con ROSA". El claim de max_abs_diff 6e-05 para ROSA en GGML: grep en docs/ y scripts/ no lo encuentra -> "no verificado". El -10% tokens/s: no aparece en ningun doc -> "no verificado".
- Experimento: correr el binario GGML con el port ROSA y reportar tok/s vs el mismo binario sin ROSA, y RSS del proceso a ctx 512 vs 8192.
- Impacto: alto / esfuerzo: bajo-medio.

## F. CAMINO GGUF/ESTADOS

**F1. Inyeccion de estado nativa funciona y el round-trip es PASS.**
- Evidencia: `docs/rosa-ggml-viabilidad.md` (re-corrida del gate):
```
test1_identidad: OK (21627668 payload bytes identical after 8 header bytes)
max_abs_diff: 0
test2_logits: OK
test3_continuacion: OK
RESULTADO: PASS
script_exit: 0
```
y los scripts `script/probar-gguf-estado.sh` / `script/probar-estado-pth.sh` existen. Estado nativo 21.627.676 bytes (~20,6 MB) entra en el GGUF del mismo modelo.
- Experimento: re-correr `bash script/probar-gguf-estado.sh` hoy y guardar salida; es el experimento de confirmacion de la afirmacion.

**F2. Los estados de voz son del 0.4B; el camino quedo sin cerrar para 2.9B.**
- Evidencia: `docs/rosa-ggml-viabilidad.md: (a)`: "Es lo unico que bloquea de verdad... Los unicos checkpoints ROSA del arbol son RWKV-8... este repo es RWKV-7". SPEC.md:42 declara base ctx8192; los states de `config/state_tuning.yaml` se generaron para el 0.4B (0.4B en los logs de out/entrenar-local-0.4b*.log). No hay `.pth` de estado para el 2.9B.
- Experimento: listar `find . -name "*.pth" | grep -i state` y verificar que todos cuelgan del 0.4B; intentar cargar uno en el 2.9B y ver que rompe (dimension mismatch) — confirma que "los estados de voz son del 0.4B".
- Impacto: alto (todo el camino GGML de estados queda atado al 0.4B) / esfuerzo: bajo.

## G. EFICIENCIA

**G1. El cuello real no es el swap de estados (48-135 ms); es ROSA O(T) y el ctx 512 de train.**
- Evidencia: `docs/decision-gguf-vs-nativo.md:44`: swap de voz 0.4B 48-135 ms, promedio 47.6 ms sobre 15 swaps. Estado 21.627.676 bytes. Con workers de voz que rotan cada varios segundos, 48 ms por swap es despreciable. ROSA `step_infer` cat del buffer es O(T) por token (E2): a ctx 4096 cada token paga proyeccion sobre 4096 pares K/V. El buffer de 20,6 MB es el estado nativo, entra en RAM facil.
- Experimento: perfilado de 1 turno de 64 tokens con y sin --rosa a ctx 512/2048/8192 (time-per-token); el que crece es el cuello.
- Impacto: alto / esfuerzo: bajo.

**G2. "ROSA suma 2 MiB de estado por cada 2048 de contexto": calculo plausible pero no persistido.**
- Calculo: retrieval_dim 128 x 2 (K,V) x 2 bytes (bf16) x 24 capas x 2048 tokens = 128*2*2*24*2048 = 25,165,824 B ~= 24 MB por 2048, no 2 MiB. Mi cuenta da ~12x mas. Marcar "no verificado, y el numero del enunciado no cierra contra la aritmetica del modulo".
- Experimento: medir RSS del proceso nativo con --rosa a ctx 1024/2048/4096 y ajustar la pendiente.
- Impacto: medio / esfuerzo: bajo.

## H. LO QUE NO SE SOSTIENE

- "Loss 2,67 -> 1,93" para el head de 20k: el metrics.jsonl crudo muestra 3,81 -> 0,637 (E1). No se sostiene como esta escrito.
- "ROSA matematica iguala a PyTorch con max_abs_diff 6e-05": en el repo solo `max_abs_diff: 0` del round-trip de ESTADO (sin ROSA) (E3). No se sostiene.
- "ROSA cuesta -10% tokens/s": no aparece en ningun doc medido. No se sostiene.
- "ROSA suma 2 MiB por 2048 ctx": la aritmetica del modulo (E3/G2) da ~24 MB por 2048 con retrieval_dim 128, bf16, 24 capas. No se sostiene.
- "swap de voz 48-135 ms" es promedio de 15 swaps sobre CPU/GPU no especificada en el doc; el doc mismo admite que no se midio velocidad en el camino nativo con ROSA ni en GGML. Se sostiene solo como orden de magnitud.
- "epoch_steps 750, epoch_count 2" en pissa_2.3b.yaml con warmup 50 y lr 2e-4: sin sweep persistido.
- "el modelo base 6/64" y "sin penalizaciones 9/64": existen como base-carry.json (6/64) y 04b-carry-nopen.json (9/64); pero el nombre "04b-carry-nopen" no documenta que penalizaciones se sacaron. No se sostiene la interpretacion sin leer el script.
- HANDOFF.md:274 dice que el repo "Inicializa RosaAssociativeMemory": falso segun grep (E/diagnostico): no hay ningun import ni instanciacion fuera de la definicion. No se sostiene.
- "RWKV-8 checkpoints de ROSA portables": falso para este repo (RWKV-7), ya cubierto en F2.
- "los estados de voz sirven para el 2.9B": no hay evidencia; todo lo corrido es 0.4B.

## Los 5 arreglos por impacto/esfuerzo

1. **Limpiar `<|no_response|>` del SFT o separar `vacia_centinela` de `vacia_estricta` en el eval** (A1/C2). Experimento: re-correr eval con la metrica separada; si el 35/64 de vacia de 04b-carry son todos centinela, el diagnostico cambia de "modelo roto" a "modelo que aprendio a callarse", y el arreglo es limpiar el dato. Impacto alto, esfuerzo bajo.
2. **Re-correr eval con `--ctx-truncate 2048` y con 3 seeds** (B3/D2). Experimento: mismo checkpoint, ctx 512 vs 2048, 3 seeds; si utilizables sube y la varianza baja, las tablas actuales median ctx+seed, no el modelo. Impacto alto, esfuerzo bajo.
3. **A/B real de hiperparametros del state-tune** (B1). Experimento: lr sweep de 3 valores x 500 pasos, PPL en eval split; reemplaza el 1e-2 "agresivo" por un numero medido. Impacto alto, esfuerzo medio.
4. **Verificar si `<|no_response|>` domina el primer token del turno 1** (C1). Experimento: script de 30 lineas que cargue el 0.4B y loguee logprob del centinela en el primer step del turno 1 sobre las 8 conversaciones; si ranking top-1, es decision aprendida del dato, no del decoder. Impacto alto, esfuerzo bajo.
5. **Portar el camino de estados al 2.9B o bajar el claim** (F2). Experimento: intentar cargar el state .pth del 0.4B en el 2.9B y documentar el fallo; si se quiere el claim, correr state-tuning en el 2.9B y regenerar estados. Impacto alto (cierra o corrige la narrativa), esfuerzo medio-alto.

## Lo que NO sabemos

- Si ROSA con el head de 20k pasos mejora de verdad la utilizabilidad multiturno: el A/B es de un seed y 64 filas; no se separa del ruido.
- Si el silencio desde turno 1 es por el centinela aprendido (A1) o por ctx 512 (B3) o por el template: no hay medicion aislada.
- Que pasa con el modelo a ctx >512 en train: todo lo corrido esta en 512.
- Si los 160 ejemplos por voz generalizan o estan memorizados: no hay PPL por voz.
- El costo de ROSA en GGML (tokens/s, memoria): port existe, numeros no.
- Si el estado nativo sirve en el 2.9B: no hay estado del 2.9B.
- Si el set 8x8 representa conversaciones reales: no hay conversaciones reales en el set.
- Que penalizaciones se sacaron en "nopen": el nombre no lo dice.
