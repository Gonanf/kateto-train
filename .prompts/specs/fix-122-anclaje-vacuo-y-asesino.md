# fix-122 — La regla de anclaje es vacua en un escenario y asesina en el otro

**Delta sobre fix-121.** Mismo archivo, va solo (los briefs 119/120/121 ya aterrizaron).

## Qué pasa — MEDIDO por el orquestador sobre la ola w79+ (30 muestras), no re-diagnostiques
`sin_anclaje_en_payload` corre en los dos escenarios de bit y falla en las DOS direcciones:

**1. `bit_sobre_video`: la regla es VACUA.** El `tool_result` va DENTRO del turno de la voz y trae el
ancla textual, así que el conteo matchea siempre. Medido sobre 19 muestras: con el texto completo,
**19/19 matchean 9 palabras del ancla** (todas); contando SOLO la prosa de la voz (sin las líneas
`tool_call`/`tool_result`), pasan **15/19 con umbral 2**. O sea: hoy la regla no puede fallar, y cuando
se la hace honesta, el 79 % de las muestras sigue pasando igual. No hay daño colateral en arreglarla.

**2. `bit_sobre_evento`: la regla es ASESINA.** 0 aceptadas y **11 rechazos, todos por esto**. El ancla
del evento es corta (`bloque`, `juego`, `corte`, `cenar`) y el teacher PARAFRASEA en vez de repetir:
medido sobre las 11, contando solo la prosa, **6 matchean 0 palabras y 5 matchean exactamente 1**.
Con umbral 2 no pasa ninguna; con umbral 1 pasan 5/11. Ejemplos reales rechazados, que son buenos bits:
- *"El horario es el de un oficinista que no quiere morir en el escritorio. Como el descanso de la
  maquina de café que taquea media hora antes de que el jefe te eche el café frio encima. Fin."*
  (bit anclado al evento semánticamente, 0 palabras literales)
- *"El horario está clavado como un clavo en un tablón: dos horas de juego, ni un minuto más…"*
  (1 palabra literal: `juego`)

**3. Bug de diagnóstico:** el `for/else` de la regla (líneas 595-601) descarta el máximo encontrado y
reporta `sin_anclaje_en_payload(0/2)` aunque haya matcheado 1. El mensaje miente.

## REGLA 0 — prohibido explorar
Abrí SOLO `scripts/gen_toolcalling_dataset.py`, `scripts/test_fix121.py` y `scripts/test_fix122.py`.
No leas `data/`, `docs/`, ni otros scripts. **No corras el generador**: hay 12 workers VIVOS contra el
gateway (unidad `kateto-ola79`). Lo único ejecutable son los gates locales.

## Cambio 1 — contar SOLO la prosa de la voz
En la regla de anclaje (líneas 575-603): antes de contar, del turno de la voz **borrá las líneas que
empiezan con `tool_call` o `tool_result`** (`re.sub(r"(?m)^\s*tool_(call|result)\b.*$", "", t_v)`).
El `tool_result` es material que llega, no bit de la voz. Sin esto la regla no mide nada.

## Cambio 2 — umbral por escenario
- Parámetro nuevo en `validate()`: `ancla_min: int = 2`.
- La comparación pasa de `if n_ancla >= 2` (y del `n_ancla < 2` final) a usar `ancla_min`.
- En `main()`: `ancla_min=sc.get("ancla_min", 2)`.
- En el escenario `bit_sobre_evento` agregá `"ancla_min": 1` (su ancla es corta y el teacher
  parafrasea: con 2 no pasa NINGUNA, medido). En `bit_sobre_video` **no** pongas la clave: se queda
  en 2, donde pasan 15/19.

## Cambio 3 — arreglar el mensaje de diagnóstico
El `for/else` de las líneas 595-601 tiene que quedarse con el **máximo** `n_ancla` de todos los turnos
de la voz (`max(..., default=0)`), no poner 0 cuando ninguno llega al umbral. El error tiene que decir
el número real: `sin_anclaje_en_payload(1/2)`, no `(0/2)`.

## Cambio 4 — el escenario del evento pide una palabra textual
En el `txt` de `bit_sobre_evento` agregá, al final:
`Apoya el comentario en al menos UNA palabra textual del evento ('bloque', 'juego', 'corte', 'cenar'):
el bit tiene que engancharse en algo concreto de lo que llego.`
6 de las 11 rechazadas no nombraban NINGUNA palabra del evento: el prompt y la regla tienen que
apuntar al mismo lado.

## Lo que NO hay que tocar
- No toques los umbrales ni la lógica de los otros escenarios, ni los escenarios existentes.
- No toques `SCENARIOS` más allá de las dos claves nuevas (`ancla_min` en `bit_sobre_evento` y el
  texto agregado). No toques `ancla_payload` de ninguno.
- No toques `out/`, la ola `kateto-ola79`, ni los teachers.
- **No commitees ni pushees.** El orquestador revisa y commitea.
- Si algo no coincide con el código, decilo en el reporte y seguí con el ancla textual.

## Tests: `scripts/test_fix122.py` (pytest, sin red)
- Una muestra de la VOZ cuyo `tool_result` trae el ancla pero cuya prosa NO la toca → **tiene** que dar
  `sin_anclaje_en_payload` (hoy pasa: es el bug 1).
- Una muestra con 1 palabra del ancla en la prosa: con `ancla_min=2` → error `(1/2)`; con
  `ancla_min=1` → sin error.
- El mensaje reporta el número real (1, no 0).
- `bit_sobre_evento` tiene `ancla_min == 1`; `bit_sobre_video` no tiene la clave.
- Los tests de `test_fix121.py` que hoy afirman el comportamiento viejo se actualizan (la regla, no el
  script, es lo que cambió).

## Gates (correlos y pegá la salida cruda)
```bash
cd /run/media/chaos/terciario/proyectos/kateto-train
./venv/bin/python -m pytest scripts/ -q
./venv/bin/python scripts/gen_toolcalling_dataset.py --dry-run
```

## Reporte — `/home/chaos/harness-run/fix122/REPORT.md` (ruta ABSOLUTA)
1. Qué cambió, `archivo:línea`, con los fragmentos pegados.
2. Salida CRUDA de los gates.
3. Qué quedó sin hacer y qué viste y no tocaste.
