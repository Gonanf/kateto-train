# k2 — Clasificador: elegir modelo y partirlo en dos binarias (investigación + diseño + medición)

**Repo de trabajo:** `/run/media/chaos/terciario/proyectos/kateto-train` (ahí viven el venv, los modelos
y los datasets de Kateto). **ESTE BRIEF NO TOCA CÓDIGO DEL RUNTIME**: es investigación, medición y
diseño. El runtime de Kateto se toca después, en otro brief y otro repo.

**Ojo:** hay 12 workers de generación corriendo contra el gateway local. No los toques, no lances olas,
no toques `out/`.

## Qué hay que decidir (lo pidió el usuario, textual)
1. **Qué modelo conviene** para el clasificador de turno: `convaiinnovations/laya-multilingual` vs
   `microsoft/Multilingual-MiniLM-L12-H384` vs "OpenJev o similar". Se aceptan otros candidatos si
   hay evidencia.
2. El clasificador hoy usa **ejemplos** (few-shot). Hay que dejarlo **one shot** (un solo ejemplo, o
   zero-shot si el modelo lo justifica).
3. **Partirlo en dos clasificaciones binarias**, en serie:
   - **Etapa 1 — ¿terminó la idea?** Mientras no terminó, el texto se **acumula en un buffer**.
   - **Etapa 2 — (sólo si terminó) ¿esto requiere respuesta?** Se clasifica el **buffer completo**.
   Las dos salidas son binarias. Nada de clasificación de 3 clases acá.

## Lo que ya está medido (no lo re-diagnostiques)
Del dataset y del runtime de Kateto, en esta sesión:
- El corte de turno hoy es un **timer fijo**: `silence_timeout=1.5s`, `turn_silence_timeout=2.0s`, VAD
  0.2 en `kateto/plugins/audio_input/base.py` del repo `~/proyectos/OpenaiBuildWeek/Kateto`.
- El clasificador de intención del runtime usa ONNX con `Qdrant/all-MiniLM-L6-v2-onnx`
  (`kateto/providers/classifier.py`, `kateto/plugins/executor/classifier.py`).
- Frases ambiguas reales a resolver: "Sí, claro" (backchannel vs respuesta) y "Y después me dijo
  que..." (idea incompleta). El techo del texto solo, sin señal acústica, está estimado en 75-80 %
  por la literatura que trajo el usuario.

## Tareas
### 1. Ficha de cada modelo candidato (con fuente)
Tamaño de parámetros, idiomas, licencia, si corre en ONNX/CPU, latencia estimada para secuencias de
3-50 tokens, y si hay evidencia de que sirve para **español rioplatense conversacional**. Citá dónde lo
viste. `laya-multilingual` y `Multilingual-MiniLM-L12-H384`: bajá la model card de HF y leela.

### 2. Prueba real, chica y honesta
Si el disco lo permite (ver restricciones), bajá **2 de los modelos** (el que parezca mejor y un
baseline) y corré una prueba de sanidad **con datos que ya existen**: usá el dataset de Kateto que está
en `/run/media/chaos/terciario/proyectos/kateto-train/out/dataset/shards/` (JSONL, campo `text`) y los
escenarios de `scripts/gen_toolcalling_dataset.py` como fuente de etiquetas débiles.
- Lo mínimo: que el modelo cargue, que clasifique, y cuanto tarda por llamada en CPU.
- **No entrenes nada** y no inventes números: si no llegás a medir precisión, decilo y entregá la
  latencia y la facilidad de uso, que alcanza para decidir.
- Si no hay internet o el disco está lleno, **no lo fuerces**: entregá la comparación documental y
  decilo en el reporte.

### 3. Diseño de las dos etapas (esto es lo importante)
- Cómo se serializa cada etapa: entrada (texto del buffer), salida binaria, formato exacto de la
  respuesta del modelo (un token, un logit, un JSON), y qué se hace con la confianza.
- **Dónde vive el buffer**: qué lo llena (parciales del ASR), cuándo se vacía (tras la etapa 2), qué
  pasa con el timeout del VAD y con el barge-in, y qué pasa si la Etapa 1 dice "terminó" pero la
  Etapa 2 dice "no requiere respuesta" (¿silencio? ¿`<|no_response|>`?).
- Umbrales: qué confianza mínima para decidir, y qué se hace por debajo (¿esperar más audio?).
- Latencia objetivo del pipeline completo (Etapa 1 + Etapa 2) y qué se mide para verificarlo.

### 4. Plan de evaluación con criterio de aceptación
Cómo se mide que las dos etapas andan mejor que el timer fijo: qué set de prueba, cuántas muestras,
qué métrica (precisión/recall de "terminó" y de "requiere respuesta") y qué número lo da por bueno.

## Restricciones de entorno (van en el brief porque rompen corridas)
- **El disco raíz está al 95 %.** Cualquier descarga de modelos va a `HF_HOME=/run/media/chaos/terciario/hf`
  (72 GB libres ahí). Verificá con `df -h /` y `df -h /run/media/chaos/terciario` antes y después.
- No uses `/tmp` para nada grande (es tmpfs chico).
- Si un modelo pesa más de ~1 GB, no lo bajes: medilo sólo documentalmente.

## Reglas
- **No escribas ni modifiques nada en el repo** (ni código, ni datasets, ni `out/`). Sólo lectura.
- **No commitees, no pushees.**
- Todo número que reportes tiene que venir de un comando que corriste, pegado en el reporte.
- Lo que no pudiste verificar va en una sección **No verificado**.

## Entregable
`/home/chaos/harness-run/k2/DESIGN.md` (ruta ABSOLUTA):
1. `## TL;DR` (15 líneas máximo) con la recomendación.
2. Fichas de los modelos, con fuentes.
3. Resultados de la prueba local (comandos + salidas crudas) o el motivo por el que no se hizo.
4. Diseño de las dos etapas binarias con el buffer y los umbrales.
5. Plan de evaluación y criterio de aceptación.
6. No verificado / supuestos.
