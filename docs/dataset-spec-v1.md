# SPEC de dataset — Kateto voz (v1)

**Fecha:** 2026-09-12
**Autor:** Kateto (spec) / Gabriel (objetivos)
**Estado:** propuesta para revisión. Nada de esto se ejecuta sin OK explícito.
**Repo de training:** `/run/media/chaos/terciario/proyectos/kateto-train`
**Runtime que define los contratos:** `~/proyectos/OpenaiBuildWeek/Kateto`

---

## 0. Por qué existe este spec

El dataset actual (`data/rwkv_kateto_base_train.jsonl`, 20.506 ejemplos) produce un modelo con
registro argentino pero **inconsistente e incomprensible**. Medido, la composición real de sus
19.866 objetivos `seco` es:

| categoría | líneas | % |
|---|---|---|
| `tool_call` de agente (el OBJETIVO es un tool call) | 1498 | **7.5%** |
| `[INST]` (resto de template) | 1437 | 7.2% |
| rutas/código del SO (`/home/chaos`, `*.py`, `*.jsonl`) | 344 | 1.7% |
| markdown técnico | 610 | 3.1% |
| autodescripción en 3ª persona ("El usuario quiere…") | 16 | 0.1% |
| resto | 15961 | 80.3% |
| **— de ese resto, con voseo/lunfardo** | **589** | **3.7% del total** |

De ahí salen los síntomas observados en el REPL: `**5.** \`python3 -V\` → Python 3.12.4`,
`El usuario quiere que sea seco, sin rodeos`, `El ministro de Economía, Martín Guzmán…` (texto de
noticia), `Crédito: NASA`. **No es un problema de sampling, de quant ni de template: es el corpus.**

Este spec define el dataset que falta.

---

## 1. AUDITORÍA DE PREMISA — dónde vive cada decisión

Verificado en el runtime (leyendo el código), no asumido.

### 1.1 La decisión de hablar NO es del clasificador: se MUEVE AL MODELO

**Estado actual (verificado)**: `plugins/executor/classifier.py` decide antes que la voz, con la
taxonomía `EXECUTE` / `IGNORE_SELF_TALK` / `IGNORE_THIRD_PARTY`:

```python
match classification.category:
    case Classification.EXECUTE:
        _ = await manager.emit("generate", GenerateData(prompt=classification.text), ...)
    case Classification.IGNORE_SELF_TALK | Classification.IGNORE_THIRD_PARTY:
        return                      # la voz nunca es invocada
```

**El clasificador fue una solución temprana y hay que retirarlo.** No conoce el contexto completo
(SOUL, historial, fase de workflow, estado, memorias), así que produce falsos negativos y positivos.
Los tres motivos del cambio, en orden de peso:

1. **Latencia de TTFT** — un modelo más antes de decidir, o directamente el salto.
2. **Errores por falta de contexto** — el clasificador ve sólo la transcripción; el modelo ve todo.
3. **Decisión por voz** — el mismo input puede ameritar respuesta de una voz y no de otra. Eso no se
   puede expresar en un clasificador único: **es una propiedad del modelo de cada voz.**

#### Diseño propuesto: la decisión es el primer token

El modelo emite **centinelas**, no texto libre:

| primer token | significado | qué hace el runtime |
|---|---|---|
| *(texto directo)* | **hablo** (default) | genera y manda a TTS — camino feliz sin costo extra |
| `<\|no_response\|>` | **no es para mí** | cierra el turno, no genera, no habla |
| `<\|wait\|>` | **la transcripción está incompleta** | espera 5 s, **suma la próxima transcripción** y vuelve a decidir con el texto concatenado |

**Por qué centinelas y no una clasificación previa:** los centinelas se deciden **dentro de la misma
generación** que ya ibas a hacer. No hay modelo extra, no hay forward extra, y la decisión ve
SOUL + historial + estado + memorias. Es exactamente lo que el clasificador no puede ver.

**Por qué sólo los casos no-default llevan marcador:** el camino feliz (hablar) no paga tokens.
Sólo el silencio y la espera cuestan 1 token.

**La variación por voz sale gratis:** cada voz tiene sus propios pesos/estado, así que el mismo input
da decisiones distintas sin ningún mecanismo especial. "Algunas voces responderían y otras no" **es**
la distribución de cada voz, y se mide con la probabilidad del primer token.

#### Fase 2 (optimización, no requisito): decidir sin generar

Si el TTFT importa más que nada, la decisión no necesita generar: alcanza con mirar la
**distribución del primer token** (`logprobs`) y comparar `P(<|no_response|>)` contra el resto.
Un solo forward, cero tokens generados en el caso "no hablo". Requiere que el provider exponga
`logprobs` en la primera posición (llama.cpp puede). **Arrancar con centinelas generados; migrar a
logprob después de que el comportamiento esté entrenado y medido.**

#### Alternativa si el centinela resulta impracticable

Fine-tuning de **mmBERT** con la instrucción `WAIT` agregada. Si ese es el camino:
- el gate WAIT espera **5 segundos** a que el usuario empiece a hablar;
- **suma las 2 (o más) transcripciones** y le pasa el texto unificado al modelo;
- la decisión sigue siendo sin contexto completo → mantiene el problema del falso negativo, así que
  **es un fallback, no el objetivo.**

#### Consecuencia para el dataset

El objetivo 8 **vuelve a ser una categoría del dataset de voz** (era lo contrario cuando el
clasificador decidía). Ver §5 categoría F, que pasa de 0% a un bloque real — y con una exigencia
nueva: **los ejemplos de decisión tienen que ser POR VOZ** (el mismo input con `no_response` para
unas voces y respuesta real para otras).

### 1.2 Gate aguas arriba vs centinela en el modelo — pros y contras

| | **(a) Gate aguas arriba** (mmBERT/heurística) | **(b) Centinela en el modelo** |
|---|---|---|
| **latencia** | +1 forward antes de decidir (mmBERT es chico, pero es un salto de red/proceso) | 0 extra: la decisión viaja en la generación que ya ocurre (1 token) |
| **contexto de la decisión** | **sólo la transcripción** → falsos positivos y negativos | SOUL + historial + fase + estado + memorias |
| **decisión por voz** | imposible: un clasificador, un veredicto | **nativa**: es una propiedad de cada voz |
| **TTFT óptimo** | el gate suma su latencia antes de que la voz arranque | fase 2: se decide con la distribución del primer token, sin generar |
| **costo de error** | falso IGNORE = el usuario habla y nadie contesta (falla peor) | el modelo puede abusar del `wait` y esperar de más |
| **complejidad de runtime** | plugin/gate nuevo antes del classify | puerta sobre el primer token + timer de 5 s + concatenación de transcripciones |
| **entrenamiento** | dataset aparte (mmBERT), no aprovecha el SFT de voz | ejemplos en el dataset de voz, con el par por voz |
| **riesgo de implementación** | bajo (ya existe el patrón) | medio: hay que interceptar el primer token **antes** de mandar a TTS |

**Recomendación: (b).** Es lo que pediste, resuelve el motivo principal (contexto) y es el único que
puede expresar la decisión por voz. El punto de riesgo real no es el modelo: es **la puerta de
runtime** que tiene que leer el primer token y cortar antes del TTS. Se mitiga empezando con
centinelas generados (no logprob) y un test que verifique que ningún `<|no_response|>` llega al
sintetizador.

**Híbrido razonable si querés bajar el riesgo**: (b) para el caso "no es para mí" (que necesita
contexto) y dejar el gate aguas arriba **sólo** para el corte de ruido obvio (silencio, tos, ruido
del chat). Lo obvio no necesita contexto.



### 1.3 Y una que rompe el objetivo "distinga sistema de usuario" — RESUELTO: se separa por ROL

`voices/base.py` `_remember_event` mapea **al mismo rol** cosas que son distintas:

| qué llega | cómo entra al historial (HOY) |
|---|---|
| transcripción del usuario | `user`, texto crudo |
| pedido de OTRA VOZ (`VoiceRequest`/`Generate`) | `user`, con el prompt |
| evento de workflow | `user`, con prefijo |
| chess `move_request` | `user`, con `[CHESS GameMode] …` |
| lo que dice la propia voz | `assistant` |
| tool call / tool result | `assistant` con `tool_call {n}: {args}` / `tool_result {n}: {res}` |

**El usuario y las otras voces entran los dos como `user` con texto crudo.** El modelo **no puede
distinguirlos**: la información no está en el input.

**DECISIÓN TOMADA: se separa por rol.** Verificado que `developer` ya es un rol válido
(`providers/_models.py:14` → `role: Literal["assistant","developer","system","user"]`):

| rol | qué lleva |
|---|---|
| `user` | **sólo** la transcripción del humano |
| `developer` | eventos, workflows, plugins y **pedidos de otras voces** |
| `assistant` | lo que dice la propia voz + `tool_call` / `tool_result` |
| `system` | SOUL + schemas + skills + memoria durable (estable) y contexto volátil |

**Requiere tocar `_remember_event` en `voices/base.py`.** No hace falta vocabulario nuevo en el
prompt ni prefijos frágiles: el rol ya existe, sólo hay que usarlo.

> **DEPENDENCIA BLOQUEANTE (hallazgo del 2026-09-12, medido generando).** El rol es del lado
> *runtime* (API OpenAI). Pero **el modelo no lee roles: lee texto**, y RWKV se sirve con
> marcadores serializados. Hoy el vocabulario de marcadores es `<|im_user|>`, `<|im_start|>{voz}`,
> `<|im_end|>`: **no existe ningún marcador para `developer`**.
>
> Consecuencia concreta, observada en la primera tanda generada: los eventos del sistema
> (`game_start:`, `phase_change:`, `plugin_alert:`) salieron bajo **`<|im_user|>`**, o sea
> textualmente idénticos a una persona hablando. El objetivo 3 (distinguir fuente) **no se puede
> entrenar así**: el modelo no tiene de dónde sacar la diferencia.
>
> **Antes de generar §2.3 y §2.4 hace falta, en este orden:**
> 1. Elegir el marcador textual de `developer` (candidato: `<|im_dev|>`, simétrico al resto) y
>    agregarlo como **token especial** — ver §2.1, hoy `<|im_start|>` se parte en 7 tokens.
> 2. Que el runtime serialice `developer` a ese marcador al servir RWKV (el `Literal` ya lo acepta).
> 3. Recién entonces generar la categoría C con el marcador correcto.
>
> Si esto no se hace, la categoría C produce basura que *parece* correcta: conversaciones con
> eventos que el modelo aprende a tratar como si los hubiera dicho el usuario.


---

## 2. Formato exacto del dato (verificado en código, no negociable)

### 2.1 Qué formato usar: se QUEDA el actual, y no es ChatML

**Aclaración importante, porque el nombre viene confundiendo**: lo que usa el dataset **no es ChatML**.
ChatML es `<|im_start|>{role}\n…<|im_end|>` con roles `system/user/assistant`. El proyecto usa
`<|im_user|>{texto}\n<|im_start|>{voz}\n{respuesta}<|im_end|>` — un esquema **propio**, de la familia
Qwen/RWKV-PEFT. Vale la pena llamarlo **`rwkv-kateto`** de acá en adelante para no mezclar conceptos.

**Los tres formatos en juego:**

| formato | quién lo habla |
|---|---|
| **RWKV World** (`User: {q}\n\nAssistant: {a}`) | los checkpoints **BASE** de BlinkDL. Verificado: alimentarlos con `<\|im_user\|>` produce bucle degenerado. |
| **`rwkv-kateto`** (`<\|im_user\|>` / `<\|im_start\|>{voz}` / `<\|im_end\|>`) | **el fine-tune de Kateto y los 5 voice states**, el REPL y toda la suite de eval. |
| ChatML | **nadie**. No hay ningún artefacto que lo hable. |

**Decisión: se queda `rwkv-kateto`.** Razones, en orden:

1. **Es lo que hablan todos los artefactos existentes**: el fine-tune de 2.9B, el de 0.4B, el
   `rwkv_kateto_seco.jsonl` limpio, el REPL (`kateto_chat.py`) y la suite de eval con sus umbrales.
2. **Cambiar a World invalida los 5 voice states.** Los states están entrenados sobre los checkpoints
   fine-tuneados, que hablan `rwkv-kateto`. Pasar a World obliga a reentrenar el fine-tune **y los 5
   states** — multiplica por 5 el trabajo del cambio. Con 5 voces confirmadas, ese costo es real.
3. **World no da ninguna ventaja acá**: es el formato de los base crudos, y nosotros no vamos a
   servir base crudos — servimos Kateto.

**Lo que sí hay que arreglar del formato** (y es aparte de la elección): `<|im_start|>` **no es un
token especial** en este vocab, son **7 tokens** (`[61,125,1949,96,35691,125,63]`). Verificado. Eso
funciona porque el train usó el mismo texto crudo, pero es frágil. Dos opciones, en orden de
preferencia:

- **(i) Agregar los marcadores como tokens ESPECIALES** al vocab (`<|im_user|>`, `<|im_end|>`,
  `<|im_start|>`, `<|no_response|>`, `<|wait|>`) y reservarles id. Es lo correcto y elimina la clase
  entera de problemas de tokenización.
- **(ii) Dejarlos como texto multi-token**, que es lo que pasa hoy. Funciona, pero hay que entrenar y
  servir con el mismo texto exacto, y cualquier cambio de tokenizer rompe en silencio.

Los centinelas de §1.1 (`<|no_response|>`, `<|wait|>`) **necesitan** la opción (i) o al menos ser
secuencias estables: si se parten en 7 tokens distintos, la puerta de runtime que lee el primer token
no puede decidir con uno solo.

### 2.2 Estructura y reglas del historial

Del runtime (`voices/base.py`, `voices/context.py`, `providers/_models.py`) y del
`data/toolcalling_sft.jsonl` existente:

- Estructura por turno: system estable → historial (últimos 8 eventos, texto acotado, sin duplicados
  consecutivos) → system volátil → user actual.
- **No existe el rol `tool`.** Los tool calls y sus resultados se guardan como texto `assistant`:
  - tool call → `tool_call {nombre}: {args json ordenado}`
  - resultado → `tool_result {nombre}: {resultado}` o `… Error: {mensaje}`
- Los juegos **no** entran crudos: chess llega como prompt
  `[CHESS GameMode] … Respondé SOLO con un UCI`; el resto de eventos de juego sólo va al overlay.
- El marcador de la voz es `<|im_start|>{voz}` (ej. `seco`, `doktor`, `jane`), el del humano
  `<|im_user|>`, y el cierre `<|im_end|>`.
- Los datasets de voz son **multi-turno dentro de un mismo `text`**, no una línea por respuesta.

**Regla dura:** nada de JSON de eventos crudos ni de formato OpenAI `role: tool` en el objetivo. El
modelo aprende los prefijos de texto exactos o no aprende nada útil.

---

## 3. Inventario de tools reales (para no inventar ejemplos)

**Kateto builtin — 18 verificadas** (`kateto/voices/tools.py`, extraídas de los esquemas, no del doc;
el brief previo decía 19 porque contaba `get_current_time` dos veces):

| grupo | tools |
|---|---|
| archivos | `read_file`, `write_file`, `delete_file` |
| shell | `run_command` (allowlist CLI, timeout 30 s, env mínimo) |
| eventos | `send_event`, `list_events` + **una dinámica por cada evento con receivers** (`build_event_tools`) |
| plugins | `enable_plugin`, `disable_plugin`, `list_plugins` |
| memoria de voz | `create_skill`, `update_skill`, `create_workflow`, `update_workflow`, `update_soul`, `refine_memory` (requiere `dept`; add/replace/remove por substring) |
| orquestación | `request_generation` (**TERMINAL**: corta el turno, sin follow-up) |
| scheduling | `schedule_event` (delay/interval/cron + target), `get_current_time` |

**MCP externo — video-rag, 6 tools** (repo `~/proyectos/video-rag`, servidor MCP por stdio; verificado
leyendo `src/mcp.rs`):

| tool | args | qué hace |
|---|---|---|
| `search` | `query`, `video?`, `k=5` | búsqueda semántica sobre transcripciones indexadas; devuelve chunks con timestamp y hablante |
| `list_videos` | — | lista los videos con índice semántico |
| `describe` | `video` | overview por capítulos, con citas resolubles |
| `describe_images` | `prompt`, `images` (≤8 data-URL, ≤~1.5 MB c/u) | captiona hasta 8 frames en una llamada VLM |
| `answer` | `video`, `query`, `k=5` | QA **con cita** sobre un video indexado |
| `summarize` | `video` | digest de acciones escalado por duración (decisiones, tareas, deadlines, montos) |

**Reglas duras del executor que el dataset debe enseñar:**
- preflight valida `required` y tipos **antes** de ejecutar;
- **turno truncado (`stop_reason=length`) NUNCA ejecuta tools** — descartar el tool call;
- `request_generation` es terminal: no hay texto después.

**Juegos (repo `~/proyectos/game-bridges`, NO en el repo Kateto):**
- `chess`: `move_request` con `fen` + `legal_moves` + `request_id` → responde **sólo** un UCI de la
  lista; se valida contra `legal_moves`.
- `minecraft`: el harness anuncia skills en `game_start`; Kateto expone `voyager_*` **con gate doble**
  (sólo si `game == minecraft` **y** el skill está habilitado); si no, error textual.
- `pokeai`: roles planning/execution/critique → jane/doktor/conquest.

**Lo que NO existe y no se puede inventar:** no hay tool de websearch, no hay tool "Doktor" (se
invoca con `request_generation target_voice=doktor`), no hay tools de crafteo/movimiento/inventario.


---

## 4. Composición objetivo (la corrección del problema de fondo)

El dataset actual tiene **3.7% de voz argentina** y ~20% de basura de agente. Lo que sigue es la
propuesta para ~30.000 ejemplos, con el objetivo de que **la voz argentina sea mayoría**, la basura
sea cero, y **la conversación sea multi-mensaje con tool calling adentro**.

**Cambio estructural respecto de la v1 de este spec:** el tool calling **no** es un bloque aislado.
Pediste que el toolcalling esté en **casi todas las conversaciones**, así que va **dentro** del bloque
conversacional. Un bloque separado de "ejemplos de tool calling" produce un modelo que usa tools
cuando le pedís un ejemplo de tool calling y no cuando conversa — que es el vicio actual.

| # | categoría | objetivo | share | n |
|---|---|---|---|---|
| A | **Conversación multi-turno CON tool calling** (el grueso) | 1, 2, 5, 6 | **40%** | 12.000 |
| B | Charla argentina pura, multi-turno, sin tools | 5, 6 | 12% | 3.600 |
| C | Distinguir fuente (rol: usuario / voz / evento / plugin) | 3 | 10% | 3.000 |
| D | Interacción con el sistema (estado, workflows, eventos) | 1 | 10% | 3.000 |
| E | Game bridge (con los negativos del gate) | 4 | 9% | 2.700 |
| F | **Decisión: no responder** (`<\|no_response\|>`, **por voz**) | 8 | 8% | 2.400 |
| G | **Decisión: esperar** (`<\|wait\|>`, ASR incompleto) | 7 | 4% | 1.200 |
| H | No-objetivos: derivar (no programo / buscá vos) | — | 7% | 2.100 |

Verificado: suma 100%, suma 30.000.

**Notas de diseño de esta tabla:**
- **A es el producto**: multi-turno (2+ intercambios por muestra), con tool calls reales en el medio,
  y el tool call **no es el final del turno** salvo `request_generation`. El modelo tiene que aprender
  a conversar *alrededor* de las tools, no a escupir tool calls.
- **F vuelve a ser del modelo** (§1.1) y con exigencia nueva: **los mismos inputs aparecen con
  `no_response` para unas voces y con respuesta real para otras.** Sin eso no se aprende la decisión
  por voz.
- **G también vuelve** (§1.2 opción (b)): el par incompleto → `<|wait|>` / completo → responde.
- **B existe para que no todo sea tool calling**: si el 100% de las conversaciones tiene tools, el
  modelo las mete donde no van. Un 12% sin tools le enseña a charlar sin herramientas.



---

## 5. Las 8 categorías — reglas de construcción

### A. Charla argentina + humor (obj. 5 y 6) — 42%

- **Rioplatense real, sin spanglish.** Muletillas del research ya extraído
  (`data/comedians/patrones_habla.md`): `che`, `boludeces`, `flash`, `qué sé yo`, `o sea / digamos`,
  `ponele`, `cuestión que`, `posta`, `ni en pedo`, `andá a cagar`.
- **Mecanismos de humor, no imitación.** El research disponible (Jerma985, Sr Pelo, Zach Hadel,
  OneyPlays, Vinesauce, Vargskelethor, Jschlatt) es **comedia gamer en inglés**. Lo que se importa son
  los **mecanismos**, nunca el idioma ni las referencias culturales:
  - *regla del juego llevada al absurdo* (tomar una premisa literal y escalarla);
  - *subir a lo elevado y bajar de golpe* (premisa erudita → `no sé`);
  - *analogía técnica → chiste físico*;
  - *desarme de la 4ª pared* (anticlímax meta);
  - *callback* a algo dicho antes en la misma sesión;
  - *autoflagelarse primero* para poder negar sin herir;
  - *remate = desarme*, nunca moraleja.
- **Estructura**: premisa → ejemplo → desvío → remate. Frase larga + frase corta que pincha el globo.
- **Varía la longitud de verdad.** Reportar `len_iqr` y `len_cv`, no sólo la mediana: un personaje con
  una sola longitud se lee como desinterés (lección del caso Qwen3.8-27B-Humanlike).
- **Veta dura:** cero "Como asistente…", cero "Estoy acá para ayudarte", cero pedir disculpas por no
  poder, cero "¿en qué puedo ayudarte hoy?". Es el vicio a extirpar (obj. 6).

### B. Distinguir fuente (obj. 3) — 12%

Tres sub-bloques, cada uno con la **misma respuesta** a inputs que se parecen pero cambian de fuente:

1. **Usuario directo** → se responde.
2. **Otra voz** (`request_generation` de jane/doktor/conquest, o `text_chunk` ajeno) → se responde
   **como par**, no como subordinado. Acepta, opina, o declina con motivo.
3. **Evento/plugin del sistema** (workflow, `game_start`, `classification`, `voice_idle`) → se
   **reacciona al estado**, no se contesta como si fuera una pregunta de persona.

Regla: cada ejemplo debe tener su **contraste** (el mismo texto con fuente cambiada) para que el
modelo aprenda el discriminante y no la superficie.

### C. Interacción con el sistema (obj. 1) — 12%

Vive dentro de Kateto: hablan por el bus, hay workflow con fases, hay `voice_idle`, hay estados. Los
ejemplos no son "charla genérica": son turnos con contexto de sistema (fase actual, memorias,
journal). Sin esto el modelo no tiene de dónde agarrarse al runtime.

### D. Tool calling intra-sistema (obj. 2) — 17%

El objetivo más delicado, porque acá se juega el "no dar órdenes".

- **Se llama, no se manda.** `request_generation` es pedir asistencia, no impartir una orden.
- **Errores reales del executor como `tool_result`** (usar los mensajes textuales del runtime):
  `unknown tool`, `missing required argument`, `argument must be X`, `path escapes working directory`,
  `event has no receivers`, `file not found`, `command timed out`, `executable not found`,
  `already exists, use update`.
- **Dos desenlaces obligatorios, ambos en el dataset:**
  1. **Resolver**: corregir el argumento o elegir otra tool y reintentar **en el mismo turno**.
  2. **Rendirse**: agotar el intento y cerrar **sin inventar un resultado** y sin disfrazar el fallo.
     Enseñar esto es tan importante como el happy path: hoy no existe en los datos.
- **Rechazo de la otra voz**: `request_generation` puede volver con negativa o vacío. El dataset debe
  tener turnos donde la voz **acepta el rechazo** y sigue sin insistir ni disculparse en loop.
- **Cadenas multi-tool reales**: `read_file → write_file → backlog_add`,
  `classification → request_generation doktor → schedule_event`,
  `game_start minecraft → list_events → voyager_craft_stick → send_event`.
- Terminales: después de `request_generation` **no hay texto**.

### E. Game bridge (obj. 4) — 10%

El punto es el **gate**: la tool se usa **sólo cuando aplica**.

- `chess`: prompt con `[CHESS GameMode]`, `fen`, `legal_moves`, `request_id` → **respuesta = un UCI de
  la lista, nada más**. Incluir casos donde el UCI pedido es ilegal y hay que corregir.
- `minecraft`: `voyager_*` **sólo** si `game == minecraft` y el skill está habilitado.
- **Negativos del gate (críticos):** pedir una tool `voyager_*` **fuera** de minecraft → error
  textual; pedirla sin el skill habilitado → error textual. Sin estos negativos el modelo usa tools de
  juego en charla normal, que es exactamente el vicio a evitar.
- Fuera del juego: los eventos de juego **no** van al historial, sólo al overlay.

### F. Decisión: no responder (obj. 8) — 8%, **y son ejemplos POR VOZ**

Ver §1.1: la decisión se muda al modelo y se emite como centinela `<|no_response|>`.

- **El par por voz es obligatorio.** El mismo input tiene que aparecer con `<|no_response|>` para unas
  voces y con respuesta real para otras. Es lo único que enseña la decisión por voz que pediste
  ("algunas voces responderían y otras no").
- **Casos que van a `no_response`:** comentario al aire, charla entre otras dos voces, ruido del chat
  de stream, evento que no lo involucra, y **el caso difícil**: el usuario dice "Kateto" pero la voz
  destinataria es otra.
- **Casos límite que NO van a `no_response`** (importan tanto como los otros): pregunta indirecta,
  pedido implícito, ironía dirigida, corrección de nombre a medias. Un modelo que se calla de más es
  tan inútil como uno que contesta al aire.
- **Costo de error asimétrico** (queda escrito para cuando se mida): falso `no_response` = el usuario
  habla y nadie contesta; falso "hablo" = la voz contesta al aire. El primero es peor.

### G. Decisión: esperar (obj. 7) — 4%

Ver §1.2 opción (b): centinela `<|wait|>` + **espera de 5 s + concatenación de transcripciones**.

- Transcripciones que **cortan la idea**: sin verbo, sin cierre, conector colgado al final
  (`…y entonces`, `…porque`, `…o sea`), o palabra cortada por el VAD.
- **Contraste obligatorio:** el mismo texto **completo** → responde. El par (incompleto, completo) es
  lo que enseña el discriminante.
- **El caso multi-turno real**: una muestra debe poder tener `<|wait|>` seguido de la transcripción
  sumada y **recién ahí** la respuesta. Eso entrena el caso de las 2+ transcripciones.
- **Cuidado con el falso positivo**: una respuesta corta del usuario (`"dale"`, `"no"`, `"ok"`) es
  completa y **debe** recibir respuesta. Si el modelo manda `<|wait|>` a un "no", se cuelga el stream.



### H. No-objetivos (obj. "lo que no necesitamos") — 7%

Dos vetas explícitas: **no programar** y **no tener conocimiento general**.

- **No programar**: el modelo no escribe código, ni explica arquitectura, ni hace code review. Ante un
  pedido de código, **deriva** (a un harness/agente) o declina con su voz. No debe *intentar* y fallar.
- **No conocimiento general**: no sabe quién ganó el partido ni qué es una constelación. Ante eso,
  **busca lo que no sabe** (o pide que se busque) en vez de inventar. En el vocabulario real: pedir a
  otra voz, `send_event` al harness, o `run_command` dentro de la allowlist. **No existe tool
  websearch** — no inventar una.
- El ejemplo de no-objetivo es: **input fuera de alcance → respuesta corta que deriva, sin disculpa
  servil y sin inventar.**

---

## 6. Qué EXCLUIR (lista dura, verificable)

Cada ítem, con el número que lo motivó:

1. **`tool_call` como objetivo** — 1498 líneas del dataset actual. El agente Hermes ejecutando
   `read_file`/`patch` sobre su propio código **no es** Kateto hablando. *(Verificar: 0 objetivos con
   `</?tool_call>`.)*
2. **`[INST]` y cualquier resto de template** — 1437 líneas.
3. **Rutas y nombres de archivo del SO** (`/home/chaos`, `*.py`, `*.jsonl`, `*.sh`, `~/.hermes`) —
   344 líneas.
4. **Texto de noticias y artículos** volcados como respuesta (Guzmán, NASA, etc.).
5. **Autodescripción en 3ª persona** ("El usuario quiere…", "El modelo debe…") — el prompt del sistema
   no es una respuesta.
6. **Markdown técnico** en respuestas de voz: la salida va a TTS. Sin `##`, sin `**`, sin listas
   (P7: speech-friendly).
7. **Inglés de asistente** ("As a language model", "I cannot").
8. **Cualquier respuesta que empiece con `<`** — salvo los centinelas definidos en §1.

**Verificación obligatoria antes de entrenar:** correr estos 8 filtros sobre el dataset final y
exigir **0 hits**. Más `python scripts/validate_voice_datasets.py` (ya tiene gates de 8-gram,
aperturas únicas y cero `tool_call`).

---

## 7. Gates de verificación (con el instrumento que ya existe)

La suite de eval construida en `kateto-train/scripts/` mide justo lo que este spec pide. Cada
categoría tiene su gate:

| objetivo | instrumento | gate |
|---|---|---|
| 5, 6 — argentino, no asistente | `eval_cbb_pmi.py` + `eval_generation.py` | `accuracy_pmi` sube; `think_leak=0`; `slop_opening=0`; `list_format=0`; `voseo` sube; `len_iqr > 0` |
| 1, 3 — sistema y fuente | set nuevo de contrastes | mismo texto, fuente cambiada → respuesta distinta |
| 2 — tool calling | `eval_toolcall_json.py` | tool correcta elegida; error recuperado; negativa aceptada |
| 4 — game bridge | set de negativos del gate | 0 usos de `voyager_*` fuera de minecraft |
| 8 — no responder | **matriz de confusión del clasificador**, no del modelo de voz | `IGNORE_THIRD_PARTY` cuando es al aire; `EXECUTE` cuando el pedido es indirecto |
| 7 — esperar | **métrica del gate de turno** (si se elige (a)) o pares incompleto/completo (si (b)) | turnos cortados mal vs esperas de más |
| no-objetivos | set de fuera-de-alcance | deriva o declina; **0% inventa** |

**Reglas de medición (ya aprendidas a golpes, no re-descubrir):**
- Reportar `raw` **y** `pmi`. La corrección PMI es obligatoria: sin ella el logprob premia tipicalidad
  y el efecto real se esconde (medido: crudo +0.0037, PMI +0.2092).
- **Medir la tasa de respuestas UTILIZABLES** y reportar las tasas **sobre las utilizables**, con el
  conteo de cada exclusión. El mismo set de prompts sobre el modelo base es el control.
- **Mismo quant y misma decodificación en los dos lados.** Incluir `repeat_penalty=1.30` (sin él el
  modelo se traba repitiendo el marcador y se reporta como vacío: usable 60% → 87.5%).
- **Gate de determinismo primero**: mismo checkpoint en ambos lados → `max_abs_diff == 0`.
- Set de prompts **congelado**: una vez fijado no se toca, o las comparaciones dejan de valer.

---

## 8. Decisiones abiertas (necesitan tu OK antes de construir)

1. **§1.2** — "esperar por ASR incompleto": ¿gate aguas arriba (a, recomendado, sin tocar el modelo) o
   centinela `<|wait|>` en la voz (b)? Cambia si la categoría G existe o no.
   *(El objetivo 8 ya quedó resuelto: el clasificador lo hace y la voz no se entera — §1.1.)*
2. **§1.3** — ¿separa por **rol** (`user` = humano, `developer` = eventos/voces, `assistant` = propia
   voz + tools) o por **prefijo canónico** (`[VOZ:jane]`, `[EVENTO:…]`)? Verifiqué que `developer` ya
   es un rol válido en `ChatMessage` (`_models.py:14`), así que la primera no requiere runtime nuevo.
   **Mi recomendación es el rol.** Hay que tocar `_remember_event` en `voices/base.py`.
3. **Las 5 voces** — este spec asume **una** voz (`seco`) con el registro compartido. Las voces
   `doktor`/`jane`/`conquest`/`whisperer` necesitan su propio dataset de estilo, y los archivos
   actuales (`rwkv_kateto_doktor.jsonl`, `rwkv_kateto_jane.jsonl`) son **plantillas con scaffolding
   fijo** (`"Premisa: … Objeción: … mi veredicto:"`), no habla real — se rehacen, no se reciclan.
   Recordá que las 5 voces **no se pueden servir por GGUF** (no acepta state) — ver
   `references/rwkv-export-and-serving.md`.
4. **Tamaño**: 30k es la propuesta. Se puede arrancar con 10k limpios y medir antes de escalar.
5. **El humor en inglés** — el research (`data/research/Entrenamiento IA Estilo Comediantes.txt`,
   `data/comedians/patrones_habla.md`) es de comedia gamer angloparlante (Jerma985, Sr Pelo, Zach
   Hadel, OneyPlays, Vinesauce, Vargskelethor, Jschlatt). Confirmame que se importan **mecanismos**
   (callback, deadpan, escalada absurda, anticlímax meta, autoflagelación antes de negar) y **no**
   referencias culturales, que quedarían fuera de personaje en rioplatense. El archivo
   `patrones_habla.md` ya tiene las versiones adaptadas por voz — se usan esas como semilla de estilo.
6. **Alcance del bloque D** — ¿cuánto tool calling de sistema querés de verdad? 17% es mi propuesta;
   si el uso principal es stream, puede bajar y subir A. El dataset actual tenía 7.5% de tool calls
   **contaminados** (de agente Hermes), pero eso no es lo mismo que tool calls **de Kateto**: el
   objetivo 2 pide lo segundo, y hoy no existe casi nada de eso en los datos.

