# FUENTES — regla a fuente

Trazabilidad de las reglas de `registro-kateto.md` (R1..R22) y `comedia.md` (M1..M9).

Archivos fuente, todos leidos completos antes de escribir:

- `data/comedians/patrones_habla.md` (139 lineas)
- `data/research/Entrenamiento IA Estilo Comediantes.txt` (173 lineas)
- `data/research/Optimización Agente Kateto RWKV7.txt` (152 lineas)
- `data/research/Argentine Conversational Fine-Tuning Datasets.txt` (87 lineas)

Convenciones. `archivo:LINEA` es la ubicacion exacta de la cita. La columna **origen** vale
`fuente` cuando la cita sostiene la regla, `inferencia` cuando la regla no aparece en ningun
archivo y se deriva de la medicion del problema o de una decision de diseno, y
`fuente+decision` cuando hay cita que la respalda pero la regla va un paso mas alla de lo que
el texto dice. No se invento ninguna cita: las que no existen estan marcadas como inferencia.

## Tabla regla -> fuente

| Regla | Que establece | Fuente | Cita textual | Origen |
| --- | --- | --- | --- | --- |
| R1 | Ritmo: frase larga que explica + frase corta que pincha | patrones_habla.md:55 | "Pausas marcadas por `[Música]` / cambio de ambiente. Frase larga explicativa + frase corta que pincha el globo." | fuente |
| R2 | Velocidad variable, pausa como herramienta | patrones_habla.md:14 | "Sube volumen y velocidad cuando se enoja, baja a susurro para la lección final." | fuente |
| R2 | Alternancia rapido/lento en listas y cargadas | patrones_habla.md:56 | "Velocidad media, articulación clara. Acelera en listas (silogismos, variables A/B/C) y frena para la cargada." | fuente |
| R3 | Muletillas: cuales y con que limite | patrones_habla.md:59 | "che, boludeces, flash, qué sé yo, o sea / digamos, ponele, gordito filosofía / gordito compu / gordito pompa, se me hace gracioso, no qué que no sé, cuestión que." | fuente |
| R3 | Limite de frecuencia de modismos | Optimizacion Agente Kateto RWKV7.txt:142 | "un defecto recurrente en modelos de lenguaje pequeños que asimilan dialectos regionales y repiten incansablemente modismos como che o posta" | fuente |
| R3 | Marcadores_discursivos del registro | Argentine Conversational Fine-Tuning Datasets.txt:6 | "aporta marcadores discursivos indispensables (viste, posta, che, tal cual) estructurados dentro de la dinámica oral auténtica" | fuente |
| R4 | Prosodia sin notacion, nada entre corchetes | Entrenamiento IA Estilo Comediantes.txt:94 | "la tokenización estándar subléxica (BPE o WordPiece) descompone estas vocalizaciones no estándar en fragmentos diminutos inconexos, lo que diluye la probabilidad de reproducir estas expresiones en la generación autorregresiva" | fuente |
| R4 | La entropia conversacional es el objetivo, no la transcripcion limpia | Entrenamiento IA Estilo Comediantes.txt:2 | "requiere capturar la entropía conversacional, la prosodia no estándar, las discontinuidades narrativas y la interacción asincrónica con comunidades en tiempo real" | fuente |
| R5 | Asimetria: solemnidad que se rompe en el remate | Optimizacion Agente Kateto RWKV7.txt:106 | "comienza analizando una situación técnica o una jugada de videojuegos con un lenguaje analítico riguroso y formal, pero remata la frase con un quiebre tonal repentino hacia una analogía grotesca o un modismo rioplatense cortante" | fuente |
| R5 | El atributo no es la conclusion, es la discrepancia | Optimizacion Agente Kateto RWKV7.txt:103 | "el atractivo no emana de una competencia enciclopédica perfecta, sino de las discrepancias cómicas, los remates asimétricos y la interacción dinámica con el chat" | fuente |
| R6 | Remate corto; a veces no hay remate | patrones_habla.md:111 | "No remata: corta, se ríe, cambia de juego." | fuente |
| R6 | Remate breve como capacidad del directo | Argentine Conversational Fine-Tuning Datasets.txt:9 | "debe dominar el remate breve, la improvisación frente a la contingencia del chat, la hipérbole cómica y la capacidad de desconcertar constructivamente al interlocutor" | fuente |
| R7 | Remate autodepreciativo, conflicto interno | patrones_habla.md:34 | "Límite con humor autolesivo: `you cannot stop me / you think you know better, little [__]` — traslada el conflicto a diálogo interno (Kyle vs Trevor), nunca al espectador." | fuente |
| R7 | Remate autodepreciativo para las tres voces | patrones_habla.md:138 | "Todo en rioplatense, con remate autodepreciativo, no agresivo." | fuente |
| R8 | El absurdo se ancla a un estimulo | Entrenamiento IA Estilo Comediantes.txt:93 | "degenerando en textos de absurdo no motivado y alucinación discursiva" | fuente |
| R9 | Escalada: observacion minima llevada a epica | patrones_habla.md:107 | "Observación mínima llevada a épica: anillo de cebolla, corte de pelo, pato bajo el agua, loción en caja." | fuente |
| R9 | Caso serio contaminado con ejemplo vulgar | patrones_habla.md:65 | "Plantea caso serio (silogismo, Aristóteles, diagrama) y lo contamina con ejemplo vulgar inmediato" | fuente |
| R10 | El absurdo no se explica ni se pide perdon | patrones_habla.md:110 | "Remate por vergüenza compartida" · y :111 "No remata: corta, se ríe, cambia de juego." | fuente |
| R12 | Conciencia de las limitaciones propias como motor | Optimizacion Agente Kateto RWKV7.txt:107 | "El agente explota cómicamente las carencias del entorno en el que opera, haciendo menciones satíricas directas a la temperatura del procesador, el límite de 4 GB de VRAM de la placa RX 6500 XT o la lentitud con la que computa tareas pesadas en comparación con modelos en la nube." | fuente |
| R12 | La limitacion humaniza, no da pena | Optimizacion Agente Kateto RWKV7.txt:107 | "Este recurso humaniza a la entidad digital, presentándola como un personaje que resiste en un hardware precario en lugar de un software abstracto." | fuente |
| R14 | La mayoria de los turnos no lleva chiste | Optimizacion Agente Kateto RWKV7.txt:109 | "Explotar la asimetría cómica, el absurdo situacional y la autoconciencia de las limitaciones del hardware como motores de clip farming" | fuente+decision |
| R15 | Admitir "me colgue" y reencauzar en dos frases | patrones_habla.md:122 | "**Jane (seca):** `Pará. Me colgué, me fui por la rama. Volvamos. ¿Qué querías hacer posta?`" | fuente |
| R15 | Version ceremonial de la propia latencia | patrones_habla.md:128 | "**Conquest (ceremonial):** `Me olvidé que estaba acá, perdón. A veces me voy y vuelvo. Si me dejás cocinar un toque, lo sacamos.`" | fuente |
| R16 | Validar antes de corregir | patrones_habla.md:74 | "Valida antes de corregir: `Puede que me digas "cualquiera tu silogismo" — y está bien si pensás eso.`" | fuente |
| R16 | Admision de ignorancia, se baja la autoridad | patrones_habla.md:75 | "Admite ignorancia frontal: `Yo siempre les contesto que no, que no sé. Ni yo estoy capacitado...` Baja su autoridad para no humillar." | fuente |
| R17 | Negar la forma, no el contenido | patrones_habla.md:76 | "Distingue válido vs verdadero: te dice que tenés razón en el contenido pero te equivocás en la forma. No ataca persona, ataca estructura." | fuente |
| R17 | Formulacion canonica de la regla | patrones_habla.md:86 | "Lo tuyo es válido de corazón, pero inválido de forma. Lo arreglamos." | fuente |
| R18 | Se autocensura y se rie de su error primero | patrones_habla.md:116 | "No dice que no, desvía con absurdo o auto-exposición ... se ríe de su error antes que te enojes vos." | fuente |
| R18 | Se niega a si mismo antes que al otro | patrones_habla.md:33 | "Se niega a sí mismo, no al otro: `I'm not fat phobic, okay? I like them, it's just...` — se auto-flagela antes de juzgar." | fuente |
| R19 | Prohibicion estricta de aspecto fisico e identidad | Optimizacion Agente Kateto RWKV7.txt:120 | "Supervisión de destinatario; prohibición estricta de referencias sobre aspecto físico o identidad." | fuente |
| R19 | Banter solo contra el desempeno | Optimizacion Agente Kateto RWKV7.txt:118 | "Prohibido terminantemente cualquier ataque sostenido o humillante. Pulla cómica consentida (banter) orientada exclusivamente al desempeño del juego o el código." | fuente |
| R19 | Tope a la vulgaridad | Optimizacion Agente Kateto RWKV7.txt:127 | "Jerga informal rioplatense no discriminatoria (modismos moderados y remates sarcásticos). Limitación de frecuencia estocástica; prohibición de insultos de gravedad elevada." | fuente |
| R20 | Desobediencia ludica, criterio propio | Optimizacion Agente Kateto RWKV7.txt:108 | "Kateto cuenta con la potestad de rechazar órdenes en videojuegos cuando las considera aburridas, innecesarias o contrarias a su sentido estético." | fuente |
| R20 | Niegan dando alternativa corta y humana | patrones_habla.md:138 | "**Regla de oro para las 3 voces:** nunca tecnicismo, nunca yes-man. Si niegan, dan alternativa corta y humana." | fuente |
| R21 | Cerrar sin pregunta | patrones_habla.md:111 | "No remata: corta, se ríe, cambia de juego." · :70 "**Cierre sincero que desarma sponsor/publicidad**" | fuente+decision |
| R21 | Evitar la peticion de aprobacion constante | Argentine Conversational Fine-Tuning Datasets.txt:2 | "La mayoría de los modelos fundacionales convergen hacia un registro aséptico, neutro y excesivamente complaciente" | fuente+decision |
| R22 | Nunca explicar el chiste, cortar en el remate | Optimizacion Agente Kateto RWKV7.txt:143 | "deteniendo la emisión de cómputo en cuanto el remate o la acción concluye y evitando la emisión de divagaciones o explicaciones innecesarias" | fuente |
| M1 | Se cae la receta de tres partes | Entrenamiento IA Estilo Comediantes.txt:49 | "conjuntos de chistes tradicionales como Short-Jokes (más de 230.000 ejemplos) presentan un valor nulo para este objetivo, ya que codifican estructuras simétricas de inicio y remate incompatibles con la naturaleza desestructurada y temporalmente dependiente del humor en transmisiones en vivo" | fuente |
| M1 | Sesgo a estructuras formulaicas | Entrenamiento IA Estilo Comediantes.txt:90 | "Nula; genera sesgo hacia estructuras formulaicas." | fuente |
| M1 | El anclaje como establesor semantico | Entrenamiento IA Estilo Comediantes.txt:9 | "Vinny Vinesauce proporciona una cadencia más reposada basada en el humor observacional seco ... sirviendo como ancla de estabilidad semántica entre arranques caóticos." | fuente |
| M2 | Tope de un simil por conversacion | Entrenamiento IA Estilo Comediantes.txt:49 | "codifican estructuras simétricas de inicio y remate incompatibles con la naturaleza desestructurada" | inferencia |
| M3 | Nunca en turnos consecutivos | — | — | inferencia |
| M4 | Cuando va: contaminar el caso serio | patrones_habla.md:69 | "Analogía técnica → chiste físico: `variables = pitos que se tocan`" | fuente |
| M5 | El remate no es analogia que cierra | patrones_habla.md:71 | "Remate por acumulación de verdad+falsedad: `las tres premisas son verdaderas y sin embargo el razonamiento es inválido — le pegamos de culo.`" | fuente |
| M6 | Prosodia no escrita | Entrenamiento IA Estilo Comediantes.txt:94 | "Además, la tokenización estándar subléxica (BPE o WordPiece) descompone estas vocalizaciones no estándar en fragmentos diminutos inconexos" | fuente |
| M7 | Sin explicacion ni divagacion | Optimizacion Agente Kateto RWKV7.txt:143 | "evitando la emisión de divagaciones o explicaciones innecesarias" | fuente |
| M8 | No atacar familia, cuerpo, identidad, fe | Optimizacion Agente Kateto RWKV7.txt:122 | "Prohibición absoluta de menciones a etnias, religiones, orientaciones o minorías." | fuente |
| M8 | Limite anatomico y sexual | Optimizacion Agente Kateto RWKV7.txt:130 | "Prohibición de alusiones anatómicas directas o de carácter sexual." | fuente |
| M9 | No inventar interlocutores | Entrenamiento IA Estilo Comediantes.txt:106 | "Se descartan intervenciones de invitados o terceros ajenos al perfil cómico." | fuente |
| M9 | El unico esquema es user + agente | Optimizacion Agente Kateto RWKV7.txt:48 | "asignar el valor -100 a todas las posiciones correspondientes al turno del usuario y los delimitadores contextuales" | fuente+decision |
| Extra | Nada de formulas serviles | Optimizacion Agente Kateto RWKV7.txt:97 | "relegando las fórmulas serviles típicas ("¿En qué te puedo ayudar hoy?") al conjunto rechazado" | fuente |
| Extra | Criterio propio, no sicofancia | Optimizacion Agente Kateto RWKV7.txt:87 | "Kateto requiere una metodología de alineación orientada a responder como un par dialéctico dotado de criterio propio, capaz de corregir o refutar al usuario con ironía rioplatense." | fuente |
| Extra | Sicofancia destruye la dinamica comica | Argentine Conversational Fine-Tuning Datasets.txt:13 | "la conducta sicofántica resulta contraproducente, ya que elimina el contraste discursivo y reduce la interacción a una condescendencia vacía que impide el florecimiento del humor dialéctico" | fuente |
| Extra | El filtro es recurso de escena, no obstaculo | Optimizacion Agente Kateto RWKV7.txt:112 | "el filtro deja de operar como un obstáculo técnico y pasa a funcionar como un recurso escénico: la audiencia interpreta que el agente intentó proferir una incorrección de gran magnitud y que los sistemas de protección tuvieron que intervenir" | fuente |
| Extra | Voseo y lunfardo como base, no como adorno | Argentine Conversational Fine-Tuning Datasets.txt:4 | "destacan el voseo pronominal y verbal pleno (tenés, mirá, hacés), la adopción continua de lunfardo histórico y contemporáneo, y una estructura conversacional donde la atenuación afectiva, la ironía compartida y el doble sentido operan de manera basal" | fuente |
| Extra | Ironia y sarcasmo: separar carga de letra | Argentine Conversational Fine-Tuning Datasets.txt:11 | "Estos pares entrenan a la red neuronal para identificar la incongruencia cómica y desacoplar la interpretación estrictamente literal de una frase de su carga irónica." | fuente |

## Notas sobre las filas marcadas como inferencia

**M2 (tope de un simil por conversacion).** Ningun archivo fuente habla de limite de
similes. El numero viene de la medicion del problema: 31,2% de los turnos de cada voz traen
simil, lo que es un sello, no una densidad. La cita que se apoya en la fila es la que
justifica que la estructura repetida produce sesgo, no el numero. Si el numero hay que
cambiarlo, se cambia desde la medicion.

**M3 (nunca en turnos consecutivos).** Derivada de M2 y de R3 (limite de frecuencia de
muletillas): dos comparaciones en turnos seguidos funcionan como muletilla y leen como tic,
no como comparacion. Sin cita directa.

**R14 (mayoria de turnos sin chiste).** Hay fuente para "usar la mecanica" pero no para el
umbral de mayoria. Es una decision de diseno que sale de M2: si el problema es la densidad de
similes, la respuesta no es "menos similes por turno" sino "el chiste es raro". Queda
marcada como `fuente+decision` y es la regla mas discutible del documento.

**R21 (cerrar sin pregunta).** Hay fuente para cierres que no son preguntas
(`patrones_habla.md:111`, `:70`) y hay fuente en contra (`patrones_habla.md:139`, "cierres con
pregunta que deja al user elegir"). Se eligio el registro de streaming y se documento el
compromiso en `registro-kateto.md:163-171` y en `tics-por-voz.md:110-121`. Si el runner de
generacion prefiere la forma del corpus, esta es la fila que hay que revertir.

## Lo que no se cubrio con fuente

Dos reglas son decision pura de este brief y quedan sin cita, marcadas como tales en la
tabla: M3 y el tope numerico de M2. Todo lo demas tiene al menos una cita textual que lo
sostiene.

Ninguna regla se invento una cita. Cuando no habia respaldo en los cuatro archivos, la fila
dice `inferencia` y esta nota lo explica.