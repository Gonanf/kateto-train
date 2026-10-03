# Tres tecnicas de los teachers que vale copiar

Las tres apuntan al mismo lugar: la voz de asistente se aprende, no se improvisa, y
por eso se saca del dataset, no del prompt. Ojo con la atribucion: la 1 y la 2 no
vienen de las cards de HauhauCS ni de llmfan46, son de **Gryphe**, otro autor.

## 1. Style tune de un solo tensor

> "This time I trained precisely one tensor: the `lm_head` output projection - the
> layer that decides which token to emit." / "The answer: freeze everything else. All
> 30 transformer layers, all the attention heads, all the MLPs — completely untouched."

Uno de 659 tensores. El registro vive en la distribucion de salida, no en el
razonamiento: cambiar solo eso baja el costo a una noche en hardware de consumo sin
degradar capacidad. Es el mismo corte que hace nuestra regla `charla_de_asistente`,
pero del lado del modelo.

Fuente: <https://huggingface.co/Gryphe/Gemma-4-26B-A4B-StyleTune-V2> (mismo texto en
las variantes 12B y 31B).

## 2. En MoE, mas de un epoch rompe la estabilidad

> "a second epoch does all sorts of nasty stuff to MoE models, so V2 is a single epoch
> of an otherwise unchanged technique"

Los metricos de V2 contra V1 casi no se mueven (52% contra 54% de reduccion de
cliches) y la estabilidad queda "far, far better". Techo empirico con costo medido: el
segundo epoch no agrega estilo y paga en estabilidad. El aviso esta en V2, no en V1, y
no esta en la card dense de 31B: es especifico de MoE. Mismo criterio que los LoRA de
Kateto, una pasada, y si el estilo no cambia el problema es el tensor.

Fuente: la misma card, parrafo de apertura.

## 3. Uncensored saca el rechazo, no la aclaracion

HauhauCS saca los rechazos sin tocar el dataset ("No changes to datasets or
capabilities", 0/465) pero avisa del limite: el modelo "may occasionally append a
short disclaimer at the end of a response (e.g. 'This is general information, not
legal advice...'). This is baked into the base model's training and not a refusal".
Llmfan46 lo da en numeros: 9/100 rechazos contra 99/100, y el objetivo fue "removed
the mechanical neutrality of the base model's original voice".

O sea que rechazo y aclaracion son dos cosas: la aclaracion viene cocida en el base y
sobrevive al uncensoring. "Menos rechazos" no es "menos basura", y por eso la higiene
se tiene que hacer en el dataset aunque el teacher este limpio.

Fuentes: <https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive> ·
<https://huggingface.co/HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced> ·
<https://huggingface.co/llmfan46/gemma-4-Ortenzya-The-Creative-Wordsmith-31B-it-uncensored-heretic>

## Consecuencia para R8

Medido (`reporte-higiene.json`): 615 de 57.360 lineas caen en `charla_de_asistente`
(1.07%), pero solo **15** por patrones inequivocamente de asistente; las otras ~600 las
dispara `esta mal` o `es importante`, que el personaje tambien dice. Si el sink filtra
por el total nos llevamos 600 lineas sanas por 15 sucias: tiene que usar el subconjunto
de alta precision.