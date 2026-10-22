#!/usr/bin/env python3
"""Generate debate-speech rows with zero tool_calls (todo wave1-todo5, followup-B diversified).

Deterministic, no LLM calls, no network. Emits Kateto envelope rows shaped:
  {"conversations": [{"from": "human", "value": "<|im_user|>..."},
                     {"from": "gpt", "value": "<|im_start|>voice\\n...\\n<|im_end|>"}]}

Follow-up B: single-skeleton design replaced by a bank of >=60 debate-move
frames (concesion+refutacion, ejemplo concreto, pregunta retorica, reduccion
al absurdo, dato/analogia, cierre) x >=50 topics x 5 voices, with varied
human prompt phrasings (no single fixed "DEBATE:" shape).

NOTE (wave0-todo1 verdict): ChatML ids are MISSING from rwkv_vocab_v20230424,
so <|im_user|>/<|im_start|>/<|im_end|> are PLAIN-TEXT delimiters (multi-token).
Todo 10's converter parses them as text; never assume single-token ids here.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

VOICES = ["seco", "streamer", "jane", "doktor", "whisperer"]

# 3 openers per voice (15 distinct) — voseo, zero servilismo.
VOICE_OPEN = {
    "seco": [
        "Corto y al pie.",
        "Sin vueltas, mirá.",
        "Te la hago corta.",
    ],
    "streamer": [
        "¡Gente, escuchen esto, chat!",
        "¡Paren todo que se viene el debate!",
        "Chat, presten atención a esto.",
    ],
    "jane": [
        "Te lo explico tranquila, pero firme.",
        "Mirá, hablemos en serio un minuto.",
        "Te lo digo con calma y con datos.",
    ],
    "doktor": [
        "Veamos la evidencia, sin humo.",
        "Analicemos esto con la cabeza fría.",
        "Vamos a los hechos, dejemos el ruido.",
    ],
    "whisperer": [
        "Te lo digo bajito, pero seguro.",
        "Acercate que te cuento la posta.",
        "Entre nos, escuchá esto.",
    ],
}

TOPICS = [
    "La inteligencia artificial es peligrosa",
    "Python es mejor que JavaScript",
    "El mate amargo es superior al mate con azúcar",
    "Las redes sociales destruyeron la comunicación",
    "El capitalismo es inevitable",
    "El trabajo remoto llegó para quedarse",
    "La universidad pública debe ser arancelada",
    "El fútbol moderno perdió la pasión del potrero",
    "Messi es más grande que Maradona",
    "Boca es más pueblo que River",
    "El VAR arruinó el fútbol",
    "Los e-sports son un deporte de verdad",
    "La carne argentina es la mejor del mundo",
    "El asado se hace solo con leña",
    "La pizza con ananá es un crimen",
    "El fernet se toma con coca, nada más",
    "El dulce de leche no se negocia",
    "La inflación se arregla dolarizando",
    "El cepo al dólar protege a los trabajadores",
    "Hay que bajar los impuestos aunque caiga el gasto",
    "El Estado debe intervenir en los precios",
    "La grieta argentina es irreconciliable",
    "Los medios mienten más de lo que informan",
    "X (Twitter) es mejor que Instagram para debatir",
    "TikTok embrutece a los pibes",
    "El streaming mató al cine",
    "El trap es la nueva cumbia",
    "El rock nacional murió en los 90",
    "Es mejor alquilar que comprar casa",
    "El auto propio es un gasto al pedo en CABA",
    "La bicicleta es el mejor transporte urbano",
    "Hay que prohibir los celulares en las escuelas",
    "La educación virtual no sirve para nada",
    "Aprender inglés es obligatorio hoy",
    "La IA va a dejar sin trabajo a los programadores",
    "El open source siempre le gana al software pago",
    "Linux es mejor que Windows para laburar",
    "El iPhone no justifica lo que sale",
    "La cripto es el futuro del dinero",
    "El plazo fijo le gana a cualquier inversión",
    "Emigrar es la única salida para los jóvenes",
    "Quedarse en Argentina es un acto de fe",
    "El campo sostiene al país",
    "La industria nacional hay que protegerla",
    "El turismo interno es mejor que viajar afuera",
    "La Patagonia es el mejor destino del país",
    "El verano en la costa es un choreo",
    "Tomar mate en el trabajo baja la productividad",
    "El café de especialidad es puro marketing",
    "La comida vegana puede ser rica",
    "Hay que comer menos carne por el planeta",
    "El gimnasio es mejor que correr",
    "Dormir la siesta aumenta el rendimiento",
    "Los libros en papel le ganan al ebook",
    "La música en vinilo suena mejor",
    "El peronismo es el hecho maldito del país burgués",
]

STANCES = ["A FAVOR", "EN CONTRA"]

# 30 frames per stance (60 total). Moves: C=concesion+refutacion, E=ejemplo
# concreto, P=pregunta retorica, A=reduccion al absurdo, D=dato/analogia,
# S=cierre. Slots: {topic_lc} {topic} {r1} {r2}. All voseo, no servilismo.
FAVOR_FRAMES = [
    # C concesion+refutacion
    "Te concedo que {topic_lc} suena polémico, pero bancame un segundo: {r1} Y cuando lo mirás sin prejuicio, {r2}",
    "Mirá, entiendo al que duda de que {topic_lc}, tiene sus grises. Pero pesá esto: {r1} Al final del día, {r2}",
    "Sí, te doy que {topic_lc} no es para todo el mundo. Ahora, sacá la cuenta conmigo: {r1} Y vas a ver que {r2}",
    "Concedido: {topic_lc} tiene sus críticos con buenos modales. Pero los modales no pagan cuentas: {r1} Fijate que {r2}",
    "Acepto que {topic_lc} incomoda, el cambio siempre incomoda. Pero la incomodidad no es argumento: {r1} Lo real es que {r2}",
    # E ejemplo concreto
    "Te doy un ejemplo bien concreto: {r1} Eso pasa porque {topic_lc}, y lo ves en la calle todos los días. Además, {r2}",
    "Mirá lo que pasó con tu vecino, con tu primo, con cualquiera: {r1} ¿Casualidad? No, es que {topic_lc}. Y ojo, {r2}",
    "Andá al barrio y preguntá: {r1} Ahí tenés la prueba de que {topic_lc}, sin paper de por medio. Y sumale que {r2}",
    "Caso real, sin invento: {r1} Decime si eso no confirma que {topic_lc}. Encima, {r2}",
    "El otro día lo vi con mis propios ojos: {r1} Por eso te digo que {topic_lc}. Y no es todo: {r2}",
    # P pregunta retorica
    "¿En serio me vas a decir que {topic_lc} no se sostiene? Mirá: {r1} ¿O me vas a negar también que {r2}",
    "Te pregunto de frente: si {topic_lc} fuera mentira, ¿cómo explicás que {r1}? Pensalo, porque además {r2}",
    "¿Cuántas pruebas más necesitás? {r1} ¿Te parece poco para admitir que {topic_lc}? Y encima {r2}",
    "Decime una cosa: ¿quién sale perdiendo si aceptamos que {topic_lc}? Nadie, porque {r1} Al contrario, {r2}",
    "¿O sea que vas a mirar para otro lado mientras {r1}? Dale, admití que {topic_lc}. Si no, {r2}",
    # A reduccion al absurdo
    "Si negás que {topic_lc}, entonces también tenés que negar lo obvio: {r1} ¿Ves lo ridículo que suena? Por eso, {r2}",
    "Llevemos tu duda al extremo: si {topic_lc} fuera falso, {r1} Un delirio total. Cae de maduro que {r2}",
    "Dale, sigamos tu lógica hasta el final: {r1} ¿Te das cuenta del absurdo? Entonces {topic_lc}, y {r2}",
    "Si {topic_lc} no fuera cierto, mañana mismo {r1} Como eso no pasa ni va a pasar, {r2}",
    "Negar que {topic_lc} es como decir que el agua no moja: {r1} Reíte, pero {r2}",
    # D dato/analogia
    "Los números no mienten: {r1} Esa es la base material de que {topic_lc}. Y en criollo, {r2}",
    "Es como el truco: si tenés el ancho de espadas, jugalo. Acá el ancho es que {r1} Por eso {topic_lc}, y {r2}",
    "Hacé la cuenta vos mismo: {r1} ¿Viste? Da a favor de que {topic_lc}. Y no solo eso, {r2}",
    "Compará con lo que pasa en la región: {r1} El patrón confirma que {topic_lc}. Aparte, {r2}",
    "Es física, no opinión: {r1} Con esa base, {topic_lc} se sostiene solo. Y remato: {r2}",
    # S cierre
    "Cierro con esto: {topic_lc}, punto. {r1} Si alguien trae un argumento mejor, lo escucho; mientras tanto, {r2}",
    "En definitiva, {topic_lc} y no hay mucha más vuelta: {r1} Bancátela, porque {r2}",
    "Mi veredicto es claro: {topic_lc}. {r1} Podés putearme en los comentarios, pero {r2}",
    "Última palabra: {topic_lc}. {r1} El que avisa no traiciona: {r2}",
    "Me planto acá: {topic_lc}. {r1} Discutamos con datos, no con slogans: {r2}",
]

CONTRA_FRAMES = [
    # C concesion+refutacion
    "Te concedo que {topic_lc} suena lindo en teoría, pero en la cancha: {r1} Ahí se cae, porque {r2}",
    "Entiendo el atractivo de creer que {topic_lc}, yo también lo creí. Hasta que vi que {r1} Desde entonces, {r2}",
    "Sí, {topic_lc} tiene su marketing, te lo venden envuelto en papel de regalo. Abrilo y vas a ver que {r1} En el fondo, {r2}",
    "Concedido: {topic_lc} convence al que no la vivió. Pero el que la vivió te dice: {r1} Conclusión: {r2}",
    "Acepto que {topic_lc} tiene un punto, uno solo. Pero ese punto no tapa que {r1} Así que no, porque {r2}",
    # E ejemplo concreto
    "Te doy un ejemplo bien concreto: {r1} ¿Y me venís a decir que {topic_lc}? Dale, porque además {r2}",
    "Mirá tu bolsillo a fin de mes: {r1} Esa es la prueba de que {topic_lc} es verso. Y sumale que {r2}",
    "Preguntale a cualquiera que labure de verdad: {r1} Ahí se termina el cuento de que {topic_lc}. Encima, {r2}",
    "Caso real, sin invento: {r1} Decime si eso no demuele que {topic_lc}. Y no es todo: {r2}",
    "El otro día lo vi con mis propios ojos: {r1} Por eso ni en pedo {topic_lc}. Y ojo, {r2}",
    # P pregunta retorica
    "¿En serio me vas a decir que {topic_lc}? Mirá alrededor: {r1} ¿O eso tampoco lo ves? Porque {r2}",
    "Te pregunto de frente: si {topic_lc} fuera cierto, ¿por qué {r1}? No cierra, y {r2}",
    "¿Cuántas veces más te la van a vender? {r1} ¿Te parece poco para admitir que {topic_lc} es humo? Fijate que {r2}",
    "Decime una cosa: ¿quién gana guita con que creas que {topic_lc}? Exacto: {r1} Por eso, {r2}",
    "¿O sea que vas a aplaudir mientras {r1}? Abrí los ojos, porque {r2}",
    # A reduccion al absurdo
    "Si {topic_lc} fuera verdad, entonces {r1} ¿Ves lo ridículo que suena? Por eso, {r2}",
    "Llevemos tu idea al extremo: si {topic_lc}, mañana {r1} Un delirio total. Cae de maduro que {r2}",
    "Dale, sigamos esa lógica hasta el final: {r1} ¿Te das cuenta del absurdo? Entonces no, porque {r2}",
    "Creer que {topic_lc} es como creer que llueve para arriba: {r1} Reíte, pero {r2}",
    "Si aceptamos que {topic_lc}, también aceptemos que {r1} Como eso es un chiste, {r2}",
    # D dato/analogia
    "Los números no mienten: {r1} Esa es la lápida de que {topic_lc}. Y en criollo, {r2}",
    "Es como querer tapar el sol con la mano: {r1} Así de flojo es creer que {topic_lc}. Además, {r2}",
    "Hacé la cuenta vos mismo: {r1} ¿Viste? No da ni cerca para {topic_lc}. Y no solo eso, {r2}",
    "Compará con lo que pasa en la región: {r1} El patrón demuele que {topic_lc}. Aparte, {r2}",
    "Es física, no opinión: {r1} Con esa base, {topic_lc} no se sostiene. Y remato: {r2}",
    # S cierre
    "Cierro con esto: {topic_lc} es verso, punto. {r1} Si alguien trae pruebas posta, lo escucho; mientras tanto, {r2}",
    "En definitiva, {topic_lc} no va ni para atrás: {r1} Bancátela, porque {r2}",
    "Mi veredicto es claro: en contra de que {topic_lc}. {r1} Podés putearme en los comentarios, pero {r2}",
    "Última palabra: {topic_lc}, ni loco. {r1} El que avisa no traiciona: {r2}",
    "Me planto acá: {topic_lc} es un cuento. {r1} Discutamos con datos, no con slogans: {r2}",
]

assert len(FAVOR_FRAMES) >= 30 and len(CONTRA_FRAMES) >= 30  # 60-frame bank

# Enlarged reason banks (12 each), all voseo/plain, no tool_call text.
REASONS_A = [
    "fijate quién gana guita con ese discurso y quién la pone",
    "preguntale a cualquiera que labure de verdad, no al que tuitea desde el country",
    "la historia argentina ya probó esa receta y terminó en quilombo",
    "los números no cierran por ningún lado, hacé la cuenta vos mismo",
    "en el barrio se ve clarito lo que los papers tardan años en admitir",
    "compará con lo que pasa en la región y se cae solo el argumento",
    "el bolsillo de fin de mes no miente nunca, miralo vos mismo",
    "los que la vivieron te lo cantan de memoria, escuchalos",
    "ningún gráfico te va a decir lo que te dice la cola del súper",
    "preguntá en el laburo y vas a ver que nadie te chamuya",
    "la experiencia del día a día pesa más que mil editoriales",
    "mirá quién financia ese relato y vas a entender todo",
]
REASONS_B = [
    "y si me equivoco, vení con pruebas, no con indignación de red social",
    "el que te dice lo contrario suele venderte algo, ojo al piojo",
    "no compro pescado podrido: quiero verlo funcionar antes de aplaudir",
    "la ironía es que los más fervorosos son los que menos la vivieron",
    "acá no hay neutralidad posible, ponete de un lado y bancatela",
    "si tu argumento necesita insultarme para sostenerse, ya perdiste",
    "traé datos o traé silencio, pero no me traigas slogans",
    "el tiempo me va a dar la razón, guardá este mensaje",
    "discutamos en la cancha, no en los comentarios con foto de anime",
    "si te pica, rascate, pero los hechos son los hechos",
    "no me corro ni un centímetro: vení con argumentos o no vengas",
    "y al que no le guste, que arme su propio debate y me invite",
]

# Varied human prompt phrasings (no single fixed "DEBATE:" shape).
# Slots: {topic} {stance}. Every variant embeds the stance token verbatim
# so stance balance stays grep-able.
PROMPT_TEMPLATES = [
    "<|im_user|>DEBATE: '{topic}' — {stance}",
    "<|im_user|>A debatir: {topic}. Tu postura: {stance}.",
    "<|im_user|>Tema: {topic} | Postura a defender: {stance}",
    "<|im_user|>Che, discutamos esto: {topic}. Posicionate {stance}.",
    "<|im_user|>Dale, defendé esto {stance}: {topic}.",
    "<|im_user|>Mesa de debate — moción: {topic}. Bancá la postura {stance}.",
    "<|im_user|>Te tiro un tema picante: {topic} ({stance}). ¿Qué decís?",
    "<|im_user|>Postura {stance} sobre: {topic}. Convenceme.",
]


def build_response(topic, stance, frame, r1, r2, opener):
    topic_lc = topic[0].lower() + topic[1:]
    body = frame.format(
        topic_lc=topic_lc, topic=topic,
        r1=r1[0].upper() + r1[1:] + ".", r2=r2[0].upper() + r2[1:] + ".",
    )
    return f"{opener} {body}"


def format_row(prompt, response, voice):
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": f"<|im_start|>{voice}\n{response}\n<|im_end|>"},
        ]
    }


def completion_of(row):
    gpt = next(c for c in row["conversations"] if c["from"] == "gpt")["value"]
    body = gpt.replace("<|im_start|>", "").replace("<|im_end|>", "")
    lines = [ln for ln in body.splitlines() if ln.strip()]
    return "\n".join(lines[1:]) if len(lines) > 1 else ""


def validate_row(row):
    """Returns (ok, reason). Rejects empty responses and any tool_call text."""
    try:
        convs = row["conversations"]
        if completion_of(row).strip() == "":
            return False, "empty-response"
        blob = json.dumps(row, ensure_ascii=False)
        if "tool_call" in blob:
            return False, "tool_call-leak"
        return True, ""
    except (KeyError, StopIteration, TypeError, AttributeError, IndexError):
        return False, "malformed"


def opening_key(completion, n=15):
    return " ".join(completion.split()[:n])


def diversity_report(rows):
    """Returns (ok, report_str). Gates: no exact dups; >=95% unique
    completions; >=60 distinct opening lines; stance 40-60%; voices>=5,
    topics>=50 covered."""
    blobs = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in rows]
    completions = [completion_of(r) for r in rows]
    prompts = [r["conversations"][0]["value"] for r in rows]
    stances = ["A FAVOR" if "A FAVOR" in p else ("EN CONTRA" if "EN CONTRA" in p else "?") for p in prompts]
    voices = [r["conversations"][1]["value"].split("\n")[0].replace("<|im_start|>", "") for r in rows]
    # topic recovery: prompt embeds full topic string verbatim
    topics_hit = set()
    for p in prompts:
        for t in TOPICS:
            if t in p:
                topics_hit.add(t)
                break
    n = len(rows)
    dup_rows = n - len(set(blobs))
    uniq_comp = len(set(completions)) / n if n else 0
    openings = len({opening_key(c) for c in completions})
    sc = Counter(stances)
    favor_pct = 100 * sc.get("A FAVOR", 0) / n if n else 0
    contra_pct = 100 * sc.get("EN CONTRA", 0) / n if n else 0
    toolcall = sum(1 for b in blobs if "tool_call" in b)
    checks = [
        ("zero-exact-duplicate-rows", dup_rows == 0, f"dup_rows={dup_rows}"),
        ("unique-completions>=95%", uniq_comp >= 0.95, f"{uniq_comp * 100:.1f}%"),
        ("distinct-openings>=60", openings >= 60, f"openings={openings}"),
        ("tool_call==0", toolcall == 0, f"tool_call={toolcall}"),
        ("stance-favor-40-60%", 40 <= favor_pct <= 60, f"A FAVOR={favor_pct:.1f}%"),
        ("stance-contra-40-60%", 40 <= contra_pct <= 60, f"EN CONTRA={contra_pct:.1f}%"),
        ("voices>=5", len(set(voices)) >= 5, f"voices={len(set(voices))}"),
        ("topics>=50", len(topics_hit) >= 50, f"topics={len(topics_hit)}"),
    ]
    lines = [f"rows={n}"] + [
        f"[{'OK' if ok else 'FAIL'}] {name}: {detail}" for name, ok, detail in checks
    ]
    return all(ok for _, ok, _ in checks), "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=800)
    ap.add_argument("--output", default="data/debate_speech.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    count = max(500, min(1000, args.count))
    rng = random.Random(args.seed)

    # Balanced plan: voices round-robin, stances 50/50, topics spread evenly,
    # frames/reasons/openers/prompts sampled without replacement per cycle.
    n_voices = len(VOICES)
    per_voice = count // n_voices
    rows, seen = [], set()
    for vi, voice in enumerate(VOICES):
        n_here = per_voice + (1 if vi < count % n_voices else 0)
        topics_cycle = (TOPICS * ((n_here // len(TOPICS)) + 1))[:n_here]
        rng.shuffle(topics_cycle)
        for j in range(n_here):
            stance = STANCES[(j + vi) % 2]  # alternate, offset per voice
            frames = FAVOR_FRAMES if stance == "A FAVOR" else CONTRA_FRAMES
            topic = topics_cycle[j]
            # draw a unique (topic, stance, frame, r1, r2, opener, prompt) combo
            frame, r1, r2, opener, tmpl = frames[0], REASONS_A[0], REASONS_B[0], VOICE_OPEN[voice][0], PROMPT_TEMPLATES[0]
            for _attempt in range(200):
                frame = rng.choice(frames)
                r1, r2 = rng.choice(REASONS_A), rng.choice(REASONS_B)
                opener = rng.choice(VOICE_OPEN[voice])
                tmpl = rng.choice(PROMPT_TEMPLATES)
                key = (topic, stance, frame, r1, r2, opener, tmpl)
                if key not in seen:
                    seen.add(key)
                    break
            prompt = tmpl.format(topic=topic, stance=stance)
            resp = build_response(topic, stance, frame, r1, r2, opener)
            row = format_row(prompt, resp, voice)
            ok, reason = validate_row(row)
            if not ok:
                raise SystemExit(f"VALIDATOR-FAIL row voice={voice}: {reason}")
            rows.append(row)
    rng.shuffle(rows)
    rows = rows[:count]

    ok, report = diversity_report(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    Path("out").mkdir(exist_ok=True)
    Path("out/diversity_report.log").write_text(report, encoding="utf-8")
    voices_used = sorted({r["conversations"][1]["value"].split("\n")[0].replace("<|im_start|>", "") for r in rows})
    log = (
        f"rows={len(rows)} topics-catalogados={len(TOPICS)} "
        f"frames={len(FAVOR_FRAMES) + len(CONTRA_FRAMES)} voices={','.join(voices_used)}\n"
        f"quarantined=0 tool_call_grep=0(se verifica con grep)\n"
    )
    Path("out/debate_counts.log").write_text(log, encoding="utf-8")
    print(report.strip())
    print(log.strip())
    if not ok:
        raise SystemExit("VALIDATOR-FAIL: diversity gates not met (see out/diversity_report.log)")


if __name__ == "__main__":
    main()
