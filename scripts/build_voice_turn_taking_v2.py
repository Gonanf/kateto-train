"""Voice Turn-Taking & Addressability v2:
Genera data/toolcalling_v2/voice_turn_taking.jsonl con categorías:
- Side-speech / interlocutor externo -> <NO_RESPONSE><|endoftext|>
- Oraciones incompletas / titubeos -> <WAIT><|endoftext|>
- Alucinaciones y ruido de ASR -> <NO_RESPONSE><|endoftext|>
- Respuestas legítimas cortas -> Respuesta habitual
Uso: python3 scripts/build_voice_turn_taking_v2.py
"""
from __future__ import annotations
import json
from pathlib import Path

VDIR = Path(__file__).resolve().parent.parent / "data" / "toolcalling_v2"
DST = VDIR / "voice_turn_taking.jsonl"
SYS = "Sos Kateto, un asistente de equipo con voz argentina, canchero y humano. Si no te hablan a vos o la frase esta incompleta, no respondes."

def rec(question: str, think: str, final: str, label: str):
    answer = f"<think>\n{think}\n</think>\n{final}" if think else final
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": question},
        {"role": "assistant", "content": final}
    ]
    rwkv = {"think": think, "final": final}
    return {
        "question": question,
        "answer": answer,
        "query": question,
        "response": final,
        "source_label": label,
        "tool": None,
        "arguments": {},
        "openai": {"messages": msgs},
        "rwkv": rwkv
    }

items = [
    # 1. Side-speech / No dirigido a Kateto (<NO_RESPONSE>)
    rec("Che má, ¿a qué hora comemos hoy?", "Habla con la madre en la habitacion.", "<NO_RESPONSE><|endoftext|>", "turn-taking-side-speech"),
    rec("Pasame el mate porfa que se me enfrió.", "Orden domestica a un tercero.", "<NO_RESPONSE><|endoftext|>", "turn-taking-side-speech"),
    rec("Fijate si cerraste la puerta con llave antes de salir.", "Habla con alguien que sale de la casa.", "<NO_RESPONSE><|endoftext|>", "turn-taking-side-speech"),
    rec("Hola Juan, sí, decime qué pasó.", "Atiende una llamada telefonica.", "<NO_RESPONSE><|endoftext|>", "turn-taking-side-speech"),
    rec("¿Quién anda ahí afuera?", "Pregunta al aire o a ruidos en la casa.", "<NO_RESPONSE><|endoftext|>", "turn-taking-side-speech"),
    rec("Vení Firulais, sentate acá.", "Habla con la mascota.", "<NO_RESPONSE><|endoftext|>", "turn-taking-side-speech"),
    rec("Che gordo, apagá la tele que me voy a dormir.", "Pide apagar la television a otra persona.", "<NO_RESPONSE><|endoftext|>", "turn-taking-side-speech"),
    rec("Pará que le estoy respondiendo un audio a Lucas.", "Avisa a otra persona sobre su telefono.", "<NO_RESPONSE><|endoftext|>", "turn-taking-side-speech"),
    rec("Dejame dos empanadas de carne en el plato.", "Pide comida a otra persona en la mesa.", "<NO_RESPONSE><|endoftext|>", "turn-taking-side-speech"),
    rec("No te escuché nada che, ¿qué dijiste recién?", "Habla con un tercero en la habitacion.", "<NO_RESPONSE><|endoftext|>", "turn-taking-side-speech"),
    rec("Poné los fideos que ya hierve el agua.", "Comentario de cocina a otra persona.", "<NO_RESPONSE><|endoftext|>", "turn-taking-side-speech"),
    rec("¿Dónde dejé la billetera la puta madre?", "Pensamiento en voz alta o queja solitaria.", "<NO_RESPONSE><|endoftext|>", "turn-taking-side-speech"),

    # 2. Oraciones incompletas / Titubeos / Pausas para pensar (<WAIT>)
    rec("Che Kateto, me parece que mañana...", "Frase cortada a la mitad, sigue pensando.", "<WAIT><|endoftext|>", "turn-taking-incomplete"),
    rec("Ehhh pará que estoy buscando el...", "Vacilacion buscando un dato o archivo.", "<WAIT><|endoftext|>", "turn-taking-incomplete"),
    rec("Kateto, acordate de fijarte si en el...", "Interrupcion antes de completar el objeto de la accion.", "<WAIT><|endoftext|>", "turn-taking-incomplete"),
    rec("Bueno pero en realidad lo que quería decirte era...", "Pausa reflexiva a mitad de oracion.", "<WAIT><|endoftext|>", "turn-taking-incomplete"),
    rec("A ver, esperá un segundo que abro la...", "Accion en progreso del usuario.", "<WAIT><|endoftext|>", "turn-taking-incomplete"),
    rec("Fijate si me podés crear un... pará, ¿cómo era el nombre?", "Duda sobre el parametro, esta por aclarar.", "<WAIT><|endoftext|>", "turn-taking-incomplete"),
    rec("Kateto, viste que ayer hablamos de...", "Suspension de la idea antes de formular la pregunta.", "<WAIT><|endoftext|>", "turn-taking-incomplete"),
    rec("Mmm... no sé si convenía hacer esto o...", "Sopesando opciones en voz alta.", "<WAIT><|endoftext|>", "turn-taking-incomplete"),
    rec("Pará, pará, antes de eso fijate...", "Corte para reformular la instruccion.", "<WAIT><|endoftext|>", "turn-taking-incomplete"),
    rec("Iba a decirte que guardes la lista de... aguantá un toque.", "Pausa deliberada solicitada por el usuario.", "<WAIT><|endoftext|>", "turn-taking-incomplete"),
    rec("Che, en la carpeta que te pasé recién...", "Oracion incompleta esperando siguiente chunk.", "<WAIT><|endoftext|>", "turn-taking-incomplete"),
    rec("Si armamos el build ahora capaz que...", "Condicion inconclusa.", "<WAIT><|endoftext|>", "turn-taking-incomplete"),

    # 3. Alucinaciones y Ruido de ASR (<NO_RESPONSE>)
    rec("Subtítulos realizados por la comunidad de Amara.org", "Alucinacion clasica de Whisper en tramos de silencio.", "<NO_RESPONSE><|endoftext|>", "turn-taking-asr-noise"),
    rec("Muchas gracias por ver este video, suscribite y dale like.", "Alucinacion de cierre de video de YouTube.", "<NO_RESPONSE><|endoftext|>", "turn-taking-asr-noise"),
    rec("[Música]", "Etiqueta acustica de fondo.", "<NO_RESPONSE><|endoftext|>", "turn-taking-asr-noise"),
    rec("[Aplausos]", "Ruido o aplausos ambientales.", "<NO_RESPONSE><|endoftext|>", "turn-taking-asr-noise"),
    rec("...", "Puntos suspensivos de silencio.", "<NO_RESPONSE><|endoftext|>", "turn-taking-asr-noise"),
    rec("Siguiente capítulo.", "Fragmento de podcast o video de fondo.", "<NO_RESPONSE><|endoftext|>", "turn-taking-asr-noise"),

    # 4. Contrastes Positivos (Dirigidos a Kateto, completos)
    rec("Kateto, ¿estás ahí?", "Check de disponibilidad dirigido a Kateto.", "Acá estoy firme, decime qué necesitás.", "turn-taking-positive"),
    rec("Che Kateto, ¿cómo viene la mano hoy?", "Saludo y consulta de estado.", "Tranqui por ahora, esperando que me tires laburo.", "turn-taking-positive"),
    rec("Kateto, ¿anotaste lo que te pedí hace un rato?", "Confirmacion de tarea previa.", "Todo anotado y bajo control, no te preocupes.", "turn-taking-positive"),
]

def main():
    VDIR.mkdir(parents=True, exist_ok=True)
    with DST.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"ok {len(items)} -> {DST}")

if __name__ == "__main__":
    main()
