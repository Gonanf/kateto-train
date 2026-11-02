from __future__ import annotations
import json, subprocess
from pathlib import Path
from generate_v2_with_agy import validate_item, build_record, call_agy, VDIR

def expand_triggers_and_pelo():
    # 1. Triggers situacionales
    p_trig = """
Generá exactamente 10 ejemplos en JSON array para 'Triggers situacionales' de Kateto:
- Caso Terry Davis (código frágil que anda de milagro): "Do not touch ring-0. God decreed this operating system at 640x480 resolution with 16 colors..."
- Caso Ultrakill (alguien soberbio o que se agranda): "A mere object... defying the will of God?! You make an appetizer out of yourself and dare speak of glory. Fall."
- Caso MF DOOM (nombres en minúscula o mezcla de audio): "Remember ALL CAPS", "best soundman alive, now mic check one two".
- Caso NTVG (recomendación que el otro odia): "Mirá, te diría que le des una chance, pero no te va a gustar; chau."
- Caso BoJack (discusión en loop absurdo): "¡Erica! ¿Qué hacés con dos cabezas y un zapato en la mano?"
Reglas:
- 100% de compromiso con la cita o su traducción brutal al rioplatense.
- Cero relleno previo, sin cortesía.
- NUNCA cerrar con pregunta.
Devolvé solo JSON array:
[{"question": "...", "think": "...", "final": "..."}]
"""
    print("--- Generando Triggers ---")
    items = call_agy(p_trig)
    with (VDIR / "voice_triggers.jsonl").open("a", encoding="utf-8") as f:
        for it in items:
            q, fn, th = it.get("question", "").strip(), it.get("final", "").strip(), it.get("think", "")
            if q and fn and not validate_item(q, fn, None):
                f.write(json.dumps(build_record(q, th, None, {}, {}, fn, "voice-triggers"), ensure_ascii=False) + "\n")

    # 2. Sr Pelo
    p_pelo = """
Generá exactamente 8 ejemplos en JSON array para la voz 'Sr Pelo' de Kateto (reacción física, onomatopeyas, remate seco):
Reglas:
- Acotaciones físicas de sonido: [respiro asmático violento], [golpe seco a la mesa], [grito agudo], [resopla contra el micrófono].
- Acusación desmedida ante un error absurdo ("¿POR QUÉ TOCASTE ESO?").
- Remate seco sin pedir disculpas ni permiso. NUNCA terminar con pregunta.
Devolvé solo JSON array:
[{"question": "...", "think": "...", "final": "..."}]
"""
    print("--- Generando Sr Pelo ---")
    items = call_agy(p_pelo)
    with (VDIR / "voice_srpelo.jsonl").open("a", encoding="utf-8") as f:
        for it in items:
            q, fn, th = it.get("question", "").strip(), it.get("final", "").strip(), it.get("think", "")
            if q and fn and not validate_item(q, fn, None):
                f.write(json.dumps(build_record(q, th, None, {}, {}, fn, "voice-srpelo"), ensure_ascii=False) + "\n")

    print("Generación de Triggers y Sr Pelo completada.")

if __name__ == "__main__":
    expand_triggers_and_pelo()
