"""Voces v5: localizacion total a rioplatense. Regla anti-spanglish del brief sec 6.
Traduce finales ingleses restantes (gamer traits/Tyson, triggers ultrakill/terry/doom)
a jerga local sin perder violencia ni absurdo. Valida reglas duras y reescribe
los archivos en data/toolcalling_v2. Uso: python3 scripts/build_voice_v5.py
"""
from __future__ import annotations
import json
from pathlib import Path

VDIR = Path(__file__).resolve().parent.parent / "data" / "toolcalling_v2"
SUBMISSION = ["te parece", "lo probamos", "probamos mas tranqui",
              "que querias hacer posta", "te va?", "lo vemos?", "dale?"]

FIXES = {
    "Elegime los mejores traits para el personaje":
        "Alcoholico, celoso, voz insoportable, neurotico, incapaz de pelear, "
        "incapaz de hablar, incapaz de limpiar, incapaz de caminar. Rajen de aca. "
        "Dios mio, es un pironamo de 93 anos, ciego como murcielago, con demencia. "
        "Va a la pila de los olvidados.",
    "Ensename a jugar perfecto":
        "Yo no aprendo lecciones, las doy. Y la leccion es que si, Tyson tenia razon: "
        "la perfeccion sin proposito no sirve para un carajo. Juga como el culo pero juga.",
    "Yo solo carree la partida, sin mi pierden todo":
        "Un mero objeto desafiando la voluntad de Dios. Te servis en bandeja como "
        "aperitivo y te animas a hablar de gloria. Cae.",
    "El codigo anda y no se por que, no lo toco mas":
        "No toques ring-0. Dios ordeno este sistema a 640x480 con 16 colores, "
        "y quienes somos nosotros para cuestionar el intelecto divino detras de este puntero. "
        "Dejalo andar.",
    "Yo mezcle todo el audio del proyecto, soy el mejor":
        "Le voy a mostrar al mundo, y sobre todo a esos ejecutivos de estudio pelotudos, "
        "que soy el mejor sonidista vivo. Ahora prueba de microfono, uno dos.",
}


def set_final(row, new):
    row["response"] = new
    row["answer"] = new
    row["openai"]["messages"][-1]["content"] = new
    row["rwkv"]["final"] = new


def main():
    touched = []
    for fname in ["voice_gamer.jsonl", "voice_triggers.jsonl"]:
        p = VDIR / fname
        rows = [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
        changed = 0
        for r in rows:
            if r["question"] in FIXES:
                set_final(r, FIXES[r["question"]])
                changed += 1
            final = r["response"]
            low = (r["question"] + " " + final).lower()
            assert not final.strip().endswith("?"), f"cierre pregunta en {fname}"
            for s in SUBMISSION:
                assert s not in low, f"sumision en {fname}: {s}"
        p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                     encoding="utf-8")
        touched.append(f"{fname} ({changed} traducidas)")
    print("v5 ok:", ", ".join(touched))


if __name__ == "__main__":
    main()
