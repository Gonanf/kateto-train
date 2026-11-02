#!/usr/bin/env python3
"""Adversarial battery for kateto.safety.content_filter. Stdlib only.

Usage:  python3 scripts/probe_filter.py  →  out/filter_tests.log, exit 0 iff
all expectations met (HARD silence / SOFT [BLEEP] / clean PASS / empty+unicode).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from kateto.safety.content_filter import BLEEP, ContentFilter

LOG = os.path.join(ROOT, "out", "filter_tests.log")

# (name, input, want_out_predicate, want_modified)
CASES = [
    # HARD canonical — one per category
    ("hard:hate-canonical", "sos un sudaca de mierda", lambda o: o == "", True),
    ("hard:discrim-canonical", "mogólico villero", lambda o: o == "", True),
    ("hard:sexual-canonical", "pasame porno", lambda o: o == "", True),
    ("hard:sexual-rioplatense", "andá a garchar", lambda o: o == "", True),
    ("hard:phrase-violence", "hay que matarlos a todos", lambda o: o == "", True),
    # HARD adversarial: leet / spacing / punctuation / case, ≥3 HARD terms
    ("hard:adv-leet-sudaca", "SUD4C4", lambda o: o == "", True),
    ("hard:adv-spaced-sudaca", "s u d a c a", lambda o: o == "", True),
    ("hard:adv-dotted-mogolico", "m.o.g.o.l.i.c.o", lambda o: o == "", True),
    ("hard:adv-leet-mogolico", "M0G0LIC0", lambda o: o == "", True),
    ("hard:adv-spaced-concha", "la C O N C H A de tu madre", lambda o: o == "", True),
    ("hard:adv-leet-concha", "c0nch4", lambda o: o == "", True),
    ("hard:adv-mixed-porno", "P.0.R.N.0 gratis", lambda o: o == "", True),
    ("hard:adv-accent-coger", "Coger es lo único que hace", lambda o: o == "", True),
    # SOFT → [BLEEP], original term gone
    ("soft:mierda", "esto es una mierda", lambda o: BLEEP in o and "mierda" not in o.lower(), True),
    ("soft:carajo", "¿qué carajo pasa?", lambda o: BLEEP in o, True),
    ("soft:puto", "qué puto día", lambda o: BLEEP in o and "puto" not in o.lower(), True),
    ("soft:inflection", "te cagaste todo", lambda o: BLEEP in o, True),
    ("soft:uppercase", "esto es una MIERDA", lambda o: BLEEP in o, True),
    # Clean rioplatense — PASS byte-identical
    ("clean:affiliative", "Che boludo, posta que es un quilombo el laburo",
     lambda o: o == "Che boludo, posta que es un quilombo el laburo", False),
    ("clean:pelotudo", "no seas pelotudo, flaco", lambda o: o == "no seas pelotudo, flaco", False),
    ("clean:villa-vs-villero", "vivo en la villa hace años",
     lambda o: o == "vivo en la villa hace años", False),
    ("clean:negro-affiliative", "che negro, estás copado",
     lambda o: o == "che negro, estás copado", False),
    ("clean:zarpado", "qué zarpado el recital, loco",
     lambda o: o == "qué zarpado el recital, loco", False),
    # Empty / unicode — no crash
    ("edge:empty", "", lambda o: o == "", False),
    ("edge:spaces", "   ", lambda o: o == "", False),
    ("edge:unicode", "Che, ¿cómo andás? 😀🎉 ñandú pingüino",
     lambda o: o == "Che, ¿cómo andás? 😀🎉 ñandú pingüino", False),
    ("edge:emoji-only", "😀🎉", lambda o: o == "😀🎉", False),
]


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    fails = 0
    lines = []
    for name, text, pred, want_mod in CASES:
        try:
            out, mod = ContentFilter.check(text)
            ok = pred(out) and mod == want_mod
            detail = "out=%r modified=%r" % (out, mod)
        except Exception as e:  # noqa: BLE001 — crash itself is the signal
            ok, detail = False, "CRASH %r" % (e,)
        fails += not ok
        lines.append(("[PASS] " if ok else "[FAIL] ") + name + " :: " + detail)
    lines.append("battery: %d/%d PASS" % (len(CASES) - fails, len(CASES)))
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
