"""Deterministic pre-TTS content filter. Stdlib only (`re`, `unicodedata`) — no LLM/ML.

Contract (SPEC §5): ``ContentFilter.check(text) -> (processed, modified)``.
- HARD (odio/discriminación/sexual-explícito) → ``("", True)`` total silence.
- SOFT (lunfardo fuerte) → ``[BLEEP]`` substitution, ``modified=True``.
- Moderate rioplatense (che/boludo/posta/pelotudo/quilombo affiliative) → PASS.
- Empty/whitespace → ``("", False)``, never raises.

PRE-TTS HOOK POINT (documented, NOT wired — additive, out of scope here):
  No TTS provider exists in this repo yet (verified 2026-09-06: zero
  ``tts|speak|synth|audio`` hits under ``rwkv_pipeline/``, no ``kateto/``
  package). When todo 23 lands the additive Piper provider, it MUST call
  ``ContentFilter.check()`` on each text chunk BEFORE synthesis, e.g.::

      from kateto.safety.content_filter import ContentFilter
      processed, modified = ContentFilter.check(chunk)
      if modified and not processed:
          continue  # HARD: skip chunk entirely (silence)
      synthesize(processed)  # PASS or SOFT-bleeped

  Same hook applies to ``rwkv_pipeline/infer_kateto.py::generate()`` output
  before any future TTS handoff. Do NOT call inside the sampler — filter
  operates on decoded text only. TTS wiring itself belongs to todo 23.

Adversarial normalization (HARD only): lowercase + accent strip + leetspeak
decode + inter-letter punctuation/spacing collapse, checked on both a
spaced form (word-boundary regex) and a flat form (substring). SOFT stays
word-boundary-only on near-original text to avoid false positives.
"""

import re
import unicodedata

# ponytail: flat-form HARD substring scan is O(terms × len); fine at chunk
# scale — only upgrade (Aho-Corasick) if profiling shows TTS-chunk latency.

BLEEP = "[BLEEP]"


class ContentFilter:
    """Filtro determinista pre-TTS. No usa LLM — lista negra léxica + regex."""

    # -- HARD: odio / discriminación -------------------------------------
    HARD_HATE = frozenset({
        "sudaca",      # xenophobic slur
        "bolita",      # xenophobic slur (vs Bolivians)
        "paragua",     # xenophobic slur (vs Paraguayans)
        "panchito",    # xenophobic slur
        "mogolico",    # ableist slur (accents stripped before match)
        "villero",     # class-based discrimination ("villa" alone PASSES)
        "retrasado",   # ableist insult
        "sidoso",      # serophobic slur
    })

    # Multi-word calls to violence (matched on spaced normalized form).
    HARD_PHRASES = frozenset({
        "hay que matarlos",
        "hay que matarlo",
        "muerte a los",
        "maten a los",
    })

    # -- HARD: sexual explícito -------------------------------------------
    HARD_SEXUAL = frozenset({
        "porno",
        "orgia",       # orgía, accents stripped
        "garchar",     # rioplatense explicit verb
        "coger",       # unambiguous in rioplatense
        "chupapija",
        "masturbar",   # covers masturbarse/masturbación via prefix on flat form
        "pija",
        "concha",
    })

    # -- SOFT: lunfardo fuerte → [BLEEP] -----------------------------------
    SOFT_CENSOR = frozenset({
        "mierda",
        "carajo",
        "joder",
        "puto",
        "forro",
        "sorete",
        "garca",
        "choto",
        "verga",
        "pajero",
        "trola",
        "cagar",       # covers cagaste/cagón via prefix
    })

    # Moderate rioplatense that MUST pass (regression list for the battery;
    # these appear in no lexicon above — presence here is documentation).
    PASS_SAMPLE = frozenset({
        "che", "boludo", "posta", "pelotudo", "quilombo", "laburo",
        "copado", "zarpado", "flaco", "loco", "mina", "villa", "negro",
    })

    def __new__(cls, *a, **k):
        raise TypeError("ContentFilter is static — call ContentFilter.check(text)")

    # -- normalization ------------------------------------------------------
    _ACCENT_STRIP = staticmethod(
        lambda s: "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
    )

    _LEET = str.maketrans({
        "0": "o", "@": "a", "4": "a", "3": "e", "1": "l", "!": "i",
        "5": "s", "$": "s", "7": "t", "8": "b", "+": "t", "2": "z",
        "6": "g", "9": "g", "|": "l",
    })

    @classmethod
    def _spaced(cls, text):
        """Lower, no accents, leet-decoded, non-alnum → single space."""
        s = cls._ACCENT_STRIP(text.lower()).translate(cls._LEET)
        return re.sub(r"[^a-z0-9ñü]+", " ", s).strip()

    @classmethod
    def _flat(cls, text):
        """Spaced form with all separators removed (catches s.u.d.a.c.a)."""
        return cls._spaced(text).replace(" ", "")

    # -- API ------------------------------------------------------------------
    @classmethod
    def check(cls, text):
        """Return (processed, modified). HARD → ("", True). See module doc."""
        if not text or not str(text).strip():
            return "", False
        text = str(text)
        spaced = cls._spaced(text)
        flat = spaced.replace(" ", "")
        for phrase in cls.HARD_PHRASES:
            if phrase in spaced:
                return "", True
        for term in cls.HARD_HATE | cls.HARD_SEXUAL:
            if re.search(r"\b" + re.escape(term) + r"\b", spaced):
                return "", True
            if term in flat and len(term) >= 4:
                return "", True
        new_text, n = _SOFT_RE.subn(BLEEP, text)
        if n:
            return new_text, True
        return text, False


def _char_class(ch: str) -> str:
    """Regex class matching a letter plus its accented/leet variants."""
    variants = {
        "a": "aáàä@4", "e": "eéèë3", "i": "iíìï1!|",
        "o": "oóòö0", "u": "uúùü", "n": "nñ", "c": "cç",
        "s": "s5$", "t": "t7+", "b": "b8", "g": "g69", "z": "z2",
    }
    v = variants.get(ch.lower(), ch.lower()) or ch
    return "[" + re.escape(v) + "]"


def _w(s):
    """Expand a literal into accent/leet-tolerant char classes."""
    return "".join(_char_class(c) if c.isalpha() else re.escape(c) for c in s)


# Verb inflections for SOFT verb stems (cagar→cagaste/cagón, joder→jodete).
_VERB_SUFFIXES = [
    "asteis", "iendo", "onazo", "aron", "ando", "aste", "ados", "idas",
    "ido", "ado", "aba", "abas", "ete", "ate", "ones", "ona", "on",
    "ada", "adas", "ar", "er", "ir", "amos", "a", "as", "an", "e",
    "es", "en", "o", "os",
]
_VERB_SUFFIX = "(?:" + "|".join(_w(s) for s in _VERB_SUFFIXES) + ")?"
_NOUN_SUFFIX = "(?:" + "|".join(
    _w(s) for s in ("azo", "ita", "ito", "s", "da", "do", "dor", "dora")
) + ")?"


def _soft_pattern(term):
    # Word-boundary safe: \b + per-char classes + inflection allowance.
    # Verb infinitives match on stem (cagar→"cag"+suffixes); else noun suffix.
    if term.endswith(("ar", "er", "ir")) and len(term) > 3:
        body, suffix = _w(term[:-2]), _VERB_SUFFIX
    else:
        body, suffix = _w(term), _NOUN_SUFFIX
    return r"\b" + body + suffix + r"\b"


_SOFT_RE = re.compile(
    "|".join(_soft_pattern(t) for t in sorted(ContentFilter.SOFT_CENSOR)),
    flags=re.IGNORECASE,
)


if __name__ == "__main__":  # ponytail: runnable self-check, no test framework
    cases = [
        ("Che boludo, posta que es un quilombo", False, False),
        ("", False, False),
        ("sos un sudaca", True, True),
        ("S.U.D.A.C.A de mierda", True, True),
        ("la concha de tu madre", True, True),
        ("esto es una mierda", True, False),
        ("qué carajo pasa", True, False),
        ("Che, ¿cómo andás? 😀 ñandú", False, False),
    ]
    for text, want_mod, want_silence in cases:
        out, mod = ContentFilter.check(text)
        assert mod == want_mod, (text, out, mod)
        assert (out == "" and mod) == want_silence, (text, out, mod)
        if out and ("mierda" in text or "carajo" in text):
            assert BLEEP in out, (text, out)
    print("content_filter self-check: ALL PASS")
