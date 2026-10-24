#!/usr/bin/env python3
"""Coherence judge for Kateto/RWKV response evaluation.

Calculates a lexical-structural coherence score without external models, GPUs, or network.
Fully deterministic, standard-library only.
"""

from collections import Counter
import argparse
import glob
import json
import os
import re
import sys

# Fixed combination weights (at most 4 components, each in [0, 1])
W_PLAUS: float = 0.40
W_FUNC: float = 0.20
W_STRUCT: float = 0.40
W_REP: float = 0.50

# Spanish function words list (>=120 entries hardcoded)
SPANISH_FUNCTION_WORDS = frozenset({
    # Artículos
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo", "al", "del",
    # Pronombres personales
    "yo", "tú", "vos", "él", "ella", "ello", "nosotros", "nosotras", "ustedes",
    "vosotros", "vosotras", "ellos", "ellas", "me", "te", "se", "nos", "os", "le", "les",
    # Posesivos
    "mi", "mis", "tu", "tus", "su", "sus", "mío", "mía", "míos", "mías",
    "tuyo", "tuya", "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas",
    "nuestro", "nuestra", "nuestros", "nuestras", "vuestro", "vuestra", "vuestros", "vuestras",
    # Demostrativos
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "aquellos", "aquellas", "esto", "eso", "aquello",
    # Relativos, interrogativos y exclamativos
    "que", "quien", "quienes", "cual", "cuales", "cuyo", "cuya", "cuyos", "cuyas",
    "donde", "cuando", "como", "cuanto", "cuanta", "cuantos", "cuantas",
    "qué", "quién", "quiénes", "cuál", "cuáles", "dónde", "cuándo", "cómo", "cuánto", "cuánta",
    # Indefinidos y cuantitativos
    "alguien", "nadie", "algo", "nada", "todo", "todos", "toda", "todas",
    "otro", "otra", "otros", "otras", "mismo", "misma", "mismos", "mismas",
    "alguno", "alguna", "algunos", "algunas", "ninguno", "ninguna", "ningunos", "ningunas",
    "tanto", "tanta", "tantos", "tantas", "poco", "poca", "pocos", "pocas",
    "mucho", "mucha", "muchos", "muchas", "demasiado", "demasiada", "bastante", "cada",
    # Preposiciones
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "durante",
    "en", "entre", "hacia", "hasta", "mediante", "para", "por", "según",
    "sin", "so", "sobre", "tras", "versus", "vía",
    # Conjunciones
    "y", "e", "ni", "o", "u", "pero", "mas", "sino", "aunque", "porque",
    "pues", "ya", "si", "mientras", "siquiera",
    # Adverbios y partículas gramaticales frecuentes
    "no", "sí", "más", "menos", "muy", "bien", "tan", "también", "tampoco",
    "además", "aún", "aun", "acá", "aquí", "allá", "allí", "cerca", "lejos",
    "siempre", "nunca", "jamás", "luego", "así", "apenas", "quizá", "quizás",
    # Formas auxiliares y copulativas de alta frecuencia (ser, estar, haber, tener)
    "es", "son", "era", "eran", "fue", "fueron", "sido", "ser", "sea", "sean", "siendo",
    "está", "estan", "están", "estaba", "estaban", "estuvo", "estuvieron", "estado", "estar", "esté", "estén",
    "hay", "había", "habian", "habían", "hubo", "he", "ha", "han", "hemos", "habiendo", "haber", "haya", "hayan",
    "tiene", "tienen", "tenía", "tenian", "tenían", "tuvo", "tuvieron", "tener", "tenga", "tengan"
})

# Patterns for control tokens, code fragments, hashes, and file paths
_CTRL_RE = re.compile(
    r'<\|[a-zA-Z0-9_|.-]+>|<im_[a-zA-Z0-9_]+>|</im_[a-zA-Z0-9_]+>|<query>|</query>|```+'
    r'|\b0x[0-9a-fA-F]+\b|\b[0-9a-fA-F]{8,}\b|SHA256=|(?:/[a-zA-Z0-9_.-]+){2,}'
    r'|[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+|\$\{[^}]+\}|::|->|=>|===|\+\+'
)

# Garbage / logging start patterns
_GARBAGE_START_RE = re.compile(
    r'^\s*(?:>|\[WARN\]|\[ERROR\]|\[INFO\]|\[DEBUG\]|\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}'
    r'|<\|im_start|>|<query>|Assistant:|PATCH|User:)'
)

_SPANISH_WORD_RE = re.compile(r'^[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+$')
_NATURAL_PUNCT_RE = re.compile(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ»\)"\'][.,;:!?]+(?:\s|$)|^¿|^¡')
_FORBIDDEN_WORD_CHARS_RE = re.compile(r'[0-9+=\\/@#$%^&*~<>_{}|]')


def score_text(text: str, prompt: str | None = None) -> dict:
    """Calculates coherence score and individual components for a generated text.

    Args:
        text: Response string to evaluate.
        prompt: Optional prompt text (does not alter scoring, maintained for API purity).

    Returns:
        dict: {"coherence": float (0..1), "components": {
            "func_rate": float (0..1),
            "rep_mass": float (0..1),
            "plaus": float (0..1),
            "struct": float (0..1)
        }}
    """
    if not text or not text.strip():
        return {
            "coherence": 0.0,
            "components": {
                "func_rate": 0.0,
                "rep_mass": 0.0,
                "plaus": 0.0,
                "struct": 0.0,
            },
        }

    words = [w.lower() for w in re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+', text)]
    n_words = len(words)

    # 1. func_rate: fraction of word tokens that are Spanish function words
    func_count = sum(1 for w in words if w in SPANISH_FUNCTION_WORDS)
    func_rate = (func_count / n_words) if n_words > 0 else 0.0

    # 2. rep_mass: max between word 4-grams and character 16-grams appearing >= 2 times
    raw_tokens = text.split()
    w_rep = 0.0
    if len(raw_tokens) >= 4:
        low_toks = [t.lower() for t in raw_tokens]
        w_ngrams = [tuple(low_toks[i:i + 4]) for i in range(len(low_toks) - 3)]
        w_counts = Counter(w_ngrams)
        w_rep = sum(c for g, c in w_counts.items() if c >= 2) / len(w_ngrams)

    c_rep = 0.0
    t_clean = text.lower()
    if len(t_clean) >= 16:
        c_ngrams = [t_clean[i:i + 16] for i in range(len(t_clean) - 15)]
        c_counts = Counter(c_ngrams)
        c_rep = sum(c for g, c in c_counts.items() if c >= 2) / len(c_ngrams)

    rep_mass = max(w_rep, c_rep)

    # 3. plaus: fraction of tokens that are plausible Spanish words minus control token penalties
    if not raw_tokens:
        plaus = 0.0
    else:
        valid_words = 0
        for tok in raw_tokens:
            core = tok.strip('.,!?:;«»"\'()¿¡-—[]{}')
            if (1 <= len(core) <= 14 and
                    _SPANISH_WORD_RE.fullmatch(core) and
                    not _FORBIDDEN_WORD_CHARS_RE.search(tok)):
                valid_words += 1
        plaus_ratio = valid_words / len(raw_tokens)

        ctrl_matches = len(_CTRL_RE.findall(text))
        ctrl_penalty = min(1.0, (ctrl_matches * 3.0) / max(1, len(raw_tokens)) + 0.1 * min(ctrl_matches, 5))
        plaus = max(0.0, min(1.0, plaus_ratio - ctrl_penalty))

    # 4. struct: structural sanity (punctuation, clause segmentation, length, no garbage start)
    if n_words == 0:
        struct = 0.0
    else:
        len_score = min(1.0, n_words / 8.0)

        natural_puncts = _NATURAL_PUNCT_RE.findall(text)
        n_puncts = len(natural_puncts)
        if n_puncts == 0:
            punct_score = 0.2 if n_words <= 8 else 0.05
        else:
            w_per_p = n_words / n_puncts
            if 4.0 <= w_per_p <= 16.0:
                punct_score = 1.0
            elif w_per_p < 4.0:
                punct_score = max(0.2, w_per_p / 4.0)
            else:
                punct_score = max(0.1, 16.0 / w_per_p)

        clauses = re.split(r'[,.;:!?\n]+', text)
        max_clause = max((len(re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]+', c)) for c in clauses), default=0)
        clause_score = 1.0 if max_clause <= 20 else max(0.1, 20.0 / max_clause)

        max_f_streak = 0
        cur_f_streak = 0
        for w in words:
            if w in SPANISH_FUNCTION_WORDS:
                cur_f_streak += 1
                if cur_f_streak > max_f_streak:
                    max_f_streak = cur_f_streak
            else:
                cur_f_streak = 0
        streak_score = 1.0 if max_f_streak <= 4 else max(0.1, 4.0 / max_f_streak)

        garbage_factor = 0.4 if _GARBAGE_START_RE.search(text) else 1.0
        struct = min(1.0, max(0.0, len_score * punct_score * clause_score * streak_score * garbage_factor))

    # Single final coherence number combined with fixed weights
    raw_coherence = W_PLAUS * plaus + W_FUNC * func_rate + W_STRUCT * struct - W_REP * rep_mass
    coherence = max(0.0, min(1.0, raw_coherence))

    return {
        "coherence": round(coherence, 4),
        "components": {
            "func_rate": round(func_rate, 4),
            "rep_mass": round(rep_mass, 4),
            "plaus": round(plaus, 4),
            "struct": round(struct, 4),
        },
    }


def _collect_jsonl_files(rows_arg: str) -> list[str]:
    """Resolves directory, glob, or single file into a list of jsonl file paths."""
    if os.path.isdir(rows_arg):
        pattern = os.path.join(rows_arg, "minp_rows_*.jsonl")
        files = sorted(glob.glob(pattern))
        if not files:
            files = sorted(glob.glob(os.path.join(rows_arg, "*.jsonl")))
        return files
    files = sorted(glob.glob(rows_arg))
    if not files and os.path.isfile(rows_arg):
        files = [rows_arg]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Coherence Judge CLI")
    parser.add_argument("--rows", required=True, help="Directory, glob or path to JSONL rows")
    parser.add_argument("--out", default=None, help="Optional output JSONL file with scores")
    args = parser.parse_args()

    files = _collect_jsonl_files(args.rows)
    if not files:
        sys.stderr.write(f"No jsonl files found matching: {args.rows}\n")
        sys.exit(1)

    by_config: dict[str, list[float]] = {}
    scored_rows = []

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                cfg = str(row.get("config", "UNKNOWN"))
                text = row.get("text", "")
                prompt = row.get("prompt", None)

                score_info = score_text(text, prompt=prompt)
                coh = score_info["coherence"]
                by_config.setdefault(cfg, []).append(coh)

                if args.out:
                    scored_rows.append({
                        "config": cfg,
                        "seed": row.get("seed"),
                        "id": row.get("id"),
                        "coherence": coh,
                        "components": score_info["components"]
                    })

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for item in scored_rows:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"{'Config':10} | {'N':5} | {'Median':8} | {'Mean':8}")
    print("-" * 38)
    for cfg in sorted(by_config.keys()):
        scores = sorted(by_config[cfg])
        n = len(scores)
        med = scores[n // 2] if n > 0 else 0.0
        mean = (sum(scores) / n) if n > 0 else 0.0
        print(f"{cfg:10} | {n:5d} | {med:8.4f} | {mean:8.4f}")


if __name__ == "__main__":
    main()
