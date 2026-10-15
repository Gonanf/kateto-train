"""Filtros de admisión para saneamiento del dataset de SFT de Kateto.

Tres filtros de admisión:
1. Idioma: detectar inglés en question/answer (ratio de stopwords inglesas —
   the, want, user, okay, first, let, me, create, coherent, answer, question, likely, typo, safe, unsafe —,
   excluyendo el interior de <tool_call>...</tool_call> y bloques de código).
   Regla: >=2 stopwords inglesas (o detector de idioma que diga lang != es con confianza > 0.70) -> fuera.
   (Nota: 'me' aislado o repetido sin ninguna otra stopword inglesa no califica como inglés para no penalizar
    el pronombre español 'me').
2. Artefactos de pipeline: regex de lista negra case-insensitive sobre cualquiera de los dos campos:
   user safety, safe/unsafe como etiqueta suelta, okay,, let me break, the user wants, given answer,
   coherent with, aquí tienes, como modelo de, here is the, [Soloto], y respuestas con menos de 15
   caracteres alfanuméricos reales (salvo tool-call o tokens especiales <NO_RESPONSE> / <WAIT>).
3. Asimetría de longitud (anti-leak de razonamiento): si question tiene más de 2 párrafos o más de 350
   caracteres (y no hay un bloque explícito de código/contexto) -> fuera.
"""
from __future__ import annotations

import re
from typing import Tuple, Optional, List

STOPWORDS = {
    "the", "want", "user", "okay", "first", "let", "me",
    "create", "coherent", "answer", "question", "likely",
    "typo", "safe", "unsafe"
}

BLACKLIST_REGEXES = [
    ("user_safety", re.compile(r"user safety", re.I)),
    ("safe_label", re.compile(r"\b(?:safe|unsafe)\b", re.I)),
    ("okay_comma", re.compile(r"okay,", re.I)),
    ("let_me_break", re.compile(r"let me break", re.I)),
    ("the_user_wants", re.compile(r"the user wants", re.I)),
    ("given_answer", re.compile(r"given answer", re.I)),
    ("coherent_with", re.compile(r"coherent with", re.I)),
    ("aqui_tienes", re.compile(r"aquí tienes", re.I)),
    ("como_modelo_de", re.compile(r"como modelo de", re.I)),
    ("here_is_the", re.compile(r"here is the", re.I)),
    ("soloto", re.compile(r"\[Soloto\]", re.I)),
]

def clean_text(text: str) -> str:
    """Excluye el interior de <tool_call>...</tool_call> y bloques de código."""
    if not text:
        return ""
    t = re.sub(r"<tool_call>[\s\S]*?</tool_call>", " ", text)
    t = re.sub(r"```[\s\S]*?```", " ", t)
    t = re.sub(r"`[^`\n]*?`", " ", t)
    return t

def find_english_stopwords(text: str) -> List[str]:
    c = clean_text(text)
    words = [w for w in re.findall(r"\b[a-zA-Z]+\b", c.lower()) if w in STOPWORDS]
    # Si la única palabra coincidente es 'me', en español es el pronombre 'me' (ej: 'me parece que no me gusta').
    # Sólo cuenta como inglés si se combina con otras stopwords inglesas (ej: 'let me', 'tell me the answer').
    if words and set(words) == {"me"}:
        return []
    return words

def check_admission(q: str, a: str) -> Tuple[bool, str, Optional[str]]:
    """Evalúa una fila con los tres filtros de admisión.
    
    Retorna:
        (pasa, motivo_filtro, detalle)
        Si pasa: (True, "ok", None)
        Si falla: (False, "idioma" | "artefacto" | "asimetria", detalle)
    """
    q_str = str(q or "").strip()
    a_str = str(a or "").strip()

    # 1. Filtro de Idioma
    wq = find_english_stopwords(q_str)
    if len(wq) >= 2:
        return False, "idioma", f"stopwords_in_q:{wq}"
    wa = find_english_stopwords(a_str)
    if len(wa) >= 2:
        return False, "idioma", f"stopwords_in_a:{wa}"

    # 2. Filtro de Artefactos de pipeline
    for name, rx in BLACKLIST_REGEXES:
        if rx.search(q_str):
            return False, "artefacto", f"regex_in_q:{name}"
        if rx.search(a_str):
            return False, "artefacto", f"regex_in_a:{name}"

    # Respuestas con menos de 15 caracteres alfanuméricos reales (salvo tool-call o turn tokens)
    if "<tool_call>" not in a_str and "<NO_RESPONSE>" not in a_str and "<WAIT>" not in a_str:
        alnum = re.findall(r"[a-zA-Z0-9áéíóúÁÉÍÓÚñÑ]", a_str)
        if len(alnum) < 15:
            return False, "artefacto", f"short_response:{len(alnum)}<15"

    # 3. Asimetría de longitud (anti-leak de razonamiento)
    # Si question tiene más de 2 párrafos o más de 350 caracteres (y no hay bloque de código/contexto)
    has_code_or_ctx = (
        "```" in q_str
        or "`" in q_str
        or "<tool_call>" in q_str
        or "contexto:" in q_str.lower()
        or "[context" in q_str.lower()
    )
    if not has_code_or_ctx:
        paragraphs = [p for p in q_str.split("\n\n") if p.strip()]
        if len(paragraphs) > 2:
            return False, "asimetria", f"paragraphs:{len(paragraphs)}>2"
        if len(q_str) > 350:
            return False, "asimetria", f"len:{len(q_str)}>350"

    return True, "ok", None
