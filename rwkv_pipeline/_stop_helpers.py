"""Helpers puros (SIN dependencia de torch) para los cortes de generacion.

Viven en un modulo aparte para que `scripts/test_fix125.py` pueda testearlos
con `python3` pelado: el gate de pytest corre sin torch, y `infer_kateto.py`
hace `import torch` en el top-level.
"""


def ends_with(tokens, suffix):
    """True si la cola de `tokens` es exactamente `suffix` (secuencia de ids).

    Es el corte por ids del modelo: en lugar de buscar el string `<|im_end|>` /
    `<|im_user|>` en el decode, se compara la secuencia de ids generada contra
    la del encode del marcador.
    """
    if not suffix or len(tokens) < len(suffix):
        return False
    return tokens[-len(suffix):] == suffix


def count_consecutive_ngram_repeats(tokens, n):
    """Cuantas veces consecutivas se repite el ultimo n-grama inmediatamente antes.

    Sirve para la guarda de repeticion: si el ultimo n-grama ya aparecio
    `repeats` veces seguidas en la cola, la generacion degenero en un loop.

    > count_consecutive_ngram_repeats([1,2,3,1,2,3,1,2,3], 3) == 2
    > count_consecutive_ngram_repeats([1,2,3,4,5,6], 3) == 0
    """
    if n <= 0 or len(tokens) < 2 * n:
        return 0
    last = tuple(tokens[-n:])
    repeats = 0
    idx = len(tokens) - 2 * n
    while idx >= 0 and tuple(tokens[idx:idx + n]) == last:
        repeats += 1
        idx -= n
    return repeats