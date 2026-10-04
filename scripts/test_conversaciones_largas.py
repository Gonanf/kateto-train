"""Tests del final por criterio de las largas (sin red, sin teacher real).

El teacher es un stub de turnos canned: si algo intenta hablar con freellmapi o
la GPU, este archivo no lo hace. Se prueba el corte (motivo + largo), no el
validador grande: el texto stub es corto a proposito.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_conversaciones_largas import (Trozos, VOICE, build_parser,  # noqa: E402
                                       decidir_corte, generate_larga)

# Pool > 5 * (intercambios que usan los tests) + 2: si la rotacion da la vuelta,
# un turno sale identico a uno viejo y higiene_dup lo rechaza (fallo_en:voz:N).
VOCAB = ("carrito teclado guitarra oficina entrega cliente oferta torta lluvia campera "
         "sonido ensayo vecino comercio ganancia turno tramite alquiler reparacion "
         "mudanza factura almacen transporte descuento empleado proveedor logistica "
         "seguro vivienda cuadrante").split()
NV = len(VOCAB)


def _voz(i: int) -> str:
    a, b, c = (VOCAB[(5 * i) % NV], VOCAB[(5 * i + 1) % NV], VOCAB[(5 * i + 2) % NV])
    return (f"posta que {a} y {b} re detonaron, che, yo lo laburo con {c} "
            f"y listo, no me rompe el ritmo")


def maestro_stub(fin_en: int | None = None):
    """(prompt) -> (texto, uso). `<FIN>` en el turno de chat `fin_en` (0-based)."""
    estado = {"n": 0}

    def maestro(prompt: str) -> tuple[str, dict]:
        i = estado["n"]
        if prompt.startswith("Estas escribiendo UNA linea de chat"):
            estado["n"] = i + 1
            return f"che, que onda el tema {i}?" + ("<FIN>" if i == fin_en else ""), {}
        return _voz(i), {}
    return maestro


def juez_stub(respuestas: list[str]):
    """Devuelve las respuestas en orden; se acaba en la ultima (error = sigue)."""
    estado = {"n": 0}

    def juez(prompt: str) -> str:
        r = respuestas[min(estado["n"], len(respuestas) - 1)]
        estado["n"] += 1
        return r
    return juez, estado


def material(n_lineas: int) -> str:
    """Material de n lineas IGUALES: partir_lineas corta despues de cada una."""
    return "\n".join(f"linea {i} del material con palabras de prueba" for i in range(n_lineas))


def n_voz(txt: str) -> int:
    """Turnos de la voz: cada uno abre con su marcador `<|im_start|>seco`."""
    return txt.count(f"<|im_start|>{VOICE}\n")


def test_agotado_corta_al_consumirse_el_material():
    """5 trozos de material, minimo 2: el material se agota y se corta en 5."""
    txt, st = generate_larga(Trozos(material(5), 12), maestro_stub(), min_ex=2, max_ex=60)
    assert (st["intercambios"], st["corte"], n_voz(txt)) == (5, "agotado", 5)


def test_cierre_con_marcador_fin():
    """<FIN> en el chat 4: un turno de voz mas de cierre y termina en 4."""
    txt, st = generate_larga(Trozos(material(30), 30), maestro_stub(fin_en=3),
                             min_ex=0, max_ex=60)
    assert (st["intercambios"], st["corte"], n_voz(txt)) == (4, "cierre", 4)
    assert "<FIN>" not in txt                      # el marcador nunca se guarda
    assert "<|im_user|>" in txt and not txt.rstrip().endswith("<|im_user|>")


def test_juez_si_corta_en_el_intercambio_que_pregunta():
    """El juez dice SI la tercera vez: corta en ese intercambio."""
    juez, nj = juez_stub(["NO", "NO", "SI"])
    txt, st = generate_larga(Trozos(material(30), 30), maestro_stub(), min_ex=0,
                             max_ex=60, juez=juez, juez_cada=1)
    assert (st["intercambios"], st["corte"], n_voz(txt)) == (3, "juez", 3)
    assert nj["n"] == 3


def test_juez_no_llega_a_max_ex():
    """Juez siempre NO: nadie corta por juez, gana el tope."""
    juez, nj = juez_stub(["NO"])
    txt, st = generate_larga(Trozos(material(30), 30), maestro_stub(), min_ex=0,
                             max_ex=4, juez=juez, juez_cada=2)
    assert (st["intercambios"], st["corte"], n_voz(txt)) == (4, "max_ex", 4)
    assert nj["n"] == 2                           # le pregunte en 2 y en 4


def test_min_ex_no_corta_antes_del_minimo():
    """Material para 2 trozos con min_ex 6: sigue hasta 6, no corta en 2."""
    txt, st = generate_larga(Trozos(material(2), 60), maestro_stub(), min_ex=6, max_ex=60)
    assert (st["intercambios"], st["corte"], n_voz(txt)) == (6, "agotado", 6)


def test_decidir_corte_pura():
    """4 decisiones directas + el piso de min_ex."""
    hist = [("user", "che"), (VOICE, "posta que el carrito re detono")] * 3   # 3 intercambios
    assert decidir_corte(hist, None, None, min_ex=0, max_ex=60) == "agotado"
    assert decidir_corte(hist, "trozo", None, hay_fin=True, min_ex=0, max_ex=60) == "cierre"
    assert decidir_corte(hist, "trozo", "SI", min_ex=0, max_ex=60) == "juez"
    assert decidir_corte(hist, "trozo", "NO", min_ex=0, max_ex=60) is None
    # min_ex pisa a todos: material agotado y sin cortes antes del minimo
    assert decidir_corte(hist, None, "SI", hay_fin=True, min_ex=12, max_ex=60) is None


def test_n_ex_es_alias_de_max_ex():
    """Invocaciones viejas con --n-ex no se rompen."""
    a = build_parser().parse_args(["--n-ex", "22"])
    assert a.n_ex == 22 and a.max_ex == 60            # el alias manda
    assert build_parser().parse_args(["--max-ex", "30"]).n_ex is None


def test_formato_del_texto_armado():
    """El texto armado tiene que traer los marcadores del dataset.

    Sin esto la muestra no entra al SFT: `check_data` (y `validate_larga`) exigen
    arranque en `<|im_user|>` y cierre en `<|im_end|>`, y los turnos de voz se
    cuentan por `<|im_start|>seco`. Un cambio en la funcion de armado puede tirar
    los marcadores y dejar todos los otros tests en verde (paso de verdad).
    """
    txt, st = generate_larga(Trozos(material(6), 12), maestro_stub(), min_ex=2, max_ex=12)
    assert txt.startswith("<|im_user|>")
    assert txt.rstrip().endswith("<|im_end|>")
    assert txt.count(f"<|im_start|>{VOICE}\n") == st["intercambios"] == 6
    assert txt.count("<|im_user|>") == 6
    assert txt.count("<|im_end|>") == 2 * st["intercambios"]
    # bloques exactos, alternados y sin sobras
    bloques = re.findall(r"<\|im_user\|>.*?<\|im_end\|>|<\|im_start\|>" + VOICE + r"\n.*?<\|im_end\|>",
                         txt, re.S)
    assert len(bloques) == 2 * st["intercambios"]
    assert "\n".join(bloques) == txt.rstrip()