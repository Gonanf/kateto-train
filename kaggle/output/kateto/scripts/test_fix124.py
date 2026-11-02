"""fix124: el validador nuevo no puede explotar con tool_call de args no-objeto.

fix123 cambio la extraccion de bloques por balance (ahora tambien levanta tool_call
cuyos argumentos NO son un objeto JSON). La validacion de argumentos asumia dict y
reventaba con `AttributeError: 'list' object has no attribute 'items'`, matando al
worker (validate() se llama fuera del try de la generacion).

Replay real de la ola cerrada: 9 bloques en 3 escenarios con args array/string:
  no_se_programar  run_command: ["systemctl", "is-active", "docker"]
  no_sabe_algo     run_command: "resultado River vs Boca anoche"
  voice_ask_help   refine_memory: ["finances", "subtraction", "evidence: upcoming bills", "fact: zero surplus"]

Estos tests son locales (sin red): ejercitan validate() sobre muestras que antes
reventaban y sobre las ramas dict / no-json que tienen que seguir como hoy.
"""
from gen_toolcalling_dataset import SAMPLE_OK, validate


# --- las tres muestras crudas del replay (args que NO son dict) -------------------

MUESTRA_ARRAY = """<|im_user|>che, fijate si docker esta corriendo<|im_end|>
<|im_start|>seco
Dale, lo chequeo.
tool_call run_command: ["systemctl", "is-active", "docker"]
tool_result run_command: active
Esta activo, che.
<|im_end|>"""

MUESTRA_STRING = """<|im_user|>che, me contas quien gano anoche<|im_end|>
<|im_start|>seco
Dale, lo busco.
tool_call run_command: "resultado River vs Boca anoche"
tool_result run_command: River 2 a 1
Fue 2 a 1 para River, che.
<|im_end|>"""

MUESTRA_REFINE = """<|im_user|>che, guardame unos datos de la memoria<|im_end|>
<|im_start|>seco
Dale, los guardo.
tool_call refine_memory: ["finances", "subtraction", "evidence: upcoming bills", "fact: zero surplus"]
tool_result refine_memory: ok
Listo, che.
<|im_end|>"""


def _assert_solo_args_no_dict(errs, tool):
    """El tool tiene que caer en args_no_dict, y NINGUNO de los chequeos que
    asumen claves (faltan_requeridos / args_extra / args_tipo) ni args_no_json
    puede dispararse sobre ese tool."""
    assert f"args_no_dict:{tool}" in errs, errs
    for e in errs:
        assert not e.startswith(f"faltan_requeridos:{tool}"), e
        assert not e.startswith(f"args_extra:{tool}"), e
        assert not e.startswith(f"args_tipo:{tool}"), e
        assert not e.startswith(f"args_no_json:{tool}"), e


def test_array_args_no_dict():
    errs = validate(MUESTRA_ARRAY, voice="seco")
    _assert_solo_args_no_dict(errs, "run_command")


def test_string_args_no_dict():
    errs = validate(MUESTRA_STRING, voice="seco")
    _assert_solo_args_no_dict(errs, "run_command")


def test_refine_memory_array_args_no_dict():
    errs = validate(MUESTRA_REFINE, voice="seco")
    _assert_solo_args_no_dict(errs, "refine_memory")


def test_args_dict_sigue_dando_checks_como_hoy():
    # write_file requiere path y content (string). path con tipo malo, key extra y
    # content ausente -> tienen que salir los TRES errores de dict como siempre.
    muestra = """<|im_user|>che, escribime un archivo<|im_end|>
<|im_start|>seco
Dale, lo escribo.
tool_call write_file: {"path": 42, "extra_key": 1}
tool_result write_file: ok
Listo, che.
<|im_end|>"""
    errs = validate(muestra, voice="seco")
    assert any(e.startswith("faltan_requeridos:write_file") for e in errs), errs
    assert any(e.startswith("args_extra:write_file") for e in errs), errs
    assert any(e.startswith("args_tipo:write_file") for e in errs), errs
    assert "args_no_dict:write_file" not in errs, errs


def test_json_invalido_sigue_dando_args_no_json():
    # JSON claramente invalido -> args_no_json como siempre, NO args_no_dict.
    muestra = """<|im_user|>che, correme un comando<|im_end|>
<|im_start|>seco
Dale.
tool_call run_command: {"command": }
tool_result run_command: ok
Listo, che.
<|im_end|>"""
    errs = validate(muestra, voice="seco")
    assert "args_no_json:run_command" in errs, errs
    assert "args_no_dict:run_command" not in errs, errs


def test_muestra_sana_sigue_dando_vacio():
    # La muestra buena de dry-run tiene que seguir PASANDO (sin errores).
    assert validate(SAMPLE_OK, voice="seco") == []