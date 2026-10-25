"""Tests de fix123: extraccion de bloques por balance + tuteo solo en la voz.

Sin red, sin teacher: todo corre contra validate() y extraer_bloques() de
scripts/gen_toolcalling_dataset.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import gen_toolcalling_dataset as gen  # noqa: E402


def _errs_contienen(errs: list[str], prefijo: str) -> bool:
    return any(e.startswith(prefijo) for e in errs)


# Texto REAL de una muestra rechazada en kateto-ola99 (crudo, sin tocar):
# el `}` exterior queda en la linea siguiente, partido por el wrap.
TEXTO_REAL_SEND_EVENT = (
    '<|im_user|>che, tirale un mensaje al overlay y avisame\n'
    '<|im_start|>seco\n'
    'tool_call send_event: {"event_name": "chat_message", "data": {"message": "sistema chequeado: chat y\n'  # noqa: E501
    'follows vivos", "user": "seco"}\n'
    ', "target": "overlay_plugin"}\n'
    'Listo, el overlay ya tiene el aviso. Anda al chat que hay cola.\n'
    'tool_result send_event: ok\n'
    '<|im_end|>'
)


class TestBalanceBloques:
    def test_send_event_anidado_wrap_parsea(self):
        """El caso medido: JSON valido con `data` anidado y el cierre partido.
        Hoy el regex non-greedy corta en el `}` interno y da args_no_json."""
        bloques = gen.extraer_bloques(TEXTO_REAL_SEND_EVENT)
        calls = [(n, b) for (t, n, b) in bloques if t == "call"]
        assert len(calls) == 1
        nombre, bloque = calls[0]
        assert nombre == "send_event"
        # el bloque extraido por balance tiene que parsear TAL CUAL (colapsando
        # los saltos internos, como hace el validador)
        import json
        plano = " ".join(bloque.split())
        args = json.loads(plano)
        assert args["target"] == "overlay_plugin"
        assert args["data"]["user"] == "seco"

    def test_validate_no_da_args_no_json(self):
        errs = gen.validate(TEXTO_REAL_SEND_EVENT, voice="seco",
                            min_turns=1, min_palabras_usuario=1)
        assert not _errs_contienen(errs, "args_no_json"), errs

    def test_tool_result_array_multilinea_parsea(self):
        """RX_TOOLRES viejo cortaba en el primer \\n: los arrays en dos lineas
        quedaban truncados. Por balance llegan completos."""
        sample = (
            '<|im_user|>que videos hay indexados\n'
            '<|im_start|>seco\n'
            'tool_call list_videos: {}\n'
            'tool_result list_videos: [{"id": 1, "titulo": "run de ayer",\n'
            '"duracion": 3600},\n'
            '{"id": 2, "titulo": "run del jueves", "duracion": 7200}]\n'
            'Dos runs largos en el indice. La del jueves dura el doble.\n'
            '<|im_end|>'
        )
        errs = gen.validate(sample, voice="seco",
                            min_turns=1, min_palabras_usuario=1)
        assert not _errs_contienen(errs, "args_no_json"), errs

    def test_comillas_sin_escapar_sigue_dando_args_no_json(self):
        """Un bloque que de verdad no parsea sigue siendo args_no_json."""
        sample = (
            '<|im_user|>mandale un evento al chat\n'
            '<|im_start|>seco\n'
            'tool_call send_event: {"data": {"msg": "dijo "hola" ayer"}}\n'
            '<|im_end|>'
        )
        errs = gen.validate(sample, voice="seco",
                            min_turns=1, min_palabras_usuario=1)
        assert _errs_contienen(errs, "args_no_json"), errs

    def test_bloque_sin_cerrar_llega_a_fin_de_linea(self):
        """Si nunca balancea, el bloque llega al fin de la linea, como hoy."""
        txt = 'tool_call read_file: {"path": "notas.txt"\ny mas prosa\n'
        bloques = gen.extraer_bloques(txt)
        assert len(bloques) == 1
        assert bloques[0][2] == '{"path": "notas.txt"'


def _sample_con_voz(voz_txt: str, usuario_txt: str = "che, contame algo") -> str:
    return (
        f'<|im_user|>{usuario_txt}\n'
        f'<|im_start|>seco\n{voz_txt}\n<|im_end|>'
    )


class TestTuteo:
    def _tuteo(self, errs):
        return [e for e in errs if e.startswith("tuteo")]

    def test_tuteo_solo_en_usuario_no_es_error(self):
        """147/334 falsos positivos: el tuteo esta en el turno del usuario."""
        sample = (
            '<|im_user|>dime algo, tienes que ayudarme amigo mio\n'
            '<|im_start|>seco\n'
            'Mirá el stream, boludo, que esta bueno. Andá a avisarle al Doktor.\n'
            '<|im_end|>'
        )
        errs = gen.validate(sample, voice="seco",
                            min_turns=1, min_palabras_usuario=1)
        assert not self._tuteo(errs), errs

    def test_homografo_con_clitico_no_es_error(self):
        sample = _sample_con_voz("la heladera te mira con cara de boludo, che")
        errs = gen.validate(sample, voice="seco",
                            min_turns=1, min_palabras_usuario=1)
        assert not self._tuteo(errs), errs

    def test_homografo_con_sujeto_no_es_error(self):
        sample = _sample_con_voz("todos lo ven y nadie lo tapa, es así de simple")
        errs = gen.validate(sample, voice="seco",
                            min_turns=1, min_palabras_usuario=1)
        assert not self._tuteo(errs), errs

    def test_mira_imperativo_es_tuteo(self):
        sample = _sample_con_voz("mira, boludo, esto no lo salva nadie")
        errs = gen.validate(sample, voice="seco",
                            min_turns=1, min_palabras_usuario=1)
        assert self._tuteo(errs) == ["tuteo:mira"], errs

    def test_dime_es_tuteo(self):
        sample = _sample_con_voz("primero dime qué querés planificar")
        errs = gen.validate(sample, voice="seco",
                            min_turns=1, min_palabras_usuario=1)
        assert self._tuteo(errs) == ["tuteo:dime"], errs

    def test_tienes_es_tuteo(self):
        sample = _sample_con_voz("si tienes algo profesional en mente, mejor")
        errs = gen.validate(sample, voice="seco",
                            min_turns=1, min_palabras_usuario=1)
        assert self._tuteo(errs) == ["tuteo:tienes"], errs

    def test_quieres_es_tuteo(self):
        sample = _sample_con_voz("quieres que te lo arme o no, decime")
        errs = gen.validate(sample, voice="seco",
                            min_turns=1, min_palabras_usuario=1)
        assert self._tuteo(errs) == ["tuteo:quieres"], errs

    def test_guarda_solo_mira_lo_inmediato(self):
        """El guarda mira lo que esta INMEDIATAMENTE antes: aunque haya un
        clitico antes en la frase, "mira" imperativo sigue siendo tuteo."""
        sample = _sample_con_voz("te lo dejo aca, mira lo que hizo el flaco")
        errs = gen.validate(sample, voice="seco",
                            min_turns=1, min_palabras_usuario=1)
        assert self._tuteo(errs) == ["tuteo:mira"], errs


class TestPrompts:
    def test_los_prompts_llevan_las_reglas_de_voseo(self):
        for p in (
            gen.build_prompt({"id": "x", "cat": "B", "tools": 0, "txt": "charla"},
                             "", "seco", 2),
            gen.build_pair_prompt({"id": "x", "cat": "B", "tools": 0, "txt": "charla"},
                                  "seco", [], 2, ""),
            gen.build_turn_prompt({"id": "x", "cat": "B", "tools": 0, "txt": "charla"},
                                  "seco", [], 2, ""),
        ):
            assert "Imperativos SIEMPRE en voseo" in p
            assert ('PROHIBIDO el tuteo: tienes, puedes, quieres, eres, '
                    'dime, hazlo, ven, amigo mio') in p
