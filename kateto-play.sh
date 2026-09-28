#!/usr/bin/env bash
# Hablar con Kateto 2.9B desde la terminal.
#
#   ./kateto-play.sh                                  -> chat interactivo (voz seco)
#   ./kateto-play.sh --once "Che, que onda el Kun?"   -> una sola respuesta
#   ./kateto-play.sh --voice doktor                   -> con otro marcador de voz
#
# Levanta el server solo (Q8_0, la config validada) y lo baja al salir.
# Usa el python3 del sistema: no necesita venv.
set -euo pipefail
cd "$(dirname "$0")"
exec python3 -u scripts/kateto_chat.py --serve "$@"
