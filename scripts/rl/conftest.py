"""Configuración de pytest para scripts/rl."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
for _p in [_ROOT, _ROOT / "scripts" / "pipelines", _ROOT / "scripts"]:
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)
