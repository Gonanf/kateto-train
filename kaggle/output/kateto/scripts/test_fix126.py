"""fix-126: test end-to-end del CLI (fallback a CPU roto + regresion de import anidado).

Bug 1 era un `UnboundLocalError` en el fallback a CPU de `infer_kateto.py`: un
`import os` ANIDADO dentro del `if self.device == "cpu"` convertia a `os` en
variable local de toda la funcion, asi que cuando la rama cuda NO se ejecutaba y
despues se entraba al `except` de la carga, `os.cpu_count()` explotaba.

Estas pruebas corren el CLI REAL (`rwkv_pipeline/infer_kateto.py`) por subprocess
con el python que tiene torch (`venv-unsloth-qwen/bin/python`, igual que start.sh):
  1. `--voice none`    -> exit 0, salida no vacia, sin Traceback/UnboundLocalError.
  2. `--voice seco`    -> exit 0 y sin traceback; si el guard de metadata de
                          fix-125 rechaza el estado, el test lo dice explicito.
  3. `--device cuda` con CUDA_VISIBLE_DEVICES="" (cuda inutilizable) -> cae a CPU
                          SIN excepcion: es exactamente el camino que estaba roto.

El test NO importa torch (pytest corre con venv/, que no tiene torch): toda la
inferencia va en un subproceso aparte.
"""
import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
CLI = REPO / "rwkv_pipeline" / "infer_kateto.py"
PY = REPO / "venv-unsloth-qwen" / "bin" / "python"
MODEL = REPO / "out" / "rwkv_kateto_base_plus_0.4b" / "rwkv-1.pth"
PROMPT = "dale, contame algo"

if not PY.exists():
    pytest.skip(f"no existe el python con torch: {PY}", allow_module_level=True)
if not MODEL.exists():
    pytest.skip(f"no existe el checkpoint 0.4b: {MODEL}", allow_module_level=True)


def _env(**extra):
    env = os.environ.copy()
    env.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("TORCH_COMPILE_DISABLE", "1")
    env.update(extra)
    return env


def _run_cli(args, env_extra=None, timeout=600):
    """Corre el CLI real y devuelve el CompletedProcess."""
    cmd = [str(PY), str(CLI)] + args
    return subprocess.run(
        cmd, cwd=str(REPO), env=_env(**(env_extra or {})),
        capture_output=True, text=True, timeout=timeout,
    )


def _has_traceback(out, err):
    blob = out + "\n" + err
    return "Traceback" in blob or "UnboundLocalError" in blob


def test_case1_voice_none():
    """Caso 1: `--voice none` -> exit 0, salida no vacia, sin traceback."""
    r = _run_cli(["--model", str(MODEL), "--voice", "none", "--prompt", PROMPT])
    assert r.returncode == 0, f"rc={r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert r.stdout.strip(), "salida vacia (sin generacion)"
    assert not _has_traceback(r.stdout, r.stderr), f"traceback detecado:\n{r.stderr}"


def test_case2_voice_seco():
    """Caso 2: `--voice seco` -> exit 0 y sin traceback.

    Si el guard de metadata de fix-125 rechaza el estado (base equivocada), el
    test lo dice EXPLICITAMENTE en vez de romper: ese rechazo no es un fallo de
    fix-126, es una condicion de datos.
    """
    r = _run_cli(["--model", str(MODEL), "--voice", "seco", "--prompt", PROMPT])
    reject = "[Kateto] estado 'seco' entrenado para base" in r.stderr and "reentreñá" in r.stderr
    if reject:
        pytest.skip(
            "fix-125 reventó el estado 'seco' por metadata de base (esta corriendo "
            f"la base equivocada). No es fallo de fix-126. stderr:\n{r.stderr}"
        )
    assert r.returncode == 0, f"rc={r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert r.stdout.strip(), "salida vacia (sin generacion)"
    assert not _has_traceback(r.stdout, r.stderr), f"traceback detecado:\n{r.stderr}"


def test_case3_fallback_cuda_a_cpu():
    """Caso 3: cuda inutilizable -> debe caer a CPU SIN excepcion.

    Es exactamente el camino que estaba roto (UnboundLocalError por el `import os`
    anidado). Forzamos cuda inutilizable con CUDA_VISIBLE_DEVICES="" para que la
    carga en cuda falle, entre al except y caiga a cpu sin reventar.
    """
    r = _run_cli(
        ["--model", str(MODEL), "--device", "cuda", "--voice", "none", "--prompt", PROMPT],
        env_extra={"CUDA_VISIBLE_DEVICES": ""},
    )
    assert r.returncode == 0, f"rc={r.returncode}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    # debe haber caido a cpu (y no haber reventado en el set_num_threads)
    assert "device=cpu" in r.stdout, f"no cayo a cpu:\n{r.stdout}\n{r.stderr}"
    assert r.stdout.strip(), "salida vacia (sin generacion)"
    assert not _has_traceback(r.stdout, r.stderr), f"traceback detecado:\n{r.stderr}"
    assert "device=cpu (pedido: cuda)" in r.stdout