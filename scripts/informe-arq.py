#!/usr/bin/env python3
"""
scripts/informe-arq.py — Renderizador del informe INFORME.md
"""

import argparse
import json
import os
from pathlib import Path
import sys


def format_bytes(n):
    if n is None:
        return "-"
    if n >= 1024**3:
        return f"{n / (1024**3):.2f} GB"
    elif n >= 1024**2:
        return f"{n / (1024**2):.2f} MB"
    elif n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def format_params(p):
    if p is None:
        return "-"
    if isinstance(p, (int, float)):
        if p >= 1e9:
            return f"{p / 1e9:.2f}B"
        elif p >= 1e6:
            return f"{p / 1e6:.1f}M"
        return str(p)
    return str(p)


def render_report(tput_path, auditoria_path, out_path, gates_path=None):
    with open(auditoria_path, "r", encoding="utf-8") as f:
        auditoria = json.load(f)

    with open(tput_path, "r", encoding="utf-8") as f:
        tput = json.load(f)

    # 1. Seccion 1
    sec1 = """## 1. Qué se midió y qué no

### Qué se midió
- **Throughput en CPU**: Medición estricta sobre CPU (`-ngl 0`, `-t 4`) de la velocidad de procesamiento de prompt (`pp512`, 512 tokens) y generación autoregresiva (`tg128`, 128 tokens) empleando el binario oficial `llama-bench` (build `b32-f2cc728`).
- **Pares de modelos en Q8_0**: Comparación directa entre modelos de pesos cuantiados a Q8_0:
  - Escala ~1.4B / 1.5B: `rwkv15` (RWKV-7 1.5B) vs `mamba14` (Mamba-1 1.4B base) y `mamba14inst` (Mamba-1 1.4B instruct).
  - Escala ~2.8B / 2.9B: `rwkv29` (RWKV-7 2.9B baseline) vs `mamba28` (Mamba-1 2.8B base).
- **Control de carga del host**: Registro sistemático de `loadavg` (1m, 5m, 15m) antes y después de cada ejecución, junto con tiempo de pared (`wall_s`). En ejecuciones de múltiples pases, se invierte el orden de ejecución para mitigar el sesgo por fluctuaciones de carga del sistema.
- **Auditoría de motor**: Verificación directa de símbolos de arquitecturas soportadas en la biblioteca dinámica `libllama.so` (`mamba`, `mamba2`, `jamba`, `nemotron_h`, `falcon`, `rwkv7`, `zamba2`).
- **Relevamiento de candidatos en HuggingFace Hub**: Búsqueda e inspección de metadatos (repositorios GGUF, descargas, fechas, archivos y tamaños) para 11 familias de arquitecturas recurrentes/SSM sin descargar pesos.

### Qué no se midió
- **Inferencia en GPU**: Prohibida explícitamente en este benchmark debido a que la VRAM de la GPU del host está comprometida por servicios activos de otras corridas concurrentes.
- **Calidad y coherencia generativa**: No se corrieron benchmarks de evaluación conversacional (`coherencia-larga` o suite de 121 filas) por las razones metodológicas detalladas en la Sección 4.
- **Descarga de artefactos pesados**: No se descargaron modelos ni pesos adicionales desde la red; las llamadas a HuggingFace se limitaron exclusivamente a metadatos vía su API REST.
- **Arquitecturas sin soporte**: Modelos basados en `zamba2` u otras variantes ausentes en `libllama.so` no fueron evaluados en runtime.
"""

    # 2. Seccion 2
    resumen = tput.get("resumen", [])
    corridas = tput.get("corridas", [])

    rows_sec2 = []
    for item in resumen:
        m_id = item.get("modelo", "")
        if "error" in item or item.get("nota", "").startswith("no medido"):
            motivo = item.get("error") or item.get("nota") or "error"
            # Sanitize multiline error for single markdown row
            motivo_clean = motivo.replace("\n", " ")[:80]
            rows_sec2.append(f"| `{m_id}` | - | no medido: {motivo_clean} | no medido: {motivo_clean} | - |")
            continue

        n_params = item.get("n_params")
        pp_min = item.get("pp_avg_ts_min", 0.0)
        pp_max = item.get("pp_avg_ts_max", 0.0)
        tg_min = item.get("tg_avg_ts_min", 0.0)
        tg_max = item.get("tg_avg_ts_max", 0.0)

        pp_str = f"{pp_min:.2f}–{pp_max:.2f}" if abs(pp_max - pp_min) > 1e-4 else f"{pp_min:.2f}"
        tg_str = f"{tg_min:.2f}–{tg_max:.2f}" if abs(tg_max - tg_min) > 1e-4 else f"{tg_min:.2f}"

        # Loadavg antes/despues
        m_runs = [c for c in corridas if c.get("modelo") == m_id]
        if m_runs:
            antes_list = [c["loadavg_antes"][0] for c in m_runs if "loadavg_antes" in c and c["loadavg_antes"]]
            desp_list = [c["loadavg_despues"][0] for c in m_runs if "loadavg_despues" in c and c["loadavg_despues"]]
            if antes_list and desp_list:
                carga_str = f"{min(antes_list):.2f}–{max(antes_list):.2f} / {min(desp_list):.2f}–{max(desp_list):.2f}"
            else:
                carga_str = "-"
        else:
            carga_str = "-"

        rows_sec2.append(
            f"| `{m_id}` | {format_params(n_params)} | {pp_str} | {tg_str} | {carga_str} |"
        )

    tabla_sec2 = "| modelo | params | pp512 t/s (mín–máx) | tg128 t/s (mín–máx) | carga (loadavg antes/después) |\n|---|---|---|---|---|\n"
    tabla_sec2 += "\n".join(rows_sec2)

    sec2 = f"""## 2. Throughput a igual tamaño y misma cuantización (Q8_0, CPU, mismo binario)

{tabla_sec2}
"""

    # 3. Seccion 3
    # Soporte engine
    archs_sop = auditoria.get("engine", {}).get("archs_soportados", {})
    sop_str = ", ".join([f"`{k}`: {'Sí' if v else 'No'}" for k, v in archs_sop.items()])

    # Tabla (a) Modelos locales
    locales = auditoria.get("modelos_locales", [])
    rows_local = []
    for loc in locales:
        m_id = loc.get("id", "")
        arch = loc.get("arch", "-")
        params_val = loc.get("size_label") or loc.get("params") or "-"
        b_val = loc.get("bytes")
        rows_local.append(f"| `{m_id}` | `{arch}` | {params_val} | {format_bytes(b_val)} ({b_val:,} B) |")

    tabla_locales = "| modelo local | arch | params | bytes |\n|---|---|---|---|\n" + "\n".join(rows_local)

    # Tabla (b) Candidatos HF
    candidatos = auditoria.get("candidatos", [])
    rows_cand = []
    for cand in candidatos:
        repo = cand.get("repo", "")
        fam = cand.get("familia", "")
        inst = "Sí" if cand.get("instruct") else "No"
        ggufs = cand.get("gguf", [])
        if ggufs:
            files_desc = "<br>".join([f"`{g['file']}` ({format_bytes(g.get('bytes'))})" for g in ggufs[:3]])
            if len(ggufs) > 3:
                files_desc += f"<br>*(+{len(ggufs)-3} archivos adicionales)*"
        else:
            files_desc = "Sin archivos GGUF declarados"
        dl = cand.get("downloads", 0)
        rows_cand.append(f"| [{repo}]({cand.get('fuente', '#')}) | `{fam}` | {inst} | {files_desc} | {dl:,} |")

    tabla_candidatos = "| repo HF | familia | instruct | archivos GGUF (bytes) | downloads |\n|---|---|---|---|---|\n" + "\n".join(rows_cand)

    sec3 = f"""## 3. Qué arquitecturas Mamba existen y cuáles carga nuestro llama.cpp

### Soporte en `libllama.so`
Estado verificado mediante inspección binaria de cadenas:
{sop_str}

*(Nota: `zamba2` no está implementada en esta compilación de llama.cpp).*

### Modelos locales evaluados
{tabla_locales}

### Candidatos relevados en HuggingFace Hub
{tabla_candidatos}
"""

    # 4. Seccion 4 (texto fijo exacto requerido)
    sec4 = """## 4. Por qué la comparación de calidad (coherencia-larga, baseline 30,0%) no se puede cerrar

> El baseline 30,0 % de `coherencia-larga` fue medido en el kernel `eval29v7` sobre
> `kateto-rwkv29-merged.pth` (2,9B, `max_len=256`, T=0,80, top_p=0,70, guarda de n-gramas n=3/repeat_max=12)
> — **no** sobre el 1.5B que menciona la card. El gate, además, fue medido contra un juez independiente con
> kappa 0,1986 (card t_5361ca0b): cuenta degeneración, no coherencia conversacional. Para comparar calidad
> hace falta un Mamba del mismo tamaño con fine-tune equivalente sobre los mismos datos (SFT + ORPO), y eso
> no existe: `rwkv-peft-training`/RWKV-PEFT no entrena Mamba y el venv de entrenamiento no tiene
> `mamba_ssm` ni `causal_conv1d`. Un número de `coherencia-larga` sacado con `mamba-1.4b-instruct` mediría
> el dataset de otro, no la arquitectura.
"""

    # 5. Seccion 5 Gates
    gates_content = ""
    # Try reading from gates_path or /tmp/gates.log
    check_paths = []
    if gates_path:
        check_paths.append(Path(gates_path))
    check_paths.append(Path("/tmp/gates.log"))

    for cp in check_paths:
        if cp.is_file():
            try:
                gates_content = cp.read_text(encoding="utf-8").strip()
                if gates_content:
                    break
            except Exception:
                pass

    if not gates_content:
        gates_content = "*(Gates no registrados aún o ejecutados externamente)*"

    sec5 = f"""## 5. Gates

```text
{gates_content}
```
"""

    content = f"# Mamba vs RWKV — medición (t_0240ba88)\n\n{sec1}\n{sec2}\n{sec3}\n{sec4}\n{sec5}"

    p_out = Path(out_path)
    p_out.parent.mkdir(parents=True, exist_ok=True)
    with open(p_out, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Renderizar informe INFORME.md")
    parser.add_argument("--tput", required=True, help="Ruta a tput JSON")
    parser.add_argument("--auditoria", required=True, help="Ruta a auditoria JSON")
    parser.add_argument("--out", required=True, help="Ruta de salida INFORME.md")
    parser.add_argument("--gates", default=None, help="Ruta opcional a log de gates")
    args = parser.parse_args()

    render_report(args.tput, args.auditoria, args.out, args.gates)


if __name__ == "__main__":
    main()
