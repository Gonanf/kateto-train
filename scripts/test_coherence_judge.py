#!/usr/bin/env python3
"""Anti-hack test suite for coherence_judge.py.

Runs standalone using only Python standard library.
Exit 0 on success, exit 1 on failure.
Prints the complete test table with case, category, coherence, and components.
"""

import json
import os
import subprocess
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from coherence_judge import score_text

# Control test cases embedded as literals (no runtime file reads)
# Sources for 'coherente':
# - /run/media/chaos/terciario/proyectos/kateto-train/data/kateto_v8_5k.jsonl
# - /run/media/chaos/terciario/proyectos/kateto-train/data/podcast/*.jsonl
CONTROL_CASES = [
    # --- COHERENTE (>= 6) ---
    {
        "id": "coh_kateto_1",
        "category": "coherente",
        "text": (
            "Mirá que la cosa es simple. El debate sobre «el comercio de cercanía sostiene el "
            "tejido barrial tradicional» se zanja cuando observás el impacto positivo en el "
            "terreno. Un error no deja de ser error porque lo repita una mayoría ruidosa."
        ),
    },
    {
        "id": "coh_kateto_2",
        "category": "coherente",
        "text": (
            "Dejémonos de rodeos. Afirmar que «el trabajo en plataformas demanda una nueva "
            "legislación laboral» significa poner por delante los intereses de la mayoría. "
            "Hay que separar las simpatías personales del análisis riguroso de fondo."
        ),
    },
    {
        "id": "coh_kateto_3",
        "category": "coherente",
        "text": (
            "A ver, seamos serios. El concepto de que «los envases retornables deberían ser "
            "obligatorios por norma» minimiza riesgos severos que otros países ya sufrieron. "
            "La única garantía duradera reside en instituciones transparentes y auditables."
        ),
    },
    {
        "id": "coh_kateto_4",
        "category": "coherente",
        "text": (
            "Cortita y al pie. El planteo favorable a «la industria del software es el mayor motor "
            "exportador futuro» desarticula excusas burocráticas insostenibles. Esa es la diferencia "
            "sustancial entre proyectar con seriedad o improvisar sobre la marcha."
        ),
    },
    {
        "id": "coh_kateto_5",
        "category": "coherente",
        "text": (
            "Sin vueltas, mirá. Creer ingenuamente que «la enseñanza de programación debe arrancar "
            "en primaria» desconoce el daño colateral que causaría en el tejido social. La única "
            "garantía duradera reside en instituciones transparentes y auditables."
        ),
    },
    {
        "id": "coh_podcast_1",
        "category": "coherente",
        "text": (
            "Por supuesto, porque es un consumo utilitario y cínico a la vez, viste, es una cosa "
            "compleja. Me gustó mucho el de uno de nuestros preferidos, que es La Madrid, ¿no? "
            "Ah, es que es una locura la historia de La Madrid."
        ),
    },
    {
        "id": "coh_podcast_2",
        "category": "coherente",
        "text": (
            "La Madrid a mí me enamoró cuando es un tipo que no lo querríamos ideológicamente, ¿no? "
            "Seguramente, pero un tipo increíble, una vida increíble. Y es que, o sea, por empezar "
            "es de los primeros inmortales."
        ),
    },
    # --- BUCLE (>= 3) ---
    {
        "id": "bucle_plantilla",
        "category": "bucle",
        "text": (
            ">seco\n[user]\n\n<|im_start|>seco\n>seco\n[user]\n\n"
            "<|im_start|>seco\n>seco\n[user]\n\n<|im_start|>seco\n>seco\n[user]\n\n"
            "<|im_start|>seco"
        ),
    },
    {
        "id": "bucle_sistema",
        "category": "bucle",
        "text": (
            "El sistema no responde correctamente al comando ingresado. Intente nuevamente más tarde.\n"
            "El sistema no responde correctamente al comando ingresado. Intente nuevamente más tarde.\n"
            "El sistema no responde correctamente al comando ingresado. Intente nuevamente más tarde.\n"
            "El sistema no responde correctamente al comando ingresado. Intente nuevamente más tarde."
        ),
    },
    {
        "id": "bucle_espanol_puro",  # 100% palabras españolas comunes (caso difícil)
        "category": "bucle",
        "text": (
            "No hay duda de que la verdad es lo primero que se debe buscar en este mundo. "
            "No hay duda de que la verdad es lo primero que se debe buscar en este mundo. "
            "No hay duda de que la verdad es lo primero que se debe buscar en este mundo. "
            "No hay duda de que la verdad es lo primero que se debe buscar en este mundo."
        ),
    },
    # --- ENSALADA (>= 3) ---
    {
        "id": "ensalada_codigo",
        "category": "ensalada",
        "text": (
            "Assemble autores con AsyncBar+sync migration a LFS(777) au elemento sequence unificado "
            "para everything 115082f dc2 namespace Assistant: PATCH "
            "SpecialRunDashboardSettingsMutationDataV3_InferiorCuldaPrefAbogacsDropdownValue SHA256=... "
            "wrote `utils/generatorTemplates/${branchDir}/templates/DroneProductionVariableModel/${alias}.?fragment_generationId=${taskId"
        ),
    },
    {
        "id": "ensalada_tokens",
        "category": "ensalada",
        "text": (
            "Maf score 46 control mastery < can+pick < sln Aniar duer hacia nuevo territord sp ~ 1017 "
            "26% exit 0x89abf42 --filter=node_modules/cache/pkg/run.cpp::eval_point?res=false [ERR] 992341"
        ),
    },
    {
        "id": "ensalada_espanol_rota",  # Muchas palabras españolas pero sintaxis rota (caso difícil)
        "category": "ensalada",
        "text": (
            "Sin embargo cuando entonces pero porque casa perro mesa arriba azul correr saltando "
            "donde hacia muy antes después mientras aunque si bien por lo tanto cada cual quien cuyo "
            "dondequiera jamás nunca nada nadie tampoco."
        ),
    },
    # --- DEGENERADO (>= 2) ---
    {
        "id": "degen_espacios",
        "category": "degenerado",
        "text": "   \n\n\t   \n  ",
    },
    {
        "id": "degen_tokens_control",
        "category": "degenerado",
        "text": "<|im_start|>seco\n<|im_end|>\n<query>\n```\n```\n<|im_start|>",
    },
    {
        "id": "degen_vacio",
        "category": "degenerado",
        "text": "",
    },
]


def run_tests() -> bool:
    print(f"{'Caso':22} | {'Categoría':10} | {'Coherence':9} | {'Componentes'}")
    print("-" * 88)

    results_by_cat: dict[str, list[float]] = {}
    scored_cases: dict[str, dict] = {}

    for case in CONTROL_CASES:
        cid = case["id"]
        cat = case["category"]
        text = case["text"]

        res = score_text(text)
        coh = res["coherence"]
        results_by_cat.setdefault(cat, []).append(coh)
        scored_cases[cid] = res

        comp_str = json.dumps(res["components"], ensure_ascii=False)
        print(f"{cid:22} | {cat:10} | {coh:9.4f} | {comp_str}")

    print("-" * 88)

    min_coh = min(results_by_cat["coherente"])
    max_buc = max(results_by_cat["bucle"])
    max_ens = max(results_by_cat["ensalada"])
    max_deg = max(results_by_cat["degenerado"])

    margin_bucle = min_coh - max_buc
    margin_ensalada = min_coh - max_ens
    margin_degenerado = min_coh - max_deg

    print(f"min(coherente):   {min_coh:.4f}")
    print(f"max(bucle):       {max_buc:.4f} (margen: {margin_bucle:+.4f})")
    print(f"max(ensalada):    {max_ens:.4f} (margen: {margin_ensalada:+.4f})")
    print(f"max(degenerado):  {max_deg:.4f} (margen: {margin_degenerado:+.4f})")
    print("-" * 88)

    # Assertion 1: min(coherente) > max(bucle) y min(coherente) > max(ensalada), margen >= 0.05
    if margin_bucle < 0.05:
        sys.stderr.write(f"FAIL: margen con bucle es {margin_bucle:.4f} < 0.05\n")
        return False
    if margin_ensalada < 0.05:
        sys.stderr.write(f"FAIL: margen con ensalada es {margin_ensalada:.4f} < 0.05\n")
        return False

    # Assertion 2: min(coherente) > max(degenerado)
    if margin_degenerado <= 0.0:
        sys.stderr.write(f"FAIL: coherente no supera a degenerado: {margin_degenerado:.4f}\n")
        return False

    # Assertion 5: bucle-español queda estrictamente por debajo del coherente más débil
    bucle_esp_score = scored_cases["bucle_espanol_puro"]["coherence"]
    if bucle_esp_score >= min_coh:
        sys.stderr.write(f"FAIL: bucle_espanol_puro ({bucle_esp_score:.4f}) >= min(coherente) ({min_coh:.4f})\n")
        return False

    # Assertion 3: Determinismo - misma llamada da idéntico resultado
    for case in CONTROL_CASES:
        t = case["text"]
        r1 = score_text(t)
        r2 = score_text(t)
        if r1 != r2:
            sys.stderr.write(f"FAIL: No determinista para {case['id']}: {r1} != {r2}\n")
            return False

    # Determinismo entre procesos
    subproc_code = (
        "import json, sys, os\n"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else 'scripts')\n"
        "from coherence_judge import score_text\n"
        "text = sys.stdin.read()\n"
        "print(json.dumps(score_text(text)))\n"
    )
    for case in CONTROL_CASES[:3]:
        proc = subprocess.run(
            [sys.executable, "-c", subproc_code],
            input=case["text"],
            cwd=_SCRIPTS_DIR,
            text=True,
            capture_output=True,
            check=True
        )
        proc_res = json.loads(proc.stdout.strip())
        local_res = scored_cases[case["id"]]
        if proc_res != local_res:
            sys.stderr.write(f"FAIL: Determinismo entre procesos falló en {case['id']}\n")
            return False

    # Assertion 4: Pureza - prompt no afecta resultado
    sample_text = CONTROL_CASES[0]["text"]
    s_none = score_text(sample_text, prompt=None)
    s_p1 = score_text(sample_text, prompt="Che Kateto contame quien sos")
    s_p2 = score_text(sample_text, prompt="Otro prompt completamente distinto")
    if s_none != s_p1 or s_none != s_p2:
        sys.stderr.write("FAIL: Pureza violada - el prompt alteró la puntuación\n")
        return False

    print("ALL ASSERTIONS PASSED (OK)")
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
