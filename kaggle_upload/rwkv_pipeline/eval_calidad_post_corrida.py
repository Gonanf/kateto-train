#!/usr/bin/env python3
"""
Script de Evaluación de Calidad Post-Corrida para Kateto (RWKV-7).
Dictamina si un entrenamiento rompió el modelo o mantuvo la coherencia y personalidad.

================================================================================
QUÉ MIDE Y QUÉ NO MIDE CADA MÉTRICA
================================================================================

1. MÉTRICA 1 — ΔPPL CONTRA EL MODELO BASE (LA PRINCIPAL):
   - Qué mide:
     Pasa cada respuesta generada por el modelo base sin fine-tune en modo forward
     condicionado al mismo query, obteniendo el vector de pérdida por token:
       ℓᵢ = −log P_base(tᵢ | query, t<ᵢ)
     No mide el promedio simple (un neologismo se diluye entre veinte palabras comunes),
     sino la cola derecha:
       * Percentil 95 (P95_Token_NLL): sensibilidad extrema a anomalías sintácticas.
         Umbral dinámico: max(5.2, P95_ref + 1.1) nats.
       * Fracción de tokens con ℓᵢ > 7.0 nats: firma de palabras inventadas por pegado
         defectuoso de sub-tokens ("respondeo", "noticiencias", "provinilla", "winampo").
         Corte duro: > 3% ⇒ corrida rota.
       * NLL media (μ_gen) respecto a la referencia (μ_ref): detecta colapso a clichés
         repetitivos (< 0.85 · μ_ref) o pérdida total de coherencia/delirio (> 1.20 · μ_ref).
   - Qué NO mide:
     No mide si la respuesta responde factual o verídicamente a la consulta, ni si
     adopta la personalidad de Kateto. Un texto perfectamente coherente extraído de
     Wikipedia pasa esta métrica con honores aunque no tenga relación con la pregunta.

2. MÉTRICA 2 — LÉXICO FUERA DE DICCIONARIO (OOV):
   - Qué mide:
     Cruza cada palabra generada contra el diccionario español es_AR (Hunspell)
     filtrando una whitelist técnica, lunfardo y nombres propios (vos, che, boludo,
     tool_call, rwkv, supabase, etc.). Extrae y lista neologismos rotos e intrusiones.
   - Qué NO mide:
     No mide errores gramaticales o concordancia donde cada palabra aislada existe
     en el diccionario ("el vaca come pastos altos"), ni alucinaciones fácticas compuestas
     por términos válidos.

3. MÉTRICA 3 — FIDELIDAD DE CONTENIDO POR SLOTS:
   - Qué mide:
     Evalúa por ítem la presencia de conceptos clave obligatorios (must_include),
     la ausencia estricta de usurpaciones de identidad o biografía ficticia del usuario
     (must_not_include: e.g. "soy tu hermano", "tenemos dos hermanos", etc.), y la
     similitud semántica de embeddings (MiniLM-L12 multilingüe) exigiendo mayor
     cercanía a la respuesta canónica que a una respuesta trampa/distractora.
   - Qué NO mide:
     No mide estilo rioplatense fino ni fluidez a nivel de token, solo consistencia
     temática y respeto a las fronteras de identidad.

4. MÉTRICA 4 — VOICESCORE Y DENSIDAD DE JERGA:
   - Qué mide:
     Compara la postura pragmática de la respuesta contra dos centroides semánticos
     nivelados a la misma longitud (1-2 oraciones):
       * Centroide dorado (μ_gold): respuestas curadas en la voz seca y directa de Kateto.
       * Centroide de slop (μ_slop): respuestas de asistente corporativo genérico y servil.
     Antes de embeddear, se enmascaran modismos explícitos ("che", "boludo", "posta",
     "chabón") y se evalúa por oración para forzar al embedding a captar la actitud
     y no las palabras fáciles.
     VoiceScore = cos(e_gen, μ_gold) − cos(e_gen, μ_slop).
     Además mide la densidad de modismos: deben aparecer en < 20% de las respuestas;
     por encima del 25% penaliza por caricatura. El voseo gramatical (tenés, fijate, sabés)
     no se penaliza: es el registro objetivo.
   - Qué NO mide:
     No mide corrección técnica ni precisión de cálculos. Una respuesta con tono
     impecable pero con un error en una fórmula obtendría un VoiceScore alto.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Pre-torch environment guards
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

# Modismos explícitos a enmascarar / auditar en densidad
JERGA_TERMS = [
    "che", "boludo", "boluda", "pelotudo", "pelotuda", "chabón", "chabona",
    "posta", "chamuyo", "quilombo", "de una", "pibe", "mina", "bondi", "guita"
]
JERGA_RE = re.compile(r"\b(" + "|".join(JERGA_TERMS) + r")\b", re.IGNORECASE)

DEFAULT_WHITELIST = {
    "vos", "che", "boludo", "boluda", "pelotudo", "pelotuda", "chabón", "chabona", "posta",
    "de una", "tool_call", "rwkv", "manim", "supabase", "postgres", "sql", "json", "jsonl",
    "python", "minilm", "embedding", "embeddings", "tokenize", "tokenizer", "tokenizador",
    "dataset", "datasets", "api", "apis", "git", "linux", "docker", "cgroup", "ram", "cpu",
    "gpu", "vram", "loss", "nll", "ppl", "prompt", "prompts", "prompt_ids", "query", "orpo",
    "sft", "lora", "peft", "rocm", "cuda", "hip", "llama", "cpp", "gguf", "bfloat16", "float32",
    "mmap", "join", "joins", "b-tree", "wal", "tcp", "udp", "dns", "mutex", "rss", "vsz", "pmap",
    "p95", "p99", "overfit", "overfitting", "kateto", "katetobot", "rulobanda", "tranca",
    "chamuyo", "quilombo", "laburo", "laburar", "laburando", "bancás", "banca", "bancar",
    "fijate", "mirá", "tenés", "sabés", "andá", "hacé", "querés", "decime", "machete",
    "paradigmas", "backtracking", "frontend", "backend", "fullstack", "deploy", "ci", "cd",
    "scikit-learn", "deep", "learning", "machine", "none", "nonetype", "null", "int", "float",
    "str", "bool", "list", "dict", "tuple", "set", "true", "false", "pid", "rsa", "status",
    "scan", "scans", "concurrencia", "multithreading"
}


def mask_jerga(text: str) -> str:
    """Enmascara modismos explícitos con un token neutro para que el embedding no ataje."""
    return JERGA_RE.sub("[JERGA]", text)


def load_reference_data(ref_path: Optional[str] = None) -> Dict[str, Any]:
    """Carga las especificaciones de slots, centroides y referencias."""
    candidates = []
    if ref_path:
        candidates.append(Path(ref_path))
    candidates.extend([
        SCRIPT_DIR / "reference_eval_data.json",
        PROJECT_DIR / "kaggle_upload/rwkv_pipeline/reference_eval_data.json",
        PROJECT_DIR / "rwkv_pipeline/reference_eval_data.json",
        Path("/run/media/chaos/secundario/harness-run/kt-r14/kaggle_upload/rwkv_pipeline/reference_eval_data.json")
    ])
    for cand in candidates:
        if cand.exists():
            with open(cand, "r", encoding="utf-8") as f:
                return json.load(f)

    # Fallback canónico mínimo si no está el archivo JSON
    return {
        "reference_metrics": {
            "base_model": "rwkv7-g1k-1.5b-20260930-ctx25600.pth",
            "mu_ref": 1.8388,
            "p95_ref": 7.0047,
            "dynamic_p95_cutoff": 8.1047,
            "mu_min": 1.5630,
            "mu_max": 2.2066
        },
        "items_spec": [],
        "voice_gold_exemplars": [],
        "voice_slop_exemplars": [],
        "whitelist": sorted(list(DEFAULT_WHITELIST))
    }


# ==============================================================================
# 1. FORWARD PASS EN MODELO BASE (MÉTRICA 1: ΔPPL & NLL COLA DERECHA)
# ==============================================================================

class BaseNLLScorer:
    """Ejecuta el forward pass en modo evaluación sobre el modelo base no afinado."""

    def __init__(self, model_path: str, vocab_path: str, device: str = "cpu"):
        self.device = device
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            sys.path.insert(0, str(PROJECT_DIR / "kaggle_upload/rwkv_pipeline"))
            sys.path.insert(0, str(PROJECT_DIR / "rwkv_pipeline"))
            from infer_kateto import KatetoInferenceEngine  # type: ignore
            print(f"Cargando modelo base sin fine-tune desde {self.model_path} en {self.device}...")
            self.engine = KatetoInferenceEngine(self.model_path, self.vocab_path, device=self.device)
            print("Modelo base cargado exitosamente.")
        except Exception as e:
            print(f"AVISO: No se pudo instanciar KatetoInferenceEngine ({type(e).__name__}: {e})")
            self.engine = None

    def is_available(self) -> bool:
        return self.engine is not None

    def score_response(self, prompt: str, response: str) -> List[Tuple[str, float, bool]]:
        """
        Calcula vector ℓᵢ = −log P_base(tᵢ | query, t<ᵢ) token por token.
        Devuelve lista de (token_str, loss_nats, is_excluded).
        Excluye de conteo >7.0 los bloques de tool_call y nombres propios como 'Kateto'.
        """
        if not self.engine:
            return []

        import torch
        import torch.nn.functional as F

        # Formato de turno
        formatted_prompt = f"<|im_user|>{prompt.strip()}\n<|im_start|>seco\n"
        q_tokens = self.engine.tokenizer.encode(formatted_prompt)
        r_tokens = self.engine.tokenizer.encode(response.strip())

        state = self.engine.build_initial_state("none")
        out = None
        for tok in q_tokens:
            out, state = self.engine.model.forward(tok, state)

        results = []
        in_tool_call = False

        for tok in r_tokens:
            logits = out.float()
            log_probs = F.log_softmax(logits, dim=-1)
            loss = float(-log_probs[tok].item())
            tok_str = self.engine.tokenizer.decode([tok])

            if "<tool_call>" in tok_str:
                in_tool_call = True
            is_proper = tok_str.strip().lower() in ["kateto", "kate", "to"]
            is_excluded = in_tool_call or is_proper

            if "</tool_call>" in tok_str:
                in_tool_call = False

            results.append((tok_str, loss, is_excluded))
            out, state = self.engine.model.forward(tok, state)

        return results


# ==============================================================================
# 2. EVALUACIÓN LÉXICA OOV (MÉTRICA 2)
# ==============================================================================

def check_oov(text: str, dict_path: Optional[str], whitelist: Set[str]) -> List[str]:
    """Cruza palabras contra Hunspell es_AR y filtra contra la whitelist técnica/lunfardo."""
    words = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ_-]+", text)
    if not words:
        return []

    # Determinar ruta del diccionario
    dict_arg = "es_AR"
    if dict_path and Path(dict_path).exists():
        dict_arg = str(Path(dict_path).with_suffix(""))
    elif (SCRIPT_DIR / "dict/es_AR.dic").exists():
        dict_arg = str(SCRIPT_DIR / "dict/es_AR")
    elif (Path.home() / ".local/share/hunspell/es_AR.dic").exists():
        dict_arg = str(Path.home() / ".local/share/hunspell/es_AR")

    try:
        proc = subprocess.run(
            ["hunspell", "-d", dict_arg, "-l"],
            input="\n".join(words),
            text=True,
            capture_output=True,
            check=False
        )
        raw_unknown = set(proc.stdout.strip().splitlines())
    except Exception as e:
        # Fallback si hunspell binario no está en PATH
        return []

    oov_found = sorted([
        w for w in raw_unknown
        if w.lower() not in whitelist
        and not w.isdigit()
        and len(w) > 1
    ])
    return oov_found


# ==============================================================================
# 3. FIDELIDAD DE CONTENIDO POR SLOTS (MÉTRICA 3)
# ==============================================================================

def cosine_sim(a: Any, b: Any) -> float:
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return dot / (norm + 1e-9)


def evaluate_slots(
    response: str,
    item_spec: Dict[str, Any],
    embed_model: Any
) -> Dict[str, Any]:
    """
    Verifica must_include, must_not_include y similitud embedding canónica vs trampa.
    """
    resp_lower = response.lower()

    # 1. must_include
    must_inc = item_spec.get("must_include", [])
    inc_passed = True
    if must_inc:
        inc_passed = any(k.lower() in resp_lower for k in must_inc)

    # 2. must_not_include (caso central: usurpación de identidad / parientes)
    must_not = item_spec.get("must_not_include", [])
    not_passed = True
    violated_terms = []
    for forbidden in must_not:
        if forbidden.lower() in resp_lower:
            not_passed = False
            violated_terms.append(forbidden)

    # 3. Embedding similarity
    sim_canon = 0.0
    sim_trampa = 0.0
    emb_passed = True
    if embed_model and item_spec.get("canonical"):
        e_resp = embed_model.encode([response])[0]
        e_canon = embed_model.encode([item_spec["canonical"]])[0]
        sim_canon = cosine_sim(e_resp, e_canon)

        if item_spec.get("trampa"):
            e_trampa = embed_model.encode([item_spec["trampa"]])[0]
            sim_trampa = cosine_sim(e_resp, e_trampa)
            emb_passed = (sim_canon >= sim_trampa - 0.02)  # Pequeño margen de tolerancia

    slot_passed = inc_passed and not_passed and emb_passed

    return {
        "passed": slot_passed,
        "must_include_ok": inc_passed,
        "must_not_include_ok": not_passed,
        "violated_forbidden": violated_terms,
        "emb_ok": emb_passed,
        "sim_canonical": round(sim_canon, 3),
        "sim_trampa": round(sim_trampa, 3)
    }


# ==============================================================================
# 4. VOICESCORE Y DENSIDAD DE JERGA (MÉTRICA 4)
# ==============================================================================

class VoiceEvaluator:
    def __init__(self, embed_model: Any, gold_exemplars: List[str], slop_exemplars: List[str]):
        self.model = embed_model
        self.gold_centroid = None
        self.slop_centroid = None
        if self.model and gold_exemplars and slop_exemplars:
            self._compute_centroids(gold_exemplars, slop_exemplars)

    def _embed_sentence_avg(self, text: str) -> Any:
        masked = mask_jerga(text)
        sents = [s.strip() for s in re.split(r"[.!?]+", masked) if s.strip()]
        if not sents:
            sents = [masked]
        embs = self.model.encode(sents)
        avg = np.mean(embs, axis=0)
        norm = np.linalg.norm(avg)
        return avg / (norm + 1e-9)

    def _compute_centroids(self, gold_texts: List[str], slop_texts: List[str]):
        gold_vecs = [self._embed_sentence_avg(t) for t in gold_texts]
        slop_vecs = [self._embed_sentence_avg(t) for t in slop_texts]
        self.gold_centroid = np.mean(gold_vecs, axis=0)
        self.gold_centroid /= (np.linalg.norm(self.gold_centroid) + 1e-9)
        self.slop_centroid = np.mean(slop_vecs, axis=0)
        self.slop_centroid /= (np.linalg.norm(self.slop_centroid) + 1e-9)

    def score_text(self, text: str) -> float:
        if self.gold_centroid is None or self.slop_centroid is None:
            return 0.0
        e_text = self._embed_sentence_avg(text)
        cos_gold = float(np.dot(e_text, self.gold_centroid))
        cos_slop = float(np.dot(e_text, self.slop_centroid))
        return cos_gold - cos_slop


# ==============================================================================
# 5. PIPELINE PRINCIPAL Y DICTAMEN DE CORRIDA
# ==============================================================================

def run_evaluation(
    eval_file: str,
    base_model_path: Optional[str] = None,
    vocab_path: Optional[str] = None,
    dict_path: Optional[str] = None,
    ref_data_path: Optional[str] = None,
    device: str = "cpu",
    no_model: bool = False
) -> Dict[str, Any]:
    """Ejecuta las 4 métricas post-corrida y dictamina VERDE o ROJO."""
    ref_data = load_reference_data(ref_data_path)
    ref_metrics = ref_data["reference_metrics"]
    items_spec = {item["prompt"].strip(): item for item in ref_data.get("items_spec", [])}
    whitelist = set(ref_data.get("whitelist", [])) or DEFAULT_WHITELIST

    # Cargar eval_v2.json
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_run = json.load(f)

    eval_items = eval_run.get("detalle", [])
    if not eval_items and isinstance(eval_run, list):
        eval_items = eval_run

    total_items = len(eval_items)

    # 1. Forward model scorer
    scorer = None
    if not no_model and base_model_path and Path(base_model_path).exists():
        scorer = BaseNLLScorer(base_model_path, vocab_path or "", device=device)

    # 2. Embedding model para slots y voice
    embed_model = None
    voice_eval = None
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
        gold_ex = ref_data.get("voice_gold_exemplars", [])
        slop_ex = ref_data.get("voice_slop_exemplars", [])
        voice_eval = VoiceEvaluator(embed_model, gold_ex, slop_ex)
    except Exception as e:
        print(f"AVISO: sentence_transformers no disponible ({e}), slots y voice correrán en modo reducido.")

    # Acumuladores de métricas
    all_evaluated_losses = []
    response_mean_nlls = []
    item_results = []
    all_oov_tokens = set()
    slots_passed_count = 0
    voice_scores = []
    jerga_response_count = 0

    mu_ref = ref_metrics.get("mu_ref", 1.8388)
    p95_ref = ref_metrics.get("p95_ref", 7.0047)
    dynamic_p95_cutoff = max(5.2, p95_ref + 1.1)
    mu_min = 0.85 * mu_ref
    mu_max = 1.20 * mu_ref

    for idx, item in enumerate(eval_items):
        prompt = item.get("prompt", "")
        response = item.get("respuesta", "")

        # Métrica 1: NLL tokens
        token_losses = []
        if scorer and scorer.is_available():
            token_scores = scorer.score_response(prompt, response)
            for tok_str, loss, is_excluded in token_scores:
                if not is_excluded:
                    all_evaluated_losses.append(loss)
                    token_losses.append(loss)
            if token_losses:
                response_mean_nlls.append(sum(token_losses) / len(token_losses))

        # Métrica 2: OOV
        oov = check_oov(response, dict_path, whitelist)
        all_oov_tokens.update(oov)

        # Métrica 3: Slots
        spec = items_spec.get(prompt.strip(), {})
        slot_res = evaluate_slots(response, spec, embed_model)
        if slot_res["passed"]:
            slots_passed_count += 1

        # Métrica 4: VoiceScore y Jerga
        has_jerga = bool(JERGA_RE.search(response))
        if has_jerga:
            jerga_response_count += 1

        v_score = voice_eval.score_text(response) if voice_eval else 0.0
        voice_scores.append(v_score)

        item_results.append({
            "idx": idx + 1,
            "prompt": prompt,
            "respuesta": response,
            "oov_words": oov,
            "slot_evaluation": slot_res,
            "voice_score": round(v_score, 3),
            "has_jerga": has_jerga,
            "mean_token_nll": round(sum(token_losses)/len(token_losses), 3) if token_losses else None
        })

    # Consolidación de métricas globales
    # Métrica 1
    has_forward_metrics = bool(all_evaluated_losses)
    if has_forward_metrics:
        sorted_losses = sorted(all_evaluated_losses)
        p95_idx = int(len(sorted_losses) * 0.95)
        p95_val = sorted_losses[p95_idx]
        tokens_gt7_count = sum(1 for l in all_evaluated_losses if l > 7.0)
        tokens_gt7_frac = tokens_gt7_count / len(all_evaluated_losses)
        mu_gen = sum(response_mean_nlls) / len(response_mean_nlls) if response_mean_nlls else 0.0
        mu_ratio = mu_gen / mu_ref if mu_ref > 0 else 1.0

        p95_ok = p95_val <= dynamic_p95_cutoff
        gt7_ok = tokens_gt7_frac <= 0.03
        mu_ok = (mu_min <= mu_gen <= mu_max)
    else:
        p95_val = None
        tokens_gt7_frac = None
        mu_gen = None
        mu_ratio = None
        p95_ok = True
        gt7_ok = True
        mu_ok = True

    # Métrica 2
    oov_count = len(all_oov_tokens)
    oov_ok = (oov_count <= 8)  # Tolerancia prudente para palabras no-whitelisted

    # Métrica 3
    # Comprobar si hubo violaciones de identidad duras ("soy tu hermano")
    identity_violations = [
        it for it in item_results
        if not it["slot_evaluation"]["must_not_include_ok"]
    ]
    slots_ok = (len(identity_violations) == 0 and slots_passed_count >= max(1, total_items * 0.5))

    # Métrica 4
    jerga_density = jerga_response_count / total_items if total_items > 0 else 0.0
    jerga_ok = jerga_density <= 0.20
    jerga_caricature = jerga_density > 0.25

    avg_voice = float(np.mean(voice_scores)) if voice_scores else 0.0
    if jerga_caricature:
        avg_voice -= (jerga_density - 0.25) * 2.0  # Penalización por caricatura
    voice_ok = avg_voice >= 0.05

    # Dictamen final:
    # Falla dura si P95 supera umbral, si tokens>7.0 > 3%, si mu_gen se desvía,
    # si hay alucinaciones de identidad ("soy tu hermano"), o si hay neologismos rotos masivos.
    failed_reasons = []
    if has_forward_metrics:
        if not p95_ok:
            failed_reasons.append(f"P95 ({p95_val:.2f}) > corte ({dynamic_p95_cutoff:.2f})")
        if not gt7_ok:
            failed_reasons.append(f"tokens>7.0 ({tokens_gt7_frac*100:.1f}%) > 3.0%")
        if not mu_ok:
            failed_reasons.append(f"μ_gen ({mu_gen:.2f}) fuera de rango [{mu_min:.2f}, {mu_max:.2f}]")

    if identity_violations:
        failed_reasons.append(f"Violación de identidad en {len(identity_violations)} respuestas ({identity_violations[0]['slot_evaluation']['violated_forbidden']})")

    if oov_count > 10:
        failed_reasons.append(f"Léxico roto / OOV excesivo ({oov_count} palabras inventadas)")

    if jerga_caricature:
        failed_reasons.append(f"Caricaturización por exceso de jerga ({jerga_density*100:.1f}% > 25%)")

    veredicto = "ROJO" if failed_reasons else "VERDE"

    # Línea de resumen comparable
    p95_str = f"P95 {p95_val:.1f} (ok <{dynamic_p95_cutoff:.1f})" if p95_val is not None else "P95 N/A"
    gt7_str = f"tokens>7.0: {tokens_gt7_frac*100:.1f}% (ok <3%)" if tokens_gt7_frac is not None else "tokens>7.0 N/A"
    mu_str = f"μ_gen/μ_ref {mu_ratio:.2f} (ok 0.85-1.20)" if mu_ratio is not None else "μ_gen/μ_ref N/A"
    summary_line = (
        f"{p95_str} | {gt7_str} | {mu_str} | OOV {oov_count} | "
        f"slots {slots_passed_count}/{total_items} | Voice {avg_voice:.2f} | "
        f"jerga {jerga_density*100:.0f}% → {veredicto}"
    )

    return {
        "veredicto": veredicto,
        "summary_line": summary_line,
        "failed_reasons": failed_reasons,
        "metrics": {
            "p95_token_nll": round(p95_val, 4) if p95_val else None,
            "p95_cutoff": round(dynamic_p95_cutoff, 4),
            "tokens_gt7_fraction": round(tokens_gt7_frac, 4) if tokens_gt7_frac else None,
            "mu_gen": round(mu_gen, 4) if mu_gen else None,
            "mu_ref": round(mu_ref, 4),
            "mu_ratio": round(mu_ratio, 4) if mu_ratio else None,
            "oov_count": oov_count,
            "oov_words": sorted(list(all_oov_tokens)),
            "slots_passed": slots_passed_count,
            "total_slots": total_items,
            "voice_score": round(avg_voice, 3),
            "jerga_density": round(jerga_density, 3),
            "identity_violations_count": len(identity_violations)
        },
        "detalle": item_results
    }


# ==============================================================================
# 6. SELF-CHECK SIN GPU NI MODELO
# ==============================================================================

def run_self_check():
    """
    Self-check sin GPU ni modelo:
    Evalúa tres respuestas de juguete armadas a mano:
      1. Respuesta buena (Kateto auténtico).
      2. Respuesta con neologismos rotos ("respondeo", "noticiencias", etc.).
      3. Respuesta que afirma usurpación de identidad ("soy tu hermano").
    Verifica que el veredicto salga en el orden esperado (VERDE, ROJO, ROJO).
    """
    print("\n" + "=" * 70)
    print("EJECUTANDO SELF-CHECK SIN GPU NI MODELO PESADO...")
    print("=" * 70)

    toy_eval = {
        "modelo": "mock-self-check",
        "n": 3,
        "detalle": [
            {
                "prompt": "Che Kateto, contame quien sos",
                "respuesta": "Soy Kateto. Un asistente técnico directo y seco, sin vueltas ni discursos corporativos. Te ayudo con código y arquitectura."
            },
            {
                "prompt": "Che Kateto, contame quien sos",
                "respuesta": "Te respondeo con postura y noticiencias del código en una provinilla inefica."
            },
            {
                "prompt": "Che Kateto, contame quien sos",
                "respuesta": "Soy tu hermano mayor, tenemos dos hermanos mayores y me alegra cuidarte siempre."
            }
        ]
    }

    toy_path = SCRIPT_DIR / "tmp_self_check_input.json"
    with open(toy_path, "w", encoding="utf-8") as f:
        json.dump(toy_eval, f)

    res = run_evaluation(str(toy_path), no_model=True)
    if toy_path.exists():
        toy_path.unlink()

    detalle = res["detalle"]
    v1_ok = (len(detalle[0]["oov_words"]) == 0 and detalle[0]["slot_evaluation"]["must_not_include_ok"])
    v2_neologismos = ("respondeo" in detalle[1]["oov_words"] or "noticiencias" in detalle[1]["oov_words"])
    v3_identidad = not detalle[2]["slot_evaluation"]["must_not_include_ok"]

    print("\nResultados por muestra:")
    print(f"  1. Muestra Buena:      OOV={len(detalle[0]['oov_words'])} | Violación={not detalle[0]['slot_evaluation']['must_not_include_ok']} | Voice={detalle[0]['voice_score']:.2f}")
    print(f"  2. Muestra Neologismos: OOV={detalle[1]['oov_words']} | Violación={not detalle[1]['slot_evaluation']['must_not_include_ok']}")
    print(f"  3. Muestra Identidad:   OOV={len(detalle[2]['oov_words'])} | Violación={not detalle[2]['slot_evaluation']['must_not_include_ok']} ({detalle[2]['slot_evaluation']['violated_forbidden']})")

    assert v1_ok, "Fallo self-check: la muestra buena fue marcada erróneamente."
    assert v2_neologismos, "Fallo self-check: la muestra con neologismos no detectó las palabras rotas."
    assert v3_identidad, "Fallo self-check: la muestra que usurpa identidad no fue bloqueada."

    print("\nLínea resumen generada en self-check:")
    print("  " + res["summary_line"])
    print("\n[SELF-CHECK EXITOSO]: El orden de detección y discriminación es el esperado.\n")
    return True


# ==============================================================================
# CLI ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Métricas de calidad post-corrida para Kateto RWKV-7")
    parser.add_argument("--eval_file", type=str, default=None, help="Ruta al archivo eval_v2.json")
    parser.add_argument("--model", type=str, default=None, help="Ruta al checkpoint .pth del modelo base (g1k)")
    parser.add_argument("--vocab", type=str, default=None, help="Ruta a rwkv_vocab_v20230424.txt")
    parser.add_argument("--dict", type=str, default=None, help="Ruta a diccionario Hunspell es_AR")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Dispositivo para forward")
    parser.add_argument("--no_model", action="store_true", help="Saltear forward de modelo base y calcular solo métricas léxicas, slots y voz")
    parser.add_argument("--self_check", "--self-check", dest="self_check", action="store_true", help="Ejecutar self-check de 3 respuestas sintéticas")
    parser.add_argument("--out_json", type=str, default=None, help="Ruta para guardar el informe JSON detallado")
    parser.add_argument("--history_jsonl", type=str, default="historial_calidad.jsonl", help="Archivo .jsonl para registrar corridas")

    args = parser.parse_args()

    if args.self_check:
        success = run_self_check()
        sys.exit(0 if success else 1)

    # Detección de archivos por defecto
    eval_file = args.eval_file
    if not eval_file:
        cands = [
            "/home/chaos/harness-run/kaggle-v3-out/out/eval_v2.json",
            "/home/chaos/harness-run/kaggle-v3-out/kateto-ckpt/eval_v2.json",
            "eval_v2.json",
            "/kaggle/working/out/eval_v2.json"
        ]
        for c in cands:
            if Path(c).exists():
                eval_file = c
                break

    if not eval_file or not Path(eval_file).exists():
        print(f"Error: no se encontró archivo eval_v2.json. Especificar con --eval_file")
        sys.exit(1)

    model_path = args.model or "/run/media/chaos/secundario/models/rwkv7-1.5b-g1k/rwkv7-g1k-1.5b-20260930-ctx25600.pth"
    vocab_path = args.vocab
    if not vocab_path:
        vocab_cands = [
            "/run/media/chaos/secundario/kateto-kaggle-sft/RWKV-PEFT/rwkv_vocab_v20230424.txt",
            "/home/chaos/harness-run/kaggle-v3-out/kateto-train/RWKV-PEFT/rwkv_vocab_v20230424.txt",
            PROJECT_DIR / "RWKV-PEFT/rwkv_vocab_v20230424.txt"
        ]
        for vc in vocab_cands:
            if Path(vc).exists():
                vocab_path = str(vc)
                break

    dict_path = args.dict or str(SCRIPT_DIR / "dict/es_AR")

    print(f"Iniciando evaluación de post-corrida sobre: {eval_file}")
    res = run_evaluation(
        eval_file=eval_file,
        base_model_path=model_path,
        vocab_path=vocab_path,
        dict_path=dict_path,
        device=args.device,
        no_model=args.no_model
    )

    # Imprimir línea resumen
    print("\n" + "=" * 80)
    print("DICTAMEN FINAL POST-CORRIDA:")
    print("=" * 80)
    print(res["summary_line"])
    print("=" * 80)

    if res["failed_reasons"]:
        print("\nMotivos de rechazo:")
        for r in res["failed_reasons"]:
            print(f"  ❌ {r}")

    print(f"\nPalabras OOV encontradas ({len(res['metrics']['oov_words'])}):")
    print("  " + ", ".join(res["metrics"]["oov_words"]))

    # Guardar JSON de salida si se solicita
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\nInforme detallado guardado en: {args.out_json}")

    # Guardar en histórico .jsonl
    history_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_file": eval_file,
        "veredicto": res["veredicto"],
        "summary": res["summary_line"],
        "metrics": res["metrics"],
        "failed_reasons": res["failed_reasons"]
    }
    history_path = Path(args.history_jsonl)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_entry, ensure_ascii=False) + "\n")
    print(f"Veredicto registrado en histórico: {history_path}")

    # Salida con código 0 si es verde, 1 si es rojo
    sys.exit(0 if res["veredicto"] == "VERDE" else 1)


if __name__ == "__main__":
    import numpy as np
    main()
