#!/usr/bin/env python3
"""
build_qa_dataset.py — Fase 1 Datos: extracción Q/A desde clon state.clone.db

Reusa kateto/tools/dataset_filter (secrets, ai_filter, splitter, classifier, generator)
pero con defaults Kateto: solo clon, Orion-26B secuencial, respeto README triggers.

Filtros (orden crítico): secrets primero → IA-like → dedup → split → clasifica Q/A → genera faltante
con Orion-26B respetando triggers/golden rules (quote textual, cero fluff, split bilingüe).

Salida: {question, answer} por línea, compatible con to_sft.py (query/response).

Uso:
  python scripts/build_qa_dataset.py --input data/state.clone.db --output data/kateto_qa.jsonl --dry-run --limit 100
  python scripts/build_qa_dataset.py --input data/state.clone.db --output data/kateto_qa.jsonl --mode referencias --limit 500
  # real con Orion (MoE CPU, parallel=1, timeout 600s, secuencial):
  OPENAI_BASE_URL=http://127.0.0.1:11434/v1 MODEL_NAME=Orion-26B python scripts/build_qa_dataset.py --input data/state.clone.db --output data/kateto_qa.jsonl --limit 500

Si Orion no responde (slot ocupado), el script deja modo --dry-run verificado y avisa.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Asegurar que kateto/tools está en path (Kateto repo)
KATETO_ROOT = Path.home() / "proyectos" / "OpenaiBuildWeek" / "Kateto"
if str(KATETO_ROOT) not in sys.path:
    sys.path.insert(0, str(KATETO_ROOT))

# also this repo root
TRAIN_ROOT = Path(__file__).resolve().parents[1]
os.chdir(TRAIN_ROOT)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extrae Q/A del clon state.clone.db con filtros secrets/IA/dedup y generación Orion-26B")
    p.add_argument("--input", dest="input", default="data/state.clone.db", help="clon DB (NUNCA la live ~/.hermes/state.db)")
    p.add_argument("--output", dest="output", default="data/kateto_qa.jsonl", help="jsonl salida {question, answer}")
    p.add_argument("--dry-run", action="store_true", help="no llama a Orion, genera placeholders [DRY-RUN] y verifica pipeline")
    p.add_argument("--limit", type=int, default=None, help="máx pares Q/A")
    p.add_argument("--mode", default="referencias", choices=["argento", "referencias"], help="modo generator: argento o referencias (default referencias respeta README triggers)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-chars", type=int, default=800)
    p.add_argument("--no-ai-filter", action="store_true")
    p.add_argument("--no-secrets-filter", action="store_true", help="NO recomendado: un leak es peor que perder un par")
    p.add_argument("--concurrency", type=int, default=4, help="4 para freellmapi remoto; 1 para Orion MoE CPU local")
    p.add_argument("--openai-model", default=None, help="default auto (freellmapi)")
    p.add_argument("--openai-base-url", default=None, help="default http://127.0.0.1:3001/v1 (freellmapi)")
    p.add_argument("--openai-api-key", default=None, help="default env OPENAI_API_KEY/FREELLMAPI_KEY u opencode.json")
    p.add_argument("--timeout", type=float, default=120, help="timeout por request (default 120s; 600 para Orion)")
    p.add_argument("--ref-rate", type=float, default=0.3, help="prob. de referencia cultural por mensaje (default 0.3, el resto argento plano)")
    p.add_argument("--crude-rate", type=float, default=0.25, help="prob. de boost de humor crudo por mensaje (default 0.25)")
    p.add_argument("--no-align-filter", action="store_true", help="desactiva filtro heurístico de alineación Q/A")
    p.add_argument("--judge", action="store_true", help="juez LLM 1-5 de alineación, descarta < umbral")
    p.add_argument("--judge-threshold", type=int, default=3)
    p.add_argument("--progress-file", default=None, help="json con progreso en tiempo real para tail (default <output>.progress.json)")
    p.add_argument("--progress-every", type=int, default=1, help="cada cuantos batches escribe progreso y .part (default 1)")
    p.add_argument("--onnx-model-path", default=None)
    p.add_argument("--no-vulkan", action="store_true")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv=None):
    import logging
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("build_qa")

    # Guardas clon
    inp = Path(args.input)
    if "state.clone.db" not in str(inp) and ".hermes/state.db" in str(inp):
        log.error("RECHAZADO: nunca tocar ~/.hermes/state.db live, solo el clon data/state.clone.db")
        sys.exit(2)
    if not inp.exists():
        log.error("input no existe: %s (hacer cp ~/.hermes/state.db data/state.clone.db)", inp)
        sys.exit(2)

    # Defaults freellmapi/auto; Orion-26B local queda como override por env/flags
    openai_model = args.openai_model or os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or "auto"
    openai_base_url = args.openai_base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLAMA_BASE_URL") or "http://127.0.0.1:3001/v1"
    # Forzar concurrency=1 si modelo es Orion local (MoE CPU se cuelga)
    if "orion" in openai_model.lower() and args.concurrency != 1:
        log.warning("Orion-26B es MoE CPU: forzando concurrency=1 (pediste %d)", args.concurrency)
        args.concurrency = 1

    # Verificar gateway
    if not args.dry_run:
        try:
            import httpx
            # probe rápido 5s
            r = httpx.get(openai_base_url.replace("/v1", "") + "/v1/models", timeout=5)
            # Alternatively check http://127.0.0.1:11434/v1/models
            log.info("gateway probe: %s -> %d", openai_base_url, r.status_code)
        except Exception as e:
            log.warning("gateway no responde en %s (%s) — seguís con --dry-run si está ocupado", openai_base_url, e)
            log.warning("Sugerencia: probá curl http://127.0.0.1:11434/v1/models y luego reintentá sin --dry-run")
            # No abortar: seguimos, pipeline manejará None y dejará placeholders; pero avisamos
            # Si falla todo, usuario debe usar --dry-run

    # Importar pipeline
    try:
        from kateto.tools.dataset_filter.classifier import MmBertQAClassifier
        from kateto.tools.dataset_filter.pipeline import run_pipeline
    except ImportError as e:
        log.error("no se pudo importar dataset_filter: %s — asegurá que Kateto repo está en %s", e, KATETO_ROOT)
        sys.exit(1)

    onnx_path = args.onnx_model_path or os.getenv("MMBERT_ONNX_PATH") or os.getenv("ONNX_MODEL_PATH")
    clf = MmBertQAClassifier(onnx_path=onnx_path, use_vulkan=not args.no_vulkan)
    if clf.is_onnx:
        log.info("classifier ONNX activo")
    else:
        log.warning("classifier heurístico (sin ONNX) — ok para dry-run; para mejor calidad: pip install onnxruntime tokenizers huggingface-hub")

    # Monkey-patch timeout si el pipeline lo respeta via env
    # generator.generate_complement usa timeout param; pipeline no lo expone → seteamos env para compat
    os.environ["ORION_TIMEOUT"] = str(int(args.timeout))

    # Dataset mode: referencias respeta README triggers/golden rules
    dataset_mode = args.mode
    log.info("mode=%s model=%s base=%s concurrency=%d dry_run=%s limit=%s", dataset_mode, openai_model, openai_base_url, args.concurrency, args.dry_run, args.limit)

    # Patch generator: timeout + ref_rate (pipeline no los expone, van por wrapper)
    try:
        import kateto.tools.dataset_filter.generator as gen
        _orig = gen.generate_complement

        async def _patched(data, label, *, model=None, base_url=None, api_key=None, timeout=60.0, mode=None):
            mdl = model or openai_model
            t = args.timeout if "orion" in mdl.lower() else timeout
            return await _orig(data, label, model=mdl, base_url=base_url or openai_base_url, api_key=api_key or args.openai_api_key, timeout=t, mode=mode or dataset_mode, ref_rate=args.ref_rate, crude_rate=args.crude_rate)

        gen.generate_complement = _patched
        log.info("patched generator timeout=%.0fs ref_rate=%.2f crude_rate=%.2f", args.timeout, args.ref_rate, args.crude_rate)
    except Exception as e:
        log.warning("no se pudo patchear timeout: %s", e)

    from kateto.tools.dataset_filter import pipeline as _pl
    _prog = args.progress_file or (str(Path(args.output)) + ".progress.json")
    stats = asyncio.run(run_pipeline(
        input_path=inp,
        output_path=Path(args.output),
        classifier=clf,
        limit=args.limit,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        filter_ai=not args.no_ai_filter,
        filter_secrets=not args.no_secrets_filter,
        dataset_mode=dataset_mode,
        generate_missing=True,
        max_chars=args.max_chars,
        openai_model=openai_model,
        openai_base_url=openai_base_url,
        openai_api_key=args.openai_api_key,
        concurrency=args.concurrency,
        ref_rate=args.ref_rate,
        crude_rate=args.crude_rate,
        align_filter=not args.no_align_filter,
        judge=args.judge,
        judge_threshold=args.judge_threshold,
        progress_file=_prog,
        progress_every=args.progress_every,
    ))
    log.info("progreso en vivo: tail -f %s + log batches", _prog)
    log.info("stats: %s", stats)
    # Si dry-run, pipeline no escribió archivo por diseño; lo materializamos con placeholders sin re-ejecutar pipeline completo
    out_p = Path(args.output)
    if args.dry_run and not out_p.exists() and stats.get("pairs", 0) > 0:
        import json as _json
        # El pipeline dry-run ya calculó stats pero no escribió; generamos file placeholder
        # leyendo textos filtrados directamente (sin LLM) para no duplicar costo de classifier
        # Simplificación: escribir N placeholders determinísticos respetando formato to_sft
        log.info("dry-run: materializando %d placeholders -> %s", stats["pairs"], out_p)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            for i in range(stats["pairs"]):
                # placeholder que respeta golden rules split bilingüe para no romper to_sft
                if i % 2 == 0:
                    q = f"[DRY-RUN {i}] ¿che, cómo ves esto? pregunta placeholder"
                    a = "[DRY-RUN] respuesta placeholder rioplatense — Dry run, Orion no llamado. Trigger test: Machine... appetizer."
                else:
                    q = "[DRY-RUN] pregunta placeholder sobre review"
                    a = f"[DRY-RUN {i}] Mirá, posta que no te va a gustar, chau. placeholder seco"
                rec = {"question": q, "answer": a, "source_label": "dry-run", "dry_run": True}
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        log.info("dry-run file materializado: %s", out_p)
    # Resumen + verificación to_sft compatible
    if not args.dry_run and out_p.exists():
        import json
        n_q = n_a = 0
        with out_p.open() as f:
            for line in f:
                try:
                    p = json.loads(line)
                    if p.get("question"): n_q += 1
                    if p.get("answer"): n_a += 1
                except: pass
        log.info("verificado to_sft compatible: %d líneas, question=%d answer=%d -> probá: python to_sft.py %s /tmp/sft_out", stats["pairs"], n_q, n_a, args.output)
        if stats["pairs"] == 0:
            log.warning("0 pares generados — posible Orion ocupado/slot trabado. Reintentá con --dry-run verificado o esperá 600s y probá de nuevo con concurrency=1")
    else:
        log.info("dry-run verificado: %d pares (placeholders) — pipeline de filtros OK, Orion no llamado. Para real: sacá --dry-run con Orion libre", stats["pairs"])

    print(f"\nDone. input={stats['input_texts']} chunks={stats['chunks']} kept={stats['kept_after_ai']} dropped_secrets={stats['dropped_secrets']} dropped_ai={stats['dropped_ai']} dropped_dup={stats.get('dropped_dup',0)} pairs={stats['pairs']} dry_run={stats['dry_run']}")
    if not args.dry_run and stats["pairs"] == 0:
        print("AVISO: Orion no respondió o slot ocupado — dejé script listo con modo --dry-run verificado. Probá --dry-run para validar filtros y reintentá real cuando Orion esté libre (parallel=1, timeout 600s).")


if __name__ == "__main__":
    main()
