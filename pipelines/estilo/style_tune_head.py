#!/usr/bin/env python3
"""Style tune de UN SOLO TENSOR para el RWKV de Kateto.

La tecnica es de Gryphe (Gemma-4-26B-A4B-StyleTune-V2): congelar TODO el modelo
y entrenar unicamente la proyeccion de salida. En Kateto eso es `head.weight`:
1 tensor de 798. Su V1 fue mas de un epoch y le arruino la estabilidad; el V2 es
1 epoch. Aca tambien 1 epoch, y `--dry-run` son 2 pasos de prueba.

Que se verifica de verdad (no es decorativo)
-------------------------------------------
* se congela todo y se ASSERTA que hay exactamente UN tensor con requires_grad,
  y que es el head;
* se ASSERTA que despues del backward solo el head tiene `.grad`;
* se ASSERTA que los hidden states del tronco son IDENTICOS antes y despues del
  paso del optimizador: el modelo no se toco, solo la voz. Es la frase de Gryphe
  ("The model hasn't changed. Only the voice has.") convertida en assert.

Atajo matematico: con el tronco congelado sus hidden states son constantes, asi
que no hace falta backpropagar a traves de el — se calcula una vez por tanda bajo
`no_grad` y se entrena el head sobre esas features cacheadas. El gradiente que
llega a `head.weight` es EXACTAMENTE el mismo que en un fine-tune con todo el
modelo delante. Con el tronco entero son 450 834 432 parametros y el head son
67 108 864, esto baja la memoria de la senal de gradiente de 450M a 67M.

    # prueba: 2 pasos, CPU, tanda chica. Sale 0.
    env -u PYTHONPATH -u VIRTUAL_ENV \
      /run/media/chaos/terciario/proyectos/kateto-train/venv/bin/python \
      pipelines/estilo/style_tune_head.py --dry-run

    # 1 epoch real sobre el corpus purgado (no correr en esta caja: es GPU)
    ... pipelines/estilo/style_tune_head.py --entrenar --lote 8 --ctx 512
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_PRINCIPAL = Path("/run/media/chaos/terciario/proyectos/kateto-train")

# Donde se busca el .pth de Kateto, en orden de preferencia. Primero el chico
# (0.4B): entra en RAM de CPU y es el que se puede probar de verdad aca.
CANDIDATOS_CKPT = [
    REPO_PRINCIPAL / "out/rwkv_kateto_0.4b_ctx256k/rwkv-1.pth",
    REPO_PRINCIPAL / "out/rwkv_kateto_base/rwkv-1.pth",
    REPO_PRINCIPAL / "out/rwkv_kateto_base/rwkv-0.pth",
    REPO_PRINCIPAL / "base/rwkv7-g1j-2.9b-20260831-ctx16384.pth",
    REPO_PRINCIPAL / "out/kateto-rwkv29-merged.pth",
]
CANDIDATOS_VOCAB = [
    REPO_PRINCIPAL / "RWKV-PEFT/rwkv_vocab_v20230424.txt",
    REPO_PRINCIPAL / "RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt",
]
CORPUS = Path("/home/chaos/harness-run/kateto-purge/rwkv_kateto_base_train_plus.purged.jsonl")
SALIDA = Path("/home/chaos/harness-run/kateto-estilo")


# --- checkpoint ------------------------------------------------------------


def buscar(creados: list[Path]) -> tuple[Path | None, list[Path]]:
    """Primer candidato que existe. Devuelve (encontrado, todos los buscados)."""
    for p in creados:
        if p.exists():
            return p, creados
    return None, creados


def config_de_checkpoint(sd: dict) -> dict:
    n_layer = sum(1 for k in sd if k.endswith("att.r_k"))
    n_head, head_size = sd["blocks.0.att.r_k"].shape
    return {
        "n_layer": n_layer,
        "n_embd": int(sd["emb.weight"].shape[1]),
        "n_head": int(n_head),
        "head_size": int(head_size),
        "vocab": int(sd["emb.weight"].shape[0]),
        "d_w": int(sd["blocks.0.att.w1"].shape[1]),
        "d_v": int(sd["blocks.0.att.v1"].shape[1]),
        "d_g": int(sd["blocks.0.att.g1"].shape[1]),
        "ffn": int(sd["blocks.0.ffn.key.weight"].shape[0]),
    }


def preparar_pesos(sd: dict) -> dict:
    """Pesa el checkpoint a fp32 con el preprocesado que hace infer_kateto.py.

    Mismas tres cosas que hace `RWKV_RNN.__init__`: `att.w0` en float32,
    squeeze de todos los tensores (vienen con dim (1,1,C)), `att.r_k` plano, y
    `ln0` plegado adentro de `emb.weight`. `blocks.0.att.v{0,1,2}` se copian
    de `a{0,1,2}` porque son el mismo tensor en el checkpoint.
    """
    z = {k: v.float() for k, v in sd.items()}  # att.w0 tambien: el estado corre en fp32
    for k in z:
        z[k] = z[k].squeeze()
        if k.endswith("att.r_k"):
            z[k] = z[k].flatten()
    z["emb.weight"] = torch.layer_norm(
        z["emb.weight"],
        (z["emb.weight"].shape[1],),
        weight=z["blocks.0.ln0.weight"],
        bias=z["blocks.0.ln0.bias"],
    )
    z["blocks.0.att.v0"] = z["blocks.0.att.a0"]
    z["blocks.0.att.v1"] = z["blocks.0.att.a1"]
    z["blocks.0.att.v2"] = z["blocks.0.att.a2"]
    return z


def config_mini() -> dict:
    """RWKV-7 en miniatura. Solo para probar el mecanismo de congelado.

    Las proporciones internas (d_w = 4*n_head, d_v = 2*n_head, d_g = 8*n_head,
    ffn = 4*n_embd) estan medidas sobre el checkpoint real de 0.4B, no inventadas.
    """
    n_head, n_embd = 4, 128
    return {
        "n_layer": 2,
        "n_embd": n_embd,
        "n_head": n_head,
        "head_size": n_embd // n_head,
        "vocab": 65536,
        "d_w": 4 * n_head,
        "d_v": 2 * n_head,
        "d_g": 8 * n_head,
        "ffn": 4 * n_embd,
    }


def pesos_mini(cfg: dict) -> dict:
    """Pesos al azar con la misma forma que los reales. NO es un modelo trained."""
    C, H, cfg_ = cfg["n_embd"], cfg["n_head"], cfg
    # C = n_embd, H = n_head, cfg_ = la config entera
    g = torch.Generator().manual_seed(0)

    def p(*shape: int, escala: float = 0.02) -> torch.Tensor:
        return (torch.randn(*shape, generator=g) * escala)

    z: dict[str, torch.Tensor] = {
        "emb.weight": p(cfg_["vocab"], C, escala=0.05),
        "ln_out.weight": torch.ones(C),
        "ln_out.bias": torch.zeros(C),
    }
    for i in range(cfg_["n_layer"]):
        b, a, f = f"blocks.{i}.", f"blocks.{i}.att.", f"blocks.{i}.ffn."
        z[b + "ln0.weight"], z[b + "ln0.bias"] = torch.ones(C), torch.zeros(C)
        z[b + "ln1.weight"], z[b + "ln1.bias"] = torch.ones(C), torch.zeros(C)
        z[b + "ln2.weight"], z[b + "ln2.bias"] = torch.ones(C), torch.zeros(C)
        for x in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "k_k", "k_a"):
            z[a + x] = p(C)
        z[a + "r_k"] = p(H, cfg_["head_size"]).flatten()  # el checkpoint lo guarda plano
        for x in ("w1", "a1"):
            z[a + x] = p(C, cfg_["d_w"])
        for x in ("w2", "a2"):
            z[a + x] = p(cfg_["d_w"], C)
        z[a + "v0"] = p(C)
        z[a + "v1"], z[a + "v2"] = p(C, cfg_["d_v"]), p(cfg_["d_v"], C)
        z[a + "g1"], z[a + "g2"] = p(C, cfg_["d_g"]), p(cfg_["d_g"], C)
        for x in ("receptance", "key", "value", "output"):
            z[a + f"{x}.weight"] = p(C, C)
        z[a + "ln_x.weight"], z[a + "ln_x.bias"] = torch.ones(C), torch.zeros(C)
        z[f + "x_k"] = p(C)
        z[f + "key.weight"] = p(cfg_["ffn"], C)
        z[f + "value.weight"] = p(C, cfg_["ffn"])
    z["head.weight"] = p(cfg_["vocab"], C, escala=0.05)
    return z


# --- tronco RWKV-7 -----------------------------------------------------------
# Operadores importados de la inferencia real del repo, no reimplementados: es la
# unica fuente de verdad de la matematica del modelo y evita que se desincronice.

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from rwkv_pipeline.infer_kateto import (  # noqa: E402
    RWKV_TOKENIZER,
    channel_mixing,
    time_mixing,
)


@torch.no_grad()
def hidden_states(pesos: dict, cfg: dict, tokens, state: list) -> torch.Tensor:
    """[T, n_embd] — el estado que ve el head. Tronco entero, sin gradiente."""
    H, N, C, L = cfg["n_head"], cfg["head_size"], cfg["n_embd"], cfg["n_layer"]
    salida = []
    for tok in tokens:
        x = pesos["emb.weight"][tok]
        v_first = torch.zeros_like(x)
        for i in range(L):
            b, a, f = f"blocks.{i}.", f"blocks.{i}.att.", f"blocks.{i}.ffn."
            xx = F.layer_norm(x, (C,), weight=pesos[b + "ln1.weight"], bias=pesos[b + "ln1.bias"])
            xx, state[i * 3 + 0], state[i * 3 + 1], v_first = time_mixing(
                i, H, N, xx, state[i * 3 + 0], v_first, state[i * 3 + 1],
                pesos[a + "x_r"], pesos[a + "x_w"], pesos[a + "x_k"], pesos[a + "x_v"],
                pesos[a + "x_a"], pesos[a + "x_g"], pesos[a + "w0"], pesos[a + "w1"],
                pesos[a + "w2"], pesos[a + "a0"], pesos[a + "a1"], pesos[a + "a2"],
                pesos[a + "v0"], pesos[a + "v1"], pesos[a + "v2"], pesos[a + "g1"],
                pesos[a + "g2"], pesos[a + "k_k"], pesos[a + "k_a"], pesos[a + "r_k"],
                pesos[a + "key.weight"], pesos[a + "value.weight"],
                pesos[a + "receptance.weight"], pesos[a + "output.weight"],
                pesos[a + "ln_x.weight"], pesos[a + "ln_x.bias"],
            )
            x = x + xx
            xx = F.layer_norm(x, (C,), weight=pesos[b + "ln2.weight"], bias=pesos[b + "ln2.bias"])
            xx, state[i * 3 + 2] = channel_mixing(
                xx, state[i * 3 + 2], pesos[f + "x_k"],
                pesos[f + "key.weight"], pesos[f + "value.weight"],
            )
            x = x + xx
        x = F.layer_norm(x, (C,), weight=pesos["ln_out.weight"], bias=pesos["ln_out.bias"])
        salida.append(x)
    return torch.stack(salida)


def estado_inicial(cfg: dict) -> list:
    H, N, C, L = cfg["n_head"], cfg["head_size"], cfg["n_embd"], cfg["n_layer"]
    st: list = []
    for _ in range(L):
        st += [torch.zeros(C), torch.zeros(H, N, N), torch.zeros(C)]
    return st


# --- el modelo: tronco buffer + head parametrado ---------------------------


class SoloHead(torch.nn.Module):
    """Trono congelado en buffers + `head` como unico parametro entrenable.

    Los buffers no son `nn.Parameter`, asi que `parameters()` solo ve el head y
    el assert de "un unico tensor entrenable" no depende de acordarse de
    llamar a requires_grad_(False) en 798 tensors.
    """

    def __init__(self, pesos: dict, cfg: dict):
        super().__init__()
        self.cfg = cfg
        # Un dict liso, no `Parameter`/`register_buffer`: los tensores de RWKV se
        # llaman `blocks.0.att.w0` y register_buffer rechaza nombres con punto.
        self.tronco = {k: v.detach().requires_grad_(False)
                       for k, v in pesos.items() if k != "head.weight"}
        self.head = torch.nn.Linear(cfg["n_embd"], cfg["vocab"], bias=False)
        with torch.no_grad():
            self.head.weight.copy_(pesos["head.weight"])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)


def congelar(modelo: SoloHead) -> dict:
    """Congela y ASSERTA el invariante central: 1 tensor entrenable, el head."""
    for p in modelo.parameters():
        p.requires_grad_(False)
    modelo.head.weight.requires_grad_(True)

    con_grad = [n for n, p in modelo.named_parameters() if p.requires_grad]
    assert len(con_grad) == 1, f"se esperaba 1 tensor entrenable, hay {len(con_grad)}: {con_grad}"
    assert con_grad[0] == "head.weight", f"el tensor entrenable no es el head: {con_grad[0]}"
    cfg = modelo.cfg
    esperado = cfg["vocab"] * cfg["n_embd"]
    assert modelo.head.weight.numel() == esperado, "el head no tiene la forma del checkpoint"
    assert not any(v.requires_grad for v in modelo.tronco.values()), "un tensor del tronco quedo.trainable"

    frozen = sum(v.numel() for v in modelo.tronco.values())
    total = frozen + modelo.head.weight.numel()
    return {
        "tensores_entrenables": len(con_grad),
        "params_entrenables": modelo.head.weight.numel(),
        "params_congelados": frozen,
        "params_totales": total,
        "fraccion_entrenable": round(modelo.head.weight.numel() / total, 6),
    }


def auditar(modelo: SoloHead, informe: dict, tensores: int) -> None:
    print("\n=== PARAMETROS ===")
    print(f"  tensores entrenables : {informe['tensores_entrenables']} de {tensores}")
    print(f"  params entrenables   : {informe['params_entrenables']:,}")
    print(f"  params congelados     : {informe['params_congelados']:,}")
    print(f"  params totales        : {informe['params_totales']:,}")
    print(f"  fraccion entrenable   : {informe['fraccion_entrenable'] * 100:.2f}%")
    print("  tensor entrenable     : head.weight  (proyeccion de salida)")


def assert_solo_head_recibio_grad(modelo: SoloHead) -> None:
    """Despues del backward: el unico `.grad` es el del head. Pruebo, no promesa."""
    con_grad = [n for n, p in modelo.named_parameters() if p.grad is not None]
    assert con_grad == ["head.weight"], f"hubo gradiente fuera del head: {con_grad}"
    g = modelo.head.weight.grad
    assert g is not None and torch.isfinite(g).all(), "gradiente del head nulo o no finito"
    assert g.abs().max() > 0, "gradiente del head es todo cero: no esta aprendiendo"


# --- corpus ----------------------------------------------------------------


def cargar_tokenizador(vocab: Path):
    return RWKV_TOKENIZER(str(vocab))


def chunks_de_documentos(corpus: Path, tok, ctx: int, limite: int | None) -> list:
    """Trocea el corpus purgado en ventanas de `ctx` tokens."""
    out: list = []
    with corpus.open(encoding="utf-8") as fh:
        for n, linea in enumerate(fh):
            if limite is not None and n >= limite:
                break
            linea = linea.strip()
            if not linea:
                continue
            ids = tok.encode(json.loads(linea)["text"])
            for i in range(0, len(ids) - 1, ctx):
                w = ids[i : i + ctx + 1]
                if len(w) == ctx + 1:
                    out.append(w)
    return out


# --- corrida ---------------------------------------------------------------


def correr(args) -> int:
    torch.manual_seed(args.semilla)
    # El techo de memoria es duro (cgroup ~3 GB: ya mato un run con exit -15) y el
    # checkpoint mas chico son 900 MB en disco, ~1.8 GB al pasarlo a fp32. El dry-run
    # busca para reportar la ruta, pero jamas lo carga.
    hilos = 2 if args.dry_run else (args.hilos or 4)
    torch.set_num_threads(hilos)

    print("=== STYLE TUNE DE UN SOLO TENSOR (head) ===")
    if args.dry_run:
        pedidos = [Path(args.ckpt)] if args.ckpt and args.ckpt != "mini" else CANDIDATOS_CKPT
        encontrado, buscados = buscar(pedidos)
        if encontrado is None:
            print("[ckpt] no encontre checkpoint de Kateto en el disco. Busque:")
        else:
            print(f"[ckpt] hay checkpoint de Kateto en el disco: {encontrado} "
                  f"({encontrado.stat().st_size / 1e9:.2f} GB)")
        for p in buscados:
            print(f"          {p}  ->  {'existe' if p.exists() else 'NO EXISTE'}")
        print("[ckpt] dry-run: NO lo cargo (techo de memoria). Uso un RWKV-7 en")
        print("       miniatura de pesos al azar: prueba que el congelado deja UN")
        print("       solo tensor entrenable, y nada mas.")
        pesos, cfg, ckpt = pesos_mini(config_mini()), config_mini(), None
    elif args.ckpt == "mini":
        print("[ckpt] mini pedido a mano: RWKV-7 en miniatura de pesos al azar")
        pesos, cfg, ckpt = pesos_mini(config_mini()), config_mini(), None
    else:
        pedidos = [Path(args.ckpt)] if args.ckpt else CANDIDATOS_CKPT
        ckpt, buscados = buscar(pedidos)
        if ckpt is None:
            print("[error] no encontre checkpoint de Kateto en el disco. Busque:")
            for p in buscados:
                print(f"          {p}  ->  {'existe' if p.exists() else 'NO EXISTE'}")
            print("[error] sin checkpoint no hay nada que entrenar. Pasá --ckpt <ruta>",
                  file=sys.stderr)
            return 3
        else:
            print(f"[ckpt] {ckpt}  ({ckpt.stat().st_size / 1e9:.2f} GB)")
            sd = torch.load(str(ckpt), map_location="cpu", mmap=True, weights_only=True)
            cfg = config_de_checkpoint(sd)
            pesos = preparar_pesos(sd)
            del sd
    print(f"[cfg] n_layer={cfg['n_layer']} n_embd={cfg['n_embd']} n_head={cfg['n_head']} "
          f"vocab={cfg['vocab']} head_size={cfg['head_size']}")

    vocab, _ = buscar(CANDIDATOS_VOCAB)
    if vocab is None:
        print(f"[error] no encontre el tokenizer RWKV; busca en: {[str(v) for v in CANDIDATOS_VOCAB]}",
              file=sys.stderr)
        return 3
    tok = cargar_tokenizador(vocab)

    t0 = time.time()
    modelo = SoloHead(pesos, cfg)
    tensores = len(pesos) - 1  # el head del checkpoint entra como parametro del modulo
    informe = congelar(modelo)
    auditar(modelo, informe, tensores)
    print(f"[congelado] {time.time() - t0:.1f}s")

    pasos = 2 if args.dry_run else (args.pasos or 0)
    ctx, lote, docs = args.ctx, args.lote, args.docs
    print(f"[datos] ctx={ctx} lote={lote} docs={docs} pasos={'2 (dry-run)' if args.dry_run else pasos or '1 epoch'}")

    blancos = chunks_de_documentos(CORPUS, tok, ctx, docs)
    if len(blancos) < lote:
        print(f"[error] el corpus dio {len(blancos)} chunks y el lote pide {lote}", file=sys.stderr)
        return 4
    random.Random(args.semilla).shuffle(blancos)
    if not pasos:
        blancos = blancos[: lote * 1000]  # 1 epoch: todo lo que entro en memoria
        pasos = (len(blancos) + lote - 1) // lote
        print(f"[epoch] {len(blancos)} chunks -> {pasos} pasos")

    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.0, foreach=False)
    modelo.train()
    losses: list[float] = []
    control = blancos[0][: ctx + 1]
    st_ctrl = estado_inicial(cfg)
    h_antes = hidden_states(pesos, cfg, control, st_ctrl)
    head_antes = modelo.head.weight.detach().clone()

    inicio = time.time()
    for paso in range(pasos):
        hs, ys = [], []
        for w in blancos[paso * lote : paso * lote + lote]:
            hs.append(hidden_states(pesos, cfg, w[:ctx], estado_inicial(cfg)))
            ys.append(torch.tensor(w[1:]))
        h = torch.cat(hs)
        y = torch.cat(ys)

        logits = modelo(h)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert_solo_head_recibio_grad(modelo)
        opt.step()

        losses.append(loss.item())
        if not math.isfinite(losses[-1]):
            print(f"[error] perdida no finita en el paso {paso + 1}: {losses[-1]}", file=sys.stderr)
            return 5
        if not args.dry_run and (paso + 1) % 50 == 0:
            print(f"  paso {paso + 1}/{pasos} loss={losses[-1]:.4f}")

    dt = time.time() - inicio
    print(f"\n=== ENTRENAMIENTO ===")
    print(f"  pasos: {pasos}   tiempo: {dt:.1f}s   ({dt / max(pasos, 1):.2f}s/paso)")
    print(f"  loss inicial: {losses[0]:.4f}   loss final: {losses[-1]:.4f}   "
          f"delta: {losses[-1] - losses[0]:+.4f}")

    # "The model hasn't changed. Only the voice has."
    h_despues = hidden_states(pesos, cfg, control, estado_inicial(cfg))
    assert torch.equal(h_antes, h_despues), "el tronco cambio: el style tune toco el modelo"
    assert not torch.equal(head_antes, modelo.head.weight), "el head no se movio: no hubo entrenamiento"
    print("\n=== INVARIANTES ===")
    print("  ok  el tronco congelado dio hidden states IDENTICOS antes y despues")
    print("  ok  el head SI cambio (el unico tensor que se entrena)")
    print(f"  ok  el unico .grad es head.weight (norma {modelo.head.weight.grad.norm():.3e})")

    # --- guardar ---
    SALIDA.mkdir(parents=True, exist_ok=True)
    destino = SALIDA / (args.nombre or "head-estilo.pth")
    torch.save(
        {
            "head.weight": modelo.head.weight.detach().clone(),
            "__meta__": {
                "base": str(ckpt) if ckpt else "mini(rand)",
                "vocab": cfg["vocab"],
                "n_embd": cfg["n_embd"],
                "pasos": pasos,
                "loss_inicial": losses[0],
                "loss_final": losses[-1],
                "params_entrenables": informe["params_entrenables"],
                "fraccion_entrenable": informe["fraccion_entrenable"],
                "nota": "sumar este head al checkpoint base; el resto del modelo no se toco",
            },
        },
        destino,
    )
    (SALIDA / f"{destino.stem}.json").write_text(
        json.dumps(
            {
                "base": str(ckpt) if ckpt else "mini(rand)",
                "config": cfg,
                "informe": informe,
                "pasos": pasos,
                "losses": losses,
                "head": str(destino),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[guardado] {destino}  ({destino.stat().st_size / 1e6:.1f} MB)")
    print(f"[guardado] {SALIDA / (destino.stem + '.json')}")
    print(f"[params entrenables] {informe['params_entrenables']:,} "
          f"({informe['tensores_entrenables']} tensor, {informe['fraccion_entrenable'] * 100:.2f}% del modelo)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="style_tune_head.py",
        description=(
            "Style tune de un solo tensor: entrena SOLO head.weight (la proyeccion de "
            "salida) con el resto del RWKV de Kateto congelado. Tecnica de Gryphe "
            "(Gemma-4-26B-A4B-StyleTune-V2): 1 tensor, 1 epoch."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    modo = ap.add_mutually_exclusive_group()
    modo.add_argument("--dry-run", action="store_true",
                      help="2 pasos, tanda chica, CPU. Es lo que se puede correr aca.")
    modo.add_argument("--entrenar", action="store_true",
                      help="1 epoch sobre el corpus purgado. Requiere GPU en la practica.")
    ap.add_argument("--ckpt", default=None,
                    help=f"checkpoint .pth de Kateto, o 'mini' para pesos al azar. "
                         f"Default: busca el primero existente de {[str(c) for c in CANDIDATOS_CKPT]}")
    ap.add_argument("--pasos", type=int, default=0, help="pasos fijos (0 = 1 epoch)")
    ap.add_argument("--lote", type=int, default=2, help="chunks por paso")
    ap.add_argument("--ctx", type=int, default=64, help="tokens por chunk")
    ap.add_argument("--docs", type=int, default=2000, help="maximo de documentos a leer del corpus")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hilos", type=int, default=0, help="0 = 4")
    ap.add_argument("--semilla", type=int, default=1234)
    ap.add_argument("--nombre", default=None, help="nombre del .pth de salida")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.entrenar:
        args.dry_run = True  # el modo seguro es el default
    return correr(args)


if __name__ == "__main__":
    raise SystemExit(main())