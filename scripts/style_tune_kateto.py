#!/usr/bin/env python3
"""Style tune de Kateto: adaptar el tensor de estilo (S_0) sin tocar los pesos.

Que es "el tensor de estilo": en RWKV-7 el estado inicial de la recurrencia,
`blocks.{i}.att.time_state`, un tensor `(n_head, head_size, head_size)` float32
por capa (`RWKV-PEFT/rwkvt/rwkv7/att.py:191`). Entrenar SOLO eso se llama
*state tuning*: el 100% de los pesos queda congelado y lo que se mueve es el
arranque del modelo, que es donde vive el registro de habla (ritmo, tics,
voseo) sin tocar el conocimiento.

Los pesos NO se pueden derivar del estado y no hace falta: `merge/merge_state.py`
esta verificado y documentado — escribir `w[k] = w_state[k]` sobre
`time_state` mete 24 tensores muertos que `strict=False` ignora en silencio. El
state no es una matriz de pesos; es estado de RUNTIME. Por eso este modulo
*agrega* el tensor de estilo a un checkpoint, no lo mergea.

Los tres subcomandos:

    info  <ckpt.pth>                                  que tensor de estilo tiene
    init  <ckpt.pth> --out <salida.pth>               lo inicializa (en ceros)
    apply <ckpt.pth> --head <head.pth> --out <salida.pth>   lo aplica

`apply` consume el .pth que deja `--peft state` en RWKV-PEFT: con ese flag
`train_callback` guarda SOLO las claves con 'state' en el nombre
(`RWKV-PEFT/rwkvt/lightning_train/trainer.py:170-180`), o sea un head slim de
n tensors y nada mas. `apply` lo cose sobre el base y escribe un unico .pth que
el camino nativo ya sabe leer (`rwkv_pipeline/infer_kateto.py:408` busca
exactamente `blocks.{i}.att.time_state`).

Nada de esto entrena: no hay paso de optimizador aca. El que entrena es
`RWKV-PEFT/train.py`; ver docs/style-tune.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# rwkvt/rwkv7/att.py:191 -> nn.Parameter(torch.zeros(n_head, head_size, head_size))
STYLE_SUFFIX = ".att.time_state"

# TrainingArgs.head_size_a / train.py --head_size_a. Con n_embd=1024, 24 capas
# esto da 1.57M parametros, el "1.6M" que dice HANDOFF.md:56.
DEFAULT_HEAD_SIZE = 64

# Mismo codigo que usa rwkv_pipeline/state_from_pth.py:265 para "no calza".
RC_ERROR = 3


class StyleHeadError(ValueError):
    """El ckpt no tiene tensor de estilo, o el que trae no calza con el modelo."""


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover
        raise StyleHeadError(
            "falta torch: el style tune lee y escribe .pth, que es formato torch"
        ) from e
    return torch


# ------------------------------------------------------------------ el .pth


def cargar(path) -> dict:
    """Un .pth -> dict de tensores, en cpu. Falla con mensaje, no con traceback."""
    torch = _torch()
    p = Path(path)
    if not p.is_file():
        raise StyleHeadError(f"no existe el checkpoint: {p}")
    try:
        w = torch.load(str(p), map_location="cpu", weights_only=True)
    except Exception as e:
        raise StyleHeadError(f"{p}: no se pudo leer como .pth de torch ({e})")
    if not isinstance(w, dict):
        raise StyleHeadError(f"{p}: no es un checkpoint (viene {type(w).__name__}, "
                             f"se esperaba un dict de tensores)")
    return w


def claves_estilo(w: dict) -> list[str]:
    """Las claves del tensor de estilo, ordenadas por capa."""
    return sorted(
        (k for k in w if k.endswith(STYLE_SUFFIX)),
        key=lambda k: int(k.split(".")[1]),
    )


# ------------------------------------------------------------- el reporte


def reporte(w: dict) -> dict:
    """Forma y parametros del tensor de estilo de `w`. Sin estilo -> StyleHeadError."""
    claves = claves_estilo(w)
    if not claves:
        raise StyleHeadError(
            f"este checkpoint no tiene tensor de estilo (ninguna clave "
            f"{STYLE_SUFFIX!r}); son solo pesos. Para agregarlo: "
            f"'init <ckpt> --out <salida.pth>'"
        )
    formas = {tuple(w[k].shape) for k in claves}
    params = sum(int(w[k].numel()) for k in claves)
    bytes_ = sum(int(w[k].numel()) * int(w[k].element_size()) for k in claves)
    return {
        "tensores": len(claves),
        "capas": [int(k.split(".")[1]) for k in claves],
        "forma": formas.pop() if len(formas) == 1 else None,
        "formas_distintas": sorted(formas),
        "params": params,
        "dtype": str(w[claves[0]].dtype),
        "bytes": bytes_,
    }


def imprimir_reporte(path, rep: dict, extra: str = "") -> None:
    """`clave: valor` por linea, una sola vez para info y para init/apply."""
    print(f"checkpoint: {path}")
    print(f"tensor de estilo: blocks{{i}}{STYLE_SUFFIX}")
    print(f"tensores: {rep['tensores']}")
    print(f"capas: {rep['capas']}")
    if rep["forma"] is None:
        print(f"forma: DISTINTA POR CAPA {rep['formas_distintas']}")
    else:
        print(f"forma: {rep['forma']}")
    print(f"params: {rep['params']}")
    print(f"dtype: {rep['dtype']}")
    print(f"bytes: {rep['bytes']}")
    if extra:
        print(extra)


# ------------------------------------------------------------- la geometria


def geometria(w: dict, head_size: int) -> tuple[int, tuple[int, int, int]]:
    """`(n_layer, (n_head, head_size, head_size))` deducido de los pesos del ckpt.

    n_layer sale de las claves `blocks.{i}.` y n_embd de `emb.weight`; head_size
    es un hyperparametro del modelo (no se deduce de los pesos), va por flag.
    """
    capas = sorted({int(k.split(".")[1]) for k in w if k.startswith("blocks.") and ".att." in k})
    if not capas:
        raise StyleHeadError(
            "no parece un checkpoint de RWKV: no hay claves blocks.{i}.att.*"
        )
    if "emb.weight" not in w:
        raise StyleHeadError("falta emb.weight: no se puede deducir n_embd")
    n_embd = int(w["emb.weight"].shape[1])  # nn.Embedding(vocab_size, n_embd)
    if n_embd % head_size:
        raise StyleHeadError(f"n_embd={n_embd} no es multiplo de head_size={head_size}")
    return max(capas) + 1, (n_embd // head_size, head_size, head_size)


# ------------------------------------------------------------- subcomandos


def cmd_info(args) -> int:
    w = cargar(args.ckpt)
    imprimir_reporte(args.ckpt, reporte(w))
    return 0


def cmd_init(args) -> int:
    torch = _torch()
    w = cargar(args.ckpt)

    if claves_estilo(w) and not args.force:
        raise StyleHeadError(
            f"{args.ckpt} ya tiene tensor de estilo en las capas "
            f"{[int(k.split('.')[1]) for k in claves_estilo(w)]}; sobreescribirlo "
            f"tira el estilo entrenado. Con --force si."
        )

    n_layer, forma = geometria(w, args.head_size)
    for i in range(n_layer):
        w[f"blocks.{i}{STYLE_SUFFIX}"] = torch.zeros(forma, dtype=torch.float32)

    out = Path(args.out)
    torch.save(w, str(out))
    imprimir_reporte(
        out,
        reporte(w),
        extra=f"inicializado: {n_layer} tensores {forma} en ceros "
              f"(n_layer={n_layer} deducido de las claves blocks.*)",
    )
    return 0


def cmd_apply(args) -> int:
    torch = _torch()
    w = cargar(args.ckpt)
    head = cargar(args.head)

    claves = claves_estilo(head)
    if not claves:
        raise StyleHeadError(
            f"{args.head} no tiene tensor de estilo (ninguna clave "
            f"{STYLE_SUFFIX!r}); no hay nada que aplicar"
        )

    # Se chequea contra la geometria del base y no solo contra las claves que ya
    # estan: agregar una forma cualquiera deja un .pth que el runtime no puede
    # usar, el mismo "no se salva inventando tensores" de merge_state.py.
    n_layer, forma = geometria(w, args.head_size)
    choques = []
    for k in claves:
        capa, forma_head = int(k.split(".")[1]), tuple(head[k].shape)
        if capa >= n_layer:
            choques.append(f"  {k}: capa {capa} >= n_layer {n_layer} del base")
        elif forma_head != forma:
            choques.append(f"  {k}: head {forma_head} != {forma} del base")
        elif k in w and tuple(w[k].shape) != forma_head:
            choques.append(f"  {k}: el base ya tiene {tuple(w[k].shape)} != head {forma_head}")
    if choques:
        raise StyleHeadError(
            f"el head no calza con {args.ckpt} en {len(choques)} capa(s):\n"
            + "\n".join(choques)
        )

    aplicadas = sum(1 for k in claves if k in w)
    for k in claves:
        w[k] = head[k].detach().clone()
    agregadas = len(claves) - aplicadas

    out = Path(args.out)
    torch.save(w, str(out))
    imprimir_reporte(
        out,
        reporte(w),
        extra=f"aplicadas: {aplicadas}   agregadas (capa nueva): {agregadas}   "
              f"pesos base: {len(w) - len(claves)} sin tocar",
    )
    return 0


# ------------------------------------------------------------------- el CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="style_tune_kateto.py",
        description="style tune (tensor de estilo / state tuning) sin tocar los pesos",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_info = sub.add_parser("info", help="que tensor de estilo tiene el ckpt")
    p_info.add_argument("ckpt")

    p_init = sub.add_parser("init", help="inicializa el tensor de estilo sobre el ckpt")
    p_init.add_argument("ckpt")
    p_init.add_argument("--out", required=True)
    p_init.add_argument("--head-size", type=int, default=DEFAULT_HEAD_SIZE)
    p_init.add_argument("--force", action="store_true",
                        help="sobreescribe un tensor de estilo ya presente")

    p_apply = sub.add_parser("apply", help="aplica un head de estilo y guarda el .pth nuevo")
    p_apply.add_argument("ckpt")
    p_apply.add_argument("--head", required=True, help="head slim de style tuning")
    p_apply.add_argument("--out", required=True)
    p_apply.add_argument("--head-size", type=int, default=DEFAULT_HEAD_SIZE,
                         help="el del modelo; contra esto se valida el head")

    args = ap.parse_args(argv)
    try:
        return {"info": cmd_info, "init": cmd_init, "apply": cmd_apply}[args.cmd](args)
    except StyleHeadError as e:
        print(f"error: {e}", file=sys.stderr)
        return RC_ERROR


if __name__ == "__main__":
    sys.exit(main())