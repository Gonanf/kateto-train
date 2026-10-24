#!/usr/bin/env python3
"""Aritmética exacta del presupuesto de parámetros para MoE-LoRA vs LoRA Dense.

Garantiza que el MoE no tenga ventaja de capacidad bruta sobre los baselines densos.
"""

import sys
import argparse

N_EMBD = 2048
DIM_FFN = 8192
N_LAYER = 24


def lora_layer_params(r: int) -> int:
    """Parámetros entrenables de LoRA (key + value) en la FFN de una capa."""
    return r * (N_EMBD + DIM_FFN) * 2


def lora_total_params(r: int) -> int:
    """Parámetros entrenables totales de LoRA en todas las capas."""
    return N_LAYER * lora_layer_params(r)


def router_layer_params(num_experts: int) -> int:
    """Parámetros entrenables del router en una capa: Linear(n_embd, num_experts - 1, bias=False)."""
    return N_EMBD * (num_experts - 1)


def router_total_params(num_experts: int) -> int:
    """Parámetros entrenables del router en todas las capas."""
    return N_LAYER * router_layer_params(num_experts)


def moe_total_params(num_experts: int, r: int) -> int:
    """Parámetros entrenables totales de MoE-LoRA (expertos LoRA + router)."""
    return num_experts * lora_total_params(r) + router_total_params(num_experts)


def print_table():
    lora_16 = lora_total_params(16)
    lora_32 = lora_total_params(32)
    lora_64 = lora_total_params(64)
    lora_128 = lora_total_params(128)

    moe_4_32 = moe_total_params(4, 32)
    moe_4_16 = moe_total_params(4, 16)
    router_4 = router_total_params(4)

    print("=" * 78)
    print("PRESUPUESTO DE PARÁMETROS ENTRENABLES (RWKV-7 1.5B: L=24, D=2048, FFN=8192)")
    print("=" * 78)
    print(f"{'Configuración':<32} | {'LoRA Params':<14} | {'Router':<10} | {'Total Params':<14}")
    print("-" * 78)
    print(f"{'LoRA dense r=16 (baseline v7)':<32} | {lora_16:<14,d} | {'0':<10} | {lora_16:<14,d}")
    print(f"{'LoRA dense r=32':<32} | {lora_32:<14,d} | {'0':<10} | {lora_32:<14,d}")
    print(f"{'LoRA dense r=64':<32} | {lora_64:<14,d} | {'0':<10} | {lora_64:<14,d}")
    print(f"{'LoRA dense r=128 (control honesto)':<32} | {lora_128:<14,d} | {'0':<10} | {lora_128:<14,d}")
    print("-" * 78)
    print(f"{'MoE E=4 active=2 r=16':<32} | {4 * lora_total_params(16):<14,d} | {router_4:<10,d} | {moe_4_16:<14,d}")
    print(f"{'MoE E=4 active=2 r=32':<32} | {4 * lora_total_params(32):<14,d} | {router_4:<10,d} | {moe_4_32:<14,d}")
    print("=" * 78)
    print(f"Control honesto para MoE E=4 r=32: LoRA dense r=128 ({lora_128:,d} params LoRA)")
    print(f"Diferencia: router MoE agrega {router_4:,d} params (+{router_4 / lora_128 * 100:.2f}% de sobrecosto router).")
    print("=" * 78)


def run_check():
    lora_32 = lora_total_params(32)
    router_4 = router_total_params(4)
    moe_4_32 = moe_total_params(4, 32)

    # Asserta que MoE E=4 r=32 = 4 × (LoRA r=32) + router
    expected_moe_4_32 = 4 * lora_32 + router_4
    assert moe_4_32 == expected_moe_4_32, f"Mismatch MoE params: {moe_4_32} != {expected_moe_4_32}"

    # Asserta que el r dense de igual presupuesto de adaptadores es exactamente 128
    lora_per_rank = N_LAYER * (N_EMBD + DIM_FFN) * 2
    r_dense_equal = (moe_4_32 - router_4) // lora_per_rank
    assert r_dense_equal == 128, f"Expected r_dense_equal 128, got {r_dense_equal}"
    assert lora_total_params(128) == 4 * lora_32, "LoRA r=128 does not match 4x LoRA r=32"

    print("param_math OK")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Aritmética de parámetros MoE-LoRA")
    parser.add_argument("--check", action="store_true", help="Comprueba la igualdad de presupuesto y assertions")
    args = parser.parse_args()

    if args.check:
        run_check()
    else:
        print_table()


if __name__ == "__main__":
    main()
