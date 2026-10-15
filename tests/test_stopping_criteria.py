#!/usr/bin/env python3
"""
Test runnable sin GPU de las tres condiciones de parada de entrenamiento.
Verifica que el callback evalúa correctamente:
- Serie 1: Dispara Condición A (piso duro de entropía: EMA < 1.15 durante 20 steps seguidos).
- Serie 2: Dispara Condición B (meseta de validación: val_loss desciende < 0.02 en 50 steps).
- Serie 3: No dispara nada (corrida sana que progresa normalmente hacia loss ~1.25).
- Serie 4 (Bonus): Dispara Condición C (deriva contra la base: pérdida testigo sube > 15% respecto de step 0).
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rwkv_pipeline.stopping_criteria import StoppingCriteriaCallback


def test_serie_1_triggers_condition_a_entropy_floor():
    """Serie 1: train_loss_ema desciende bajo 1.15 y se mantiene durante 20 steps seguidos."""
    cb = StoppingCriteriaCallback(
        eval_interval=50,
        entropy_floor=1.15,
        entropy_consecutive_steps=20,
        plateau_delta=0.02,
        drift_max_increase=0.15,
    )

    stopped = False
    stop_step = None

    # Pasos 0 a 29: loss desciende de 1.80 a 1.20 (EMA aún > 1.15)
    for step in range(30):
        loss = 1.80 - (step / 30.0) * 0.60
        stop, reason = cb.on_step(step, loss)
        assert not stop, f"No debía detenerse en step {step}"

    # Pasos 30 en adelante: loss cae a 1.05 (forzando train_loss_ema < 1.15)
    # A partir de que EMA cruza 1.15, debe contar 20 steps seguidos y cortar.
    steps_under_count = 0
    for step in range(30, 80):
        loss = 1.05
        stop, reason = cb.on_step(step, loss)
        if cb.train_loss_ema < 1.15:
            steps_under_count += 1

        if stop:
            stopped = True
            stop_step = step
            break

    assert stopped, "La Serie 1 debió disparar la Condición A"
    assert cb.stop_condition == "A", f"Condición esperada 'A', obtenida: {cb.stop_condition}"
    assert cb.consecutive_below_floor == 20, f"Debió disparar exactamente al step 20 consecutivo bajo el piso, fue {cb.consecutive_below_floor}"
    print(f"TEST PASSED (Serie 1): Condición A disparada correctamente en step {stop_step} ({cb.stop_reason})")


def test_serie_2_triggers_condition_b_validation_plateau():
    """Serie 2: val_loss se estanca y mejora menos de 0.02 entre step t-50 y step t."""
    cb = StoppingCriteriaCallback(
        eval_interval=50,
        entropy_floor=1.15,
        entropy_consecutive_steps=20,
        plateau_delta=0.02,
        drift_max_increase=0.15,
    )

    # Valores simulados de validación:
    # step 0: 1.50
    # step 50: 1.42 (mejora 0.08 >= 0.02 -> no corta)
    # step 100: 1.41 (mejora 0.01 < 0.02 -> CORTA por meseta)
    val_loss_map = {
        0: 1.50,
        50: 1.42,
        100: 1.41,
    }

    stopped = False
    stop_step = None

    for step in range(101):
        # Loss de train saludable (~1.30 a 1.25, por encima de 1.15)
        train_loss = 1.35 - (step / 200.0) * 0.10
        val_override = val_loss_map.get(step, None)

        stop, reason = cb.on_step(step, train_loss, val_loss_override=val_override)
        if stop:
            stopped = True
            stop_step = step
            break

    assert stopped, "La Serie 2 debió disparar la Condición B en step 100"
    assert stop_step == 100, f"Debió detenerse en step 100, se detuvo en {stop_step}"
    assert cb.stop_condition == "B", f"Condición esperada 'B', obtenida: {cb.stop_condition}"
    print(f"TEST PASSED (Serie 2): Condición B disparada correctamente en step {stop_step} ({cb.stop_reason})")


def test_serie_3_healthy_run_triggers_nothing():
    """Serie 3: corrida sana con loss decreciente hacia ~1.25 y val_loss mejorando normalmente."""
    cb = StoppingCriteriaCallback(
        eval_interval=50,
        entropy_floor=1.15,
        entropy_consecutive_steps=20,
        plateau_delta=0.02,
        drift_max_increase=0.15,
    )

    # Validación con descensos saludables (>= 0.03 en cada intervalo)
    val_loss_map = {
        0: 2.10,
        50: 1.80,   # delta = 0.30
        100: 1.60,  # delta = 0.20
        150: 1.45,  # delta = 0.15
        200: 1.35,  # delta = 0.10
        250: 1.28,  # delta = 0.07
    }
    # Pérdida en prompts testigo estable (deriva mínima +2%)
    drift_loss_map = {
        0: 1.50,
        50: 1.51,
        100: 1.52,
        150: 1.53,
        200: 1.53,
        250: 1.53,
    }

    for step in range(251):
        # Pérdida de train disminuyendo de 2.00 a 1.25 (sin cruzar bajo 1.15)
        train_loss = 2.00 - (step / 250.0) * 0.75  # finaliza en 1.25
        val_override = val_loss_map.get(step, None)
        drift_override = drift_loss_map.get(step, None)

        stop, reason = cb.on_step(
            step, train_loss, val_loss_override=val_override, drift_loss_override=drift_override
        )
        assert not stop, f"Corrida sana no debió disparar parada en step {step} (motivo: {reason})"

    assert not cb.stop_triggered, "La Serie 3 no debió activar ninguna condición de parada"
    assert cb.train_loss_ema >= 1.15, f"EMA final {cb.train_loss_ema} debe ser >= 1.15"
    print(f"TEST PASSED (Serie 3): Corrida sana completó 250 steps sin detención (loss final: {train_loss:.2f})")


def test_serie_4_triggers_condition_c_base_drift():
    """Serie 4: pérdida testigo sube más de 15% respecto de step 0."""
    cb = StoppingCriteriaCallback(
        eval_interval=50,
        entropy_floor=1.15,
        entropy_consecutive_steps=20,
        plateau_delta=0.02,
        drift_max_increase=0.15,
    )

    drift_loss_map = {
        0: 1.40,
        50: 1.65,  # subió (1.65 - 1.40) / 1.40 = +17.86% > 15%
    }
    val_loss_map = {
        0: 1.80,
        50: 1.70,  # mejora 0.10 (no dispara B)
    }

    stopped = False
    stop_step = None

    for step in range(51):
        train_loss = 1.30
        val_override = val_loss_map.get(step, None)
        drift_override = drift_loss_map.get(step, None)

        stop, reason = cb.on_step(
            step, train_loss, val_loss_override=val_override, drift_loss_override=drift_override
        )
        if stop:
            stopped = True
            stop_step = step
            break

    assert stopped, "La Serie 4 debió disparar la Condición C"
    assert stop_step == 50, f"Debió disparar en step 50, fue {stop_step}"
    assert cb.stop_condition == "C", f"Condición esperada 'C', obtenida: {cb.stop_condition}"
    print(f"TEST PASSED (Serie 4): Condición C disparada correctamente en step {stop_step} ({cb.stop_reason})")


if __name__ == "__main__":
    test_serie_1_triggers_condition_a_entropy_floor()
    test_serie_2_triggers_condition_b_validation_plateau()
    test_serie_3_healthy_run_triggers_nothing()
    test_serie_4_triggers_condition_c_base_drift()
    print("\n>>> TODOS LOS TESTS DE CRITERIOS DE PARADA PASARON CON ÉXITO <<<")
