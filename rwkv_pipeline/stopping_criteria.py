#!/usr/bin/env python3
"""
Callback de criterios de parada temprana para entrenamiento RWKV-7 G1k.
Implementa las 3 condiciones de parada evaluadas cada 50 steps
(con objetivo declarado de frenar cerca de loss 1.25 para evitar memorización):
- A: Piso duro de entropía: train_loss_ema < 1.15 durante 20 steps seguidos.
- B: Meseta de validación: val_loss baja menos de 0.02 entre step t-50 y step t (micro-set 64 muestras).
- C: Deriva contra la base: pérdida de generación sobre prompts testigo sube > 15% respecto de step 0.
"""

import os
import json
from collections import deque
from typing import Optional, Tuple, Callable

try:
    import lightning as pl
    CallbackBase = pl.Callback
except ImportError:
    try:
        import pytorch_lightning as pl
        CallbackBase = pl.Callback
    except ImportError:
        class CallbackBase:
            pass


class StoppingCriteriaCallback(CallbackBase):
    def __init__(
        self,
        eval_interval: int = 50,
        entropy_floor: float = 1.15,
        entropy_consecutive_steps: int = 20,
        plateau_delta: float = 0.02,
        drift_max_increase: float = 0.15,
        val_evaluator: Optional[Callable] = None,
        drift_evaluator: Optional[Callable] = None,
        proj_dir: Optional[str] = None,
    ):
        super().__init__()
        self.eval_interval = eval_interval
        self.entropy_floor = entropy_floor
        self.entropy_consecutive_steps = entropy_consecutive_steps
        self.plateau_delta = plateau_delta
        self.drift_max_increase = drift_max_increase
        self.val_evaluator = val_evaluator
        self.drift_evaluator = drift_evaluator
        self.proj_dir = proj_dir

        # Estado A: Piso duro de entropía
        self.train_loss_ema = None
        self.ema_alpha = 2.0 / (20.0 + 1.0)  # Media móvil exponencial período 20
        self.loss_window = deque(maxlen=20)
        self.consecutive_below_floor = 0

        # Estado B: Meseta de validación (cada 50 steps)
        self.val_losses = {}  # step -> val_loss

        # Estado C: Deriva contra base (prompts testigo)
        self.base_witness_loss = None  # pérdida testigo a step 0
        self.witness_losses = {}       # step -> witness_loss

        # Control de parada
        self.stop_triggered = False
        self.stop_reason = None
        self.stop_condition = None
        self.stopped_step = None

    def update_train_loss(self, step: int, loss: float) -> Tuple[bool, Optional[str]]:
        """Actualiza EMA de loss de entrenamiento y evalúa Condición A."""
        loss_val = float(loss)
        self.loss_window.append(loss_val)

        if self.train_loss_ema is None:
            self.train_loss_ema = loss_val
        else:
            self.train_loss_ema = self.ema_alpha * loss_val + (1.0 - self.ema_alpha) * self.train_loss_ema

        # Chequeo Condición A: piso duro de entropía (< 1.15 durante 20 steps seguidos)
        if self.train_loss_ema < self.entropy_floor:
            self.consecutive_below_floor += 1
            if self.consecutive_below_floor >= self.entropy_consecutive_steps:
                self.stop_triggered = True
                self.stop_condition = "A"
                self.stopped_step = step
                self.stop_reason = (
                    f"Condicion A (piso duro de entropia): train_loss_ema={self.train_loss_ema:.4f} "
                    f"< {self.entropy_floor} durante {self.consecutive_below_floor} steps seguidos "
                    f"(step {step})"
                )
                return True, self.stop_reason
        else:
            self.consecutive_below_floor = 0

        return False, None

    def evaluate_periodic(
        self,
        step: int,
        model=None,
        val_loss_override: Optional[float] = None,
        drift_loss_override: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Evalúa Condiciones B y C cada 50 steps (o overrides provistos para tests)."""
        if step % self.eval_interval != 0 and step != 0:
            return False, None

        # Condición B: Meseta de validación (micro-set 64 muestras)
        val_loss = None
        if val_loss_override is not None:
            val_loss = float(val_loss_override)
        elif self.val_evaluator is not None and model is not None:
            try:
                val_loss = float(self.val_evaluator(model))
            except Exception as e:
                print(f"[StoppingCriteria] Error en val_evaluator: {e}")

        if val_loss is not None:
            self.val_losses[step] = val_loss
            prev_step = step - self.eval_interval
            if prev_step in self.val_losses:
                prev_val_loss = self.val_losses[prev_step]
                improvement = prev_val_loss - val_loss
                if improvement < self.plateau_delta:
                    self.stop_triggered = True
                    self.stop_condition = "B"
                    self.stopped_step = step
                    self.stop_reason = (
                        f"Condicion B (meseta de validacion): val_loss descendio {improvement:.4f} "
                        f"(de {prev_val_loss:.4f} en step {prev_step} a {val_loss:.4f} en step {step}), "
                        f"inferior al minimo requerido {self.plateau_delta:.2f}"
                    )
                    return True, self.stop_reason

        # Condición C: Deriva contra la base (set testigo)
        drift_loss = None
        if drift_loss_override is not None:
            drift_loss = float(drift_loss_override)
        elif self.drift_evaluator is not None and model is not None:
            try:
                drift_loss = float(self.drift_evaluator(model))
            except Exception as e:
                print(f"[StoppingCriteria] Error en drift_evaluator: {e}")

        if drift_loss is not None:
            self.witness_losses[step] = drift_loss
            if self.base_witness_loss is None:
                self.base_witness_loss = drift_loss
            else:
                drift_increase = (drift_loss - self.base_witness_loss) / (self.base_witness_loss + 1e-8)
                if drift_increase > self.drift_max_increase:
                    self.stop_triggered = True
                    self.stop_condition = "C"
                    self.stopped_step = step
                    self.stop_reason = (
                        f"Condicion C (deriva contra la base): perdida en prompts testigo subio "
                        f"{drift_increase * 100:.2f}% (de {self.base_witness_loss:.4f} en step 0 a "
                        f"{drift_loss:.4f} en step {step}), superando el +{self.drift_max_increase * 100:.0f}% permitido"
                    )
                    return True, self.stop_reason

        return False, None

    def on_step(
        self,
        step: int,
        train_loss: float,
        model=None,
        val_loss_override: Optional[float] = None,
        drift_loss_override: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Punto de llamada por step durante el entrenamiento."""
        # 1. Chequeo condición A
        stop_a, reason_a = self.update_train_loss(step, train_loss)
        if stop_a:
            self._save_reason()
            return True, reason_a

        # 2. Chequeo condiciones B y C en pasos múltiplos de eval_interval
        if step % self.eval_interval == 0:
            stop_bc, reason_bc = self.evaluate_periodic(
                step, model=model, val_loss_override=val_loss_override, drift_loss_override=drift_loss_override
            )
            if stop_bc:
                self._save_reason()
                return True, reason_bc

        return False, None

    def _save_reason(self):
        """Guarda metadata de detención en proj_dir si está configurado."""
        if self.proj_dir and os.path.isdir(self.proj_dir):
            out_file = os.path.join(self.proj_dir, "stopping_reason.json")
            try:
                with open(out_file, "w") as f:
                    json.dump({
                        "stop_triggered": self.stop_triggered,
                        "stop_condition": self.stop_condition,
                        "stopped_step": self.stopped_step,
                        "reason": self.stop_reason,
                        "train_loss_ema": self.train_loss_ema,
                        "consecutive_below_floor": self.consecutive_below_floor,
                        "val_losses": self.val_losses,
                        "base_witness_loss": self.base_witness_loss,
                        "witness_losses": self.witness_losses,
                    }, f, indent=2)
            except Exception as e:
                print(f"[StoppingCriteria] No se pudo guardar stopping_reason.json: {e}")

    # --- Hook para integración nativa con PyTorch Lightning ---
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not hasattr(trainer, "my_loss") or trainer.my_loss is None:
            # TODO: El trainer no expone trainer.my_loss de forma estándar; punto de enganche alternativo vía outputs['loss']
            loss_val = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if loss_val is not None and hasattr(loss_val, "item"):
                loss_val = loss_val.item()
            else:
                return
        else:
            loss_val = float(trainer.my_loss)

        step = int(trainer.global_step)
        stop, reason = self.on_step(step, loss_val, model=pl_module)
        if stop:
            print(f"\n🛑 [STOPPING CRITERIA ACTIVADO] {reason}\n", flush=True)
            trainer.should_stop = True
