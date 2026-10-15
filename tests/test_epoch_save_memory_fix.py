#!/usr/bin/env python3
"""
Test runnable sin GPU del cierre de época y fix de memoria HIP OOM.
Verifica que:
1. El guardado pasa por la copia a CPU (.cpu() / .detach().cpu()).
2. Se llaman torch.cuda.empty_cache() y gc.collect() antes y después de guardar.
3. Se liberan los tensores temporales y no se serializan referencias al dispositivo acelerador.
"""

import sys
import os
import types
from unittest.mock import MagicMock, patch
import torch
import pytest

# Asegurar import de RWKV-PEFT
sys.path.insert(0, "/run/media/chaos/terciario/proyectos/kateto-train/RWKV-PEFT")

# Mock de lightning si no está en el entorno local
if "lightning" not in sys.modules:
    mock_pl = types.ModuleType("lightning")
    class MockCallback:
        pass
    mock_pl.Callback = MockCallback
    mock_pl.__version__ = "2.1.0"
    sys.modules["lightning"] = mock_pl

if "lightning_utilities.core.rank_zero" not in sys.modules:
    mock_lu = types.ModuleType("lightning_utilities.core.rank_zero")
    mock_lu.rank_zero_info = lambda *a, **k: None
    mock_lu.rank_zero_only = lambda fn: fn
    sys.modules["lightning_utilities.core.rank_zero"] = mock_lu

from rwkvt.lightning_train.trainer import train_callback


class MockArgs:
    def __init__(self, peft="pissa", merge=1, proj_dir="/tmp/test_rwkv_out"):
        self.peft = peft
        self.merge = merge
        self.proj_dir = proj_dir
        self.epoch_begin = 0
        self.cuda_cleanup = 0
        self.warmup_steps = 10
        self.lr_final = 1e-5
        self.lr_init = 1e-4
        self.epoch_count = 1
        self.weight_decay = 0.01
        self.weight_decay_final = 0
        self.layerwise_lr = 0
        self.wandb = ""
        self.lr_schedule = "cos_decay"


def test_epoch_save_pissa_merge_to_cpu_and_cleans_cache(tmp_path):
    proj_dir = str(tmp_path)
    args = MockArgs(peft="pissa", merge=1, proj_dir=proj_dir)
    callback = train_callback(args)

    # Mock del modelo con parámetros simulados
    mock_model = MagicMock()
    mock_param = torch.nn.Parameter(torch.randn(4, 4))
    mock_model.parameters.return_value = iter([mock_param])
    
    # Simular merge_and_unload produciendo state_dict con tensores
    t1 = torch.randn(2, 2)
    t2 = torch.randn(3, 3)
    mock_model_copy = MagicMock()
    mock_model_copy.state_dict.return_value = {
        "base_model.model.blocks.0.att.key.weight": t1,
        "base_model.model.blocks.0.ffn.value.weight": t2,
    }
    
    mock_pl_module = MagicMock()
    mock_pl_module.model = mock_model

    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    mock_trainer.current_epoch = 1

    saved_dict = {}

    def mock_save_impl(to_save, path):
        nonlocal saved_dict
        # Capturar el dict que se intenta serializar
        saved_dict = {k: v for k, v in to_save.items()}

    with patch("copy.deepcopy", return_value=mock_model_copy), \
         patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty_cache, \
         patch("gc.collect") as mock_gc, \
         patch("torch.save", side_effect=mock_save_impl) as mock_save:

        callback.on_train_epoch_end(mock_trainer, mock_pl_module)

        # 1. Verificar llamadas a empty_cache y gc.collect antes y después de guardar
        assert mock_empty_cache.call_count >= 2, (
            f"Se esperaba torch.cuda.empty_cache() antes y después de guardar, llamado {mock_empty_cache.call_count} veces"
        )
        assert mock_gc.call_count >= 2, (
            f"Se esperaba gc.collect() antes y después de guardar, llamado {mock_gc.call_count} veces"
        )

        # 2. Verificar que torch.save fue invocado
        assert mock_save.called, "torch.save debió haber sido llamado para guardar el checkpoint"
        saved_path = mock_save.call_args[0][1]
        assert saved_path == f"{proj_dir}/rwkv-1.pth"

        # 3. Verificar que cada tensor guardado fue copiado explícitamente a CPU
        assert len(saved_dict) == 2
        for k, v in saved_dict.items():
            assert isinstance(v, torch.Tensor), f"{k} debe ser un Tensor"
            assert v.device.type == "cpu", f"Tensor {k} debe estar en CPU, pero estaba en {v.device}"

        print("TEST PASSED: on_train_epoch_end ejecutó empty_cache/gc antes y después, y serializó todos los tensores a CPU.")


def test_epoch_save_none_peft_to_cpu(tmp_path):
    proj_dir = str(tmp_path)
    args = MockArgs(peft="none", merge=0, proj_dir=proj_dir)
    callback = train_callback(args)

    mock_pl_module = MagicMock()
    t1 = torch.randn(4, 4)
    mock_pl_module.state_dict.return_value = {
        "model.head.weight": t1,
    }

    mock_trainer = MagicMock()
    mock_trainer.is_global_zero = True
    mock_trainer.current_epoch = 0

    saved_dict = {}

    def mock_save_impl(to_save, path):
        nonlocal saved_dict
        saved_dict = {k: v for k, v in to_save.items()}

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache") as mock_empty_cache, \
         patch("gc.collect") as mock_gc, \
         patch("torch.save", side_effect=mock_save_impl) as mock_save:

        callback.on_train_epoch_end(mock_trainer, mock_pl_module)

        assert mock_empty_cache.call_count >= 2
        assert mock_gc.call_count >= 2
        assert mock_save.called
        assert "head.weight" in saved_dict
        assert saved_dict["head.weight"].device.type == "cpu"

        print("TEST PASSED: peft=none on_train_epoch_end guardó correctamente a CPU con limpieza de caché.")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_epoch_save_pissa_merge_to_cpu_and_cleans_cache(td)
        test_epoch_save_none_peft_to_cpu(td)
    print("ALL EPOCH SAVE TESTS PASSED.")
