from __future__ import annotations

from typing import Any, Callable, Union

import ignite.engine as ie
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from .engine import Engine
from .utils import _prepare_batch


class Evaluator(Engine):
    @staticmethod
    def factory(
        model: Module,
        data: DataLoader,
        device: Union[str, torch.device] = None,
        max_epochs: int = None,
        epoch_length: int = None,
        seed: int = None,
        non_blocking: bool = False,
        prepare_batch: Callable = _prepare_batch,
        output_transform: Callable[[Any, Any, Any], Any] = lambda x, y, y_pred: (y_pred, *y),
        amp_mode: str = None,
    ) -> Evaluator:
        device_type = device.type if isinstance(device, torch.device) else device
        on_tpu = "xla" in device_type if device_type is not None else False
        mode, _ = ie._check_arg(on_tpu, amp_mode, None)
        model.to(device)

        if mode == "amp":
            evaluate_step = ie.supervised_evaluation_step_amp(
                model=model,
                device=device,
                non_blocking=non_blocking,
                prepare_batch=prepare_batch,
                output_transform=output_transform,
            )
        else:
            evaluate_step = ie.supervised_evaluation_step(
                model=model,
                device=device,
                non_blocking=non_blocking,
                prepare_batch=prepare_batch,
                output_transform=output_transform,
            )

        evaluator = Evaluator(
            data=data,
            model=model,
            process_function=evaluate_step,
            device=device,
            max_epochs=max_epochs,
            epoch_length=epoch_length,
            seed=seed,
        )

        return evaluator
