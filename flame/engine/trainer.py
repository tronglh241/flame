from __future__ import annotations

from typing import Any, Callable, Union

import ignite.engine as ie
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .engine import Engine
from .utils import _loss_fn, _prepare_batch


class Trainer(Engine):
    def __init__(
        self,
        optimizer: Optimizer,
        loss_fn: Callable,
        **kwargs: Any,
    ):
        super(Trainer, self).__init__(**kwargs)
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @staticmethod
    def factory(
        model: Module,
        data: DataLoader,
        optimizer: Optimizer,
        loss_fn: Union[Callable, Module],
        device: Union[str, torch.device] = None,
        max_epochs: int = None,
        epoch_length: int = None,
        non_blocking: bool = False,
        prepare_batch: Callable = _prepare_batch,
        model_transform: Callable[[Any], Any] = lambda output: output,
        output_transform: Callable[[Any, Any, Any, torch.Tensor], Any] = lambda x, y, y_pred, loss: loss.item(),
        amp_mode: str = None,
        scaler: Union[bool, 'torch.cuda.amp.GradScaler'] = False,
        gradient_accumulation_steps: int = 1,
        model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
    ) -> Trainer:
        device_type = device.type if isinstance(device, torch.device) else device
        on_tpu = 'xla' in device_type if device_type is not None else False
        on_mps = 'mps' in device_type if device_type is not None else False
        mode, _scaler = ie._check_arg(on_tpu, on_mps, amp_mode, scaler)
        model.to(device)
        loss_fn = _loss_fn(loss_fn)

        if mode == 'amp':
            _update = ie.supervised_training_step_amp(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                non_blocking=non_blocking,
                prepare_batch=prepare_batch,
                model_transform=model_transform,
                output_transform=output_transform,
                scaler=_scaler,
                gradient_accumulation_steps=gradient_accumulation_steps,
                model_fn=model_fn,
            )
        elif mode == 'apex':
            _update = ie.supervised_training_step_apex(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                non_blocking=non_blocking,
                prepare_batch=prepare_batch,
                model_transform=model_transform,
                output_transform=output_transform,
                gradient_accumulation_steps=gradient_accumulation_steps,
                model_fn=model_fn,
            )
        elif mode == 'tpu':
            _update = ie.supervised_training_step_tpu(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                non_blocking=non_blocking,
                prepare_batch=prepare_batch,
                model_transform=model_transform,
                output_transform=output_transform,
                gradient_accumulation_steps=gradient_accumulation_steps,
                model_fn=model_fn,
            )
        else:
            _update = ie.supervised_training_step(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
                non_blocking=non_blocking,
                prepare_batch=prepare_batch,
                model_transform=model_transform,
                output_transform=output_transform,
                gradient_accumulation_steps=gradient_accumulation_steps,
                model_fn=model_fn,
            )

        trainer = Trainer(
            optimizer=optimizer,
            loss_fn=loss_fn,
            data=data,
            model=model,
            process_function=_update,
            device=device,
            max_epochs=max_epochs,
            epoch_length=epoch_length,
        )

        if _scaler and scaler and isinstance(scaler, bool):
            trainer.state.scaler = _scaler  # type: ignore[attr-defined]

        return trainer
