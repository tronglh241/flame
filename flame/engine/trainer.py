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
    '''
    Custom Trainer Engine for supervised training.

    This class encapsulates the training logic and stores references to the
    optimizer and loss function used in the training loop.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        loss_fn (Callable): Loss function used to compute training loss.
        **kwargs: Additional keyword arguments passed to the parent Engine class.
    '''

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
        '''
        Factory method for creating a `Trainer` instance for supervised training.

        This method sets up the appropriate training step function depending on the device
        (CPU/GPU/TPU), AMP mode, and gradient scaling configuration.

        Args:
            model (torch.nn.Module): The model to train.
            data (torch.utils.data.DataLoader): The training data loader.
            optimizer (torch.optim.Optimizer): The optimizer used for parameter updates.
            loss_fn (Union[Callable, torch.nn.Module]): The loss function used during training.
            device (Union[str, torch.device], optional): The device to run training on.
                Can be a string or a torch.device instance.
            max_epochs (int, optional): Number of epochs to run. If None, defaults to 1.
            epoch_length (int, optional): Number of iterations per epoch. If None, it's automatically
                determined from the data loader.
            non_blocking (bool): Whether to use non-blocking data transfer to the device.
            prepare_batch (Callable): Function to convert a batch into input/output tensors.
            model_transform (Callable): Optional transform to apply to the model output before loss computation.
            output_transform (Callable): Function that processes `x`, `y`, `y_pred`, `loss` and returns a
                value stored in `engine.state.output` after each iteration.
            amp_mode (str, optional): AMP mode for mixed precision training. Can be 'amp' or 'apex'.
            scaler (Union[bool, torch.cuda.amp.GradScaler]): Gradient scaler used with AMP. If True,
                a default GradScaler will be created.
            gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating weights.
            model_fn (Callable): A callable that takes `model` and input tensor `x`, and returns the predictions.

        Returns:
            Trainer: An instance of `Trainer` initialized with the configured training engine.
        '''
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
