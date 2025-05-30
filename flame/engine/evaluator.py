from __future__ import annotations

from typing import Any, Callable, Union

import ignite.engine as ie
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from .engine import Engine
from .utils import _prepare_batch


class Evaluator(Engine):
    '''
    Custom Evaluator Engine for model evaluation.

    This class encapsulates evaluation logic and sets up the appropriate
    evaluation step function depending on the device and AMP mode.

    Inherits from:
        Engine: Base class that runs a process function over each batch.
    '''

    @staticmethod
    def factory(
        model: Module,
        data: DataLoader,
        device: Union[str, torch.device] = None,
        max_epochs: int = None,
        epoch_length: int = None,
        non_blocking: bool = False,
        prepare_batch: Callable = _prepare_batch,
        model_transform: Callable[[Any], Any] = lambda output: output,
        output_transform: Callable[[Any, Any, Any], Any] = lambda x, y, y_pred: (y_pred, *y),
        amp_mode: str = None,
        model_fn: Callable[[torch.nn.Module, Any], Any] = lambda model, x: model(x),
    ) -> Evaluator:
        '''
        Factory method for creating an `Evaluator` instance for supervised evaluation.

        This method sets up the appropriate evaluation step function depending on the
        hardware (CPU/GPU/TPU) and automatic mixed precision (AMP) configuration.

        Args:
            model (torch.nn.Module): The model to be evaluated.
            data (torch.utils.data.DataLoader): The evaluation data loader.
            device (Union[str, torch.device], optional): The device to run evaluation on.
                Can be a string (e.g., 'cuda', 'cpu') or a torch.device instance.
            max_epochs (int, optional): Number of evaluation epochs to run. Defaults to 1 if not specified.
            epoch_length (int, optional): Number of iterations per epoch. If None, determined from the DataLoader.
            non_blocking (bool): Whether to use non-blocking data transfer to the device.
            prepare_batch (Callable): Function to convert a batch into input/output tensors.
            model_transform (Callable[[Any], Any]): Optional transform applied to the model output
                before post-processing.
            output_transform (Callable[[Any, Any, Any], Any]): Function that processes `x`, `y`, and `y_pred`
                and returns a value stored in `engine.state.output` after each iteration.
            amp_mode (str, optional): AMP mode for evaluation. Can be 'amp'. If None, AMP is not used.
            model_fn (Callable[[torch.nn.Module, Any], Any]): A callable that takes `model` and input tensor `x`,
                and returns predictions (e.g., for handling models with multiple inputs or non-standard forward passes).

        Returns:
            Evaluator: An instance of `Evaluator` configured for evaluation.
        '''
        device_type = device.type if isinstance(device, torch.device) else device
        on_tpu = 'xla' in device_type if device_type is not None else False
        on_mps = 'mps' in device_type if device_type is not None else False
        mode, _ = ie._check_arg(on_tpu, on_mps, amp_mode, None)
        model.to(device)

        if mode == 'amp':
            evaluate_step = ie.supervised_evaluation_step_amp(
                model=model,
                device=device,
                non_blocking=non_blocking,
                prepare_batch=prepare_batch,
                model_transform=model_transform,
                output_transform=output_transform,
                model_fn=model_fn,
            )
        else:
            evaluate_step = ie.supervised_evaluation_step(
                model=model,
                device=device,
                non_blocking=non_blocking,
                prepare_batch=prepare_batch,
                model_transform=model_transform,
                output_transform=output_transform,
                model_fn=model_fn,
            )

        evaluator = Evaluator(
            data=data,
            model=model,
            process_function=evaluate_step,
            device=device,
            max_epochs=max_epochs,
            epoch_length=epoch_length,
        )

        return evaluator
