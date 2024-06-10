from typing import Any, Callable, Sequence, Tuple, Union

import torch
from ignite.utils import convert_tensor
from torch.nn import Module


def _prepare_batch(
    batch: Sequence[Any],
    device: Union[str, torch.device] = None,
    non_blocking: bool = False,
) -> Tuple[Any, Sequence[Any]]:
    batch = list(batch)

    for i, item in enumerate(batch):
        batch[i] = convert_tensor(item, device=device, non_blocking=non_blocking)

    return (batch[0], batch[1:])


def _loss_fn(
    loss_fn: Union[Callable, Module],
) -> Callable:
    def wrapper(y_pred, y):
        return loss_fn(y_pred, *y)

    return wrapper
