from typing import Any, Callable, Sequence, Tuple, Union

import torch
from ignite.utils import convert_tensor
from torch.nn import Module


def recursive_convert(
    data: Any,
    device: Union[str, torch.device] = None,
    non_blocking: bool = False,
) -> Any:
    '''
    Recursively moves all tensors in a nested data structure to the specified device.

    This function supports structures composed of tensors, lists, tuples, and dictionaries,
    and will apply `ignite.utils.convert_tensor` to each tensor found.

    Args:
        data (Any): A tensor or nested structure (list, tuple, dict) containing tensors.
        device (Union[str, torch.device], optional): The target device to move the tensors to
            (e.g., 'cuda', 'cpu'). If None, no conversion is applied.
        non_blocking (bool): Whether the transfer should be non-blocking (if supported by the device).

    Returns:
        Any: A new structure with the same type as `data`, where all tensors are moved to the specified device.
    '''
    if isinstance(data, torch.Tensor):
        return convert_tensor(data, device=device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {
            k: recursive_convert(v, device=device, non_blocking=non_blocking)
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [recursive_convert(v, device=device, non_blocking=non_blocking) for v in data]
    elif isinstance(data, tuple):
        return tuple(recursive_convert(v, device=device, non_blocking=non_blocking) for v in data)
    else:
        return data


def _prepare_batch(
    batch: Sequence[Any],
    device: Union[str, torch.device] = None,
    non_blocking: bool = False,
) -> Tuple[Any, Sequence[Any]]:
    '''
    Recursively moves a batch of data to the specified device and splits it into input and target(s).

    Args:
        batch (Sequence[Any]): A sequence where the first item is the model input and the rest are target values.
        device (Union[str, torch.device], optional): Device to which the batch should be moved (e.g., 'cuda', 'cpu').
        non_blocking (bool): Whether transfers should be asynchronous when possible.

    Returns:
        Tuple[Any, Sequence[Any]]: A tuple `(x, y)` where `x` is the model input and `y` is a sequence of target(s),
        all moved to the specified device.
    '''

    batch = list(batch)

    for i, item in enumerate(batch):
        batch[i] = recursive_convert(item, device=device, non_blocking=non_blocking)

    return (batch[0], batch[1:])


def _loss_fn(
    loss_fn: Union[Callable, Module],
) -> Callable:
    def wrapper(y_pred, y):
        return loss_fn(y_pred, *y)

    return wrapper
