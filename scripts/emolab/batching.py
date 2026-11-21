"""Batch runner that halves the batch size when the GPU runs out of memory."""
from collections.abc import Callable, Iterator
from typing import TypeVar

import torch

T = TypeVar("T")
R = TypeVar("R")


def run_with_oom_backoff(fn: Callable[[list[T]], list[R]], items: list[T],
                         batch_size: int) -> Iterator[tuple[list[T], list[R]]]:
    """Yield (batch, fn(batch)) over items; on CUDA OOM halve the batch and retry.

    The reduced size is kept for the remaining batches. An OOM at batch size 1
    is re-raised.
    """
    i = 0
    while i < len(items):
        batch = items[i:i + batch_size]
        try:
            results = fn(batch)
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            batch_size //= 2
            results = None
        if results is None:
            torch.cuda.empty_cache()
            continue
        yield batch, results
        i += len(batch)
