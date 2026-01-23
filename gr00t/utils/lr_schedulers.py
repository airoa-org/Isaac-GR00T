import math
from bisect import bisect_right
from functools import partial
from typing import Iterable, Optional, Sequence

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _normalize_restart_steps(restart_steps: Optional[Iterable[int]], num_training_steps: int) -> list[int]:
    if not restart_steps:
        return []

    if isinstance(restart_steps, (int, float, str)):
        restart_iter: Iterable[int] = [int(restart_steps)]  # allow single scalar
    else:
        restart_iter = restart_steps

    cleaned: list[int] = []
    for step in restart_iter:
        try:
            step_int = int(step)
        except (TypeError, ValueError) as exc:
            raise ValueError("restart_steps must contain integers") from exc
        if step_int <= 0:
            continue
        if num_training_steps > 0 and step_int >= num_training_steps:
            continue
        cleaned.append(step_int)
    return sorted(set(cleaned))


def _get_cosine_with_soft_restarts_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    cycle_boundaries: Sequence[int],
):
    if not cycle_boundaries or cycle_boundaries[-1] <= 0 or current_step >= cycle_boundaries[-1]:
        return 0.0

    boundary_idx = bisect_right(cycle_boundaries, current_step) - 1
    if boundary_idx < 0 or boundary_idx + 1 >= len(cycle_boundaries):
        return 0.0

    cycle_start = cycle_boundaries[boundary_idx]
    cycle_end = cycle_boundaries[boundary_idx + 1]
    cycle_length = float(cycle_end - cycle_start)
    if cycle_length <= 0.0:
        return 0.0

    step_in_cycle = float(current_step - cycle_start)
    warmup_steps = max(0.0, float(num_warmup_steps))
    if warmup_steps > cycle_length:
        warmup_steps = cycle_length

    if warmup_steps > 0.0 and step_in_cycle < warmup_steps:
        return step_in_cycle / max(1.0, warmup_steps)

    progress = (step_in_cycle - warmup_steps) / max(1.0, cycle_length - warmup_steps)
    if progress >= 1.0:
        return 0.0
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def get_cosine_with_soft_restarts_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_specific_kwargs: Optional[dict] = None,
):
    """Cosine schedule with per-restart warmup ("soft" restarts).

    scheduler_specific_kwargs:
        - restart_steps (list[int], optional): global steps at which a new cycle starts.
        - last_epoch (int, optional): last epoch when resuming training.
    """
    kwargs = scheduler_specific_kwargs or {}
    restart_steps = _normalize_restart_steps(kwargs.get("restart_steps"), num_training_steps)
    last_epoch = kwargs.get("last_epoch", -1)
    cycle_boundaries = [0]
    cycle_boundaries.extend(restart_steps)
    cycle_boundaries.append(int(num_training_steps))

    lr_lambda = partial(
        _get_cosine_with_soft_restarts_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        cycle_boundaries=cycle_boundaries,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
