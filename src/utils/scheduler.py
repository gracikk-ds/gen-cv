"""Module provided Cosine Annealing with Warmup LR Scheduler."""

from functools import partial
from math import cos, pi
from typing import Iterator, List, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import _LRScheduler  # noqa: WPS450
from torch.optim.lr_scheduler import MultiStepLR


class CosineAnnealingWarmup(_LRScheduler):
    """Cosine Annealing with Warmup LR Scheduler.

    This LR scheduler combines a warm-up phase with a cosine annealing decay. It gradually increases the learning rate
    during the warm-up phase and then smoothly decays it using a cosine function.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        total_steps: int = 10000,
        last_epoch: int = -1,
    ):
        """Initialize the CosineAnnealingWarmup LR scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The PyTorch optimizer for which to adjust the learning rate.
            min_lr (float): The minimum learning rate to be used during the cosine annealing phase. Defaults to 0.001.
            warmup_steps (int): The number of warm-up steps during which the learning rate increases linearly. D-s to 0.
            total_steps (int): The total number of steps. Defaults to 10,000.
            last_epoch (int): The index of the last epoch. If not specified, it will be set to -1.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_steps = total_steps - warmup_steps
        self.min_lr = min_lr
        self.last_epoch = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore
        """Get the current learning rates for all parameter groups.

        Returns:
            List[float]: A list of learning rates for each parameter group.
        """
        warmup_corrected_epoch_state = self.last_epoch - self.warmup_steps

        if warmup_corrected_epoch_state < 0:  # check current epoch number adjusted by warmup_steps
            mult = self.last_epoch / self.warmup_steps  # warmup LR multiplier
        else:
            mult = 0.5 * (1 + cos(pi * (warmup_corrected_epoch_state) / self.decay_steps))  # starts cosine decay
        min_lr_coefs = [lr / self.base_lrs[0] for lr in self.base_lrs]  # noqa: WPS111
        min_lrs = [self.min_lr * min_lr_coef for min_lr_coef in min_lr_coefs]

        lrs = []

        for base_lr, min_lr in zip(self.base_lrs, min_lrs):
            lrs.append(min_lr + (base_lr - min_lr) * mult)
        return lrs


class WarmupMultiStepLR(MultiStepLR):
    """MultiStepLR scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        initial_lr: float,
        milestones: Tuple[int, ...],
        gamma: float = 0.1,
        last_epoch: int = -1,
    ):
        """Initialize the WarmupMultiStepLR.

        Args:
            optimizer (torch.optim.Optimizer): The PyTorch optimizer for which to adjust the learning rate.
            warmup_steps (int): The number of warm-up steps during which the learning rate increases linearly.
            initial_lr (float): The initial learning rate.
            milestones (Tuple[int, ...]): List of epoch indices. Must be increasing.
            gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        for param_group in optimizer.param_groups:
            param_group["initial_lr"] = initial_lr
            param_group["warmup_diff"] = param_group["lr"] - initial_lr
        super().__init__(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)

    def get_lr(self) -> List[float]:  # type: ignore
        """Get current learning rate.

        Returns:
            List[float]: list of lr for each param group.
        """
        if self.last_epoch <= self.warmup_steps:
            warmup_factor = self.last_epoch / self.warmup_steps
            new_lrs = []
            for param_group in self.optimizer.param_groups:
                if param_group["lr"] == 0:
                    new_lrs.append(param_group["lr"])
                    continue
                new_lr = param_group["initial_lr"] + warmup_factor * param_group["warmup_diff"]
                new_lrs.append(new_lr)
            return new_lrs
        return super().get_lr()  # type: ignore


class OptimizerSchedulerHandler:
    """Handler for optimizer and scheduler."""

    def __init__(
        self,
        optimizer: partial[torch.optim.Optimizer],
        scheduler: partial[_LRScheduler],
        model_parameters: Iterator[torch.Tensor],
        max_grad_norm: float | None = None,
    ):
        """Initialize the OptimizerSchedulerHandler.

        Args:
            optimizer (partial[torch.optim.Optimizer]): The optimizer instance (partially initialized).
            scheduler (partial[_LRScheduler]): The scheduler instance (partially initialized).
            model_parameters (Iterator[torch.Tensor]): The model parameters to optimize.
            max_grad_norm (float | None): The maximum gradient norm to clip. Defaults to None.
        """
        self.optimizer = optimizer(model_parameters)  # partialy initialized optimizer
        self.scheduler = scheduler(optimizer=self.optimizer)  # partialy initialized scheduler
        self.max_grad_norm = max_grad_norm

    def step(self, new_epoch: bool = False):
        """Perform a single optimization step and then update the scheduler.

        Args:
            new_epoch (bool): Whether a new epoch is starting.
        """
        # Optionally clip gradients
        if self.max_grad_norm is not None:
            params = [p for group in self.optimizer.param_groups for p in group["params"] if p.requires_grad]
            clip_grad_norm_(params, self.max_grad_norm)

        # Update model parameters
        self.optimizer.step()
        # Update the learning rate based on the scheduler only at the start of a new epoch
        if new_epoch:
            self.scheduler.step()

    def zero_grad(self):
        """Reset the gradients of all optimized parameters."""
        self.optimizer.zero_grad()

    def get_lr(self) -> List[float]:  # type: ignore
        """Return the current learning rate(s).

        Returns:
            List[float]: The current learning rate(s).
        """
        # Some schedulers have multiple param groups
        return [group["lr"] for group in self.optimizer.param_groups]
