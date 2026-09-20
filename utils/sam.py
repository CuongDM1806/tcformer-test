"""Sharpness-Aware Minimization with a closure-based PyTorch interface."""

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


class SAM(Optimizer):
    """Wrap a base optimizer with the two-pass SAM update.

    ``step`` deliberately evaluates the supplied closure twice. This matches
    Lightning automatic optimization, whose closure owns zero_grad, forward,
    backward, mixed precision, and distributed gradient synchronization.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        base_optimizer: type[Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        model: nn.Module | None = None,
        **kwargs: Any,
    ):
        if rho < 0.0:
            raise ValueError("SAM rho must be non-negative.")

        self.base_optimizer = base_optimizer(params, **kwargs)
        defaults = dict(rho=float(rho), adaptive=bool(adaptive), **kwargs)
        super().__init__(self.base_optimizer.param_groups, defaults)
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = float(rho)
            group["adaptive"] = bool(adaptive)

        self.model = model
        self._perturbations: dict[nn.Parameter, Tensor] = {}

    @torch.no_grad()
    def _gradient_norm(self) -> Tensor | None:
        norms = []
        shared_device = None
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("SAM does not support sparse gradients.")
                shared_device = parameter.device if shared_device is None else shared_device
                scale = parameter.abs() if adaptive else 1.0
                norms.append((scale * parameter.grad).norm(p=2).to(shared_device))
        if not norms:
            return None
        return torch.linalg.vector_norm(torch.stack(norms), ord=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> bool:
        grad_norm = self._gradient_norm()
        if grad_norm is None:
            return False

        self._perturbations.clear()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            adaptive = group["adaptive"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                perturbation = parameter.grad * scale.to(parameter)
                if adaptive:
                    perturbation = parameter.square() * perturbation
                parameter.add_(perturbation)
                self._perturbations[parameter] = perturbation

        if zero_grad:
            self.zero_grad(set_to_none=True)
        return True

    @torch.no_grad()
    def _restore_weights(self) -> None:
        for parameter, perturbation in self._perturbations.items():
            parameter.sub_(perturbation)
        self._perturbations.clear()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        self._restore_weights()
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    def _disable_batch_norm_updates(self):
        if self.model is None:
            return []
        states = []
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                states.append((module, module.track_running_stats))
                module.track_running_stats = False
        return states

    @staticmethod
    def _restore_batch_norm_updates(states) -> None:
        for module, track_running_stats in states:
            module.track_running_stats = track_running_stats

    def _set_second_pass(self, enabled: bool):
        if self.model is None:
            return None
        previous = getattr(self.model, "_sam_second_pass", None)
        self.model._sam_second_pass = enabled
        return previous

    def _restore_second_pass(self, previous) -> None:
        if self.model is None:
            return
        if previous is None:
            delattr(self.model, "_sam_second_pass")
        else:
            self.model._sam_second_pass = previous

    @torch.no_grad()
    def _abort_perturbed_step(self) -> None:
        self._restore_weights()
        self.zero_grad(set_to_none=True)

    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor:
        if closure is None:
            raise RuntimeError("SAM requires a closure for its two forward passes.")

        with torch.enable_grad():
            loss = closure()
        if not self.first_step(zero_grad=True):
            return loss

        batch_norm_states = self._disable_batch_norm_updates()
        previous_second_pass = self._set_second_pass(True)
        try:
            with torch.enable_grad():
                closure()
        except Exception:
            self._abort_perturbed_step()
            raise
        finally:
            self._restore_second_pass(previous_second_pass)
            self._restore_batch_norm_updates(batch_norm_states)

        self.second_step(zero_grad=True)
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
