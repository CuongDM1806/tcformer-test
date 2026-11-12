import math
from types import SimpleNamespace

import pytest
import torch
import yaml

from models.hada_tcformer import HADATCFormer, random_session_shift


def test_zero_shift_is_identity():
    x = torch.randn(4, 22, 100)
    assert torch.allclose(random_session_shift(x, 0.0, 0.0), x, atol=1e-6)


def test_energy_only_shift_stays_in_range():
    torch.manual_seed(0)
    x = torch.randn(256, 22, 200)
    shifted = random_session_shift(x, log_energy=0.47, spatial=0.0)
    ratio = shifted.square().sum(dim=(1, 2)) / x.square().sum(dim=(1, 2))
    assert ratio.min() >= math.exp(-0.47) - 1e-4
    assert ratio.max() <= math.exp(0.47) + 1e-4
    assert ratio.min() < 0.8 and ratio.max() > 1.25  # the range is actually used


def test_spatial_shift_is_spd_and_trial_specific():
    torch.manual_seed(1)
    x = torch.eye(8).unsqueeze(0).repeat(3, 1, 1)  # trials = identity, so x' = A
    mixing = random_session_shift(x, log_energy=0.0, spatial=0.4)
    assert torch.allclose(mixing, mixing.transpose(1, 2), atol=1e-5)
    assert torch.linalg.eigvalsh(mixing).min() > 0
    # det(A) = 1 without an energy change: S is traceless.
    assert torch.allclose(torch.linalg.det(mixing), torch.ones(3), atol=1e-4)
    assert not torch.allclose(mixing[0], mixing[1])


def _model(weight):
    config = yaml.safe_load(open("configs/hada_tcformer.yaml"))
    kwargs = dict(config["model_kwargs"])
    kwargs.update(shift_consistency_weight=weight, log_every_n_batches=0, mamba_scan_mode="parallel")
    torch.manual_seed(3)
    return HADATCFormer(n_channels=22, n_classes=4, max_epochs=10, **kwargs)


def _batch():
    g = torch.Generator().manual_seed(5)
    source = torch.randn(8, 22, 1000, generator=g)
    labels = torch.randint(0, 4, (8,), generator=g)
    target = torch.randn(8, 22, 1000, generator=g)
    return {"source": (source, labels), "target": target}


def _running_state(model):
    return {k: v.clone() for k, v in model.state_dict().items()
            if "running" in k or k.endswith(("source_mean", "source_var", "target_mean", "target_var"))}


@pytest.mark.filterwarnings("ignore")
def test_consistency_adds_loss_but_not_running_statistics(monkeypatch):
    batch = _batch()
    states, losses = [], []
    for weight in (0.0, 1.0):
        model = _model(weight)
        monkeypatch.setattr(model, "_grl_alpha", lambda: 1.0)
        # training_step reads trainer.num_training_batches; logging is not under test.
        model._trainer = SimpleNamespace(num_training_batches=1, current_epoch=0)
        monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
        model.train()
        torch.manual_seed(11)
        losses.append(model.training_step(batch, 0))
        states.append(_running_state(model))
    assert states[0].keys() == states[1].keys() and states[0]
    for key in states[0]:
        assert torch.allclose(states[0][key], states[1][key]), key
    assert torch.isfinite(losses[1]) and losses[1] > losses[0]


@pytest.mark.filterwarnings("ignore")
def test_consistency_gradient_reaches_backbone(monkeypatch):
    model = _model(1.0)
    monkeypatch.setattr(model, "_grl_alpha", lambda: 1.0)
    model.train()
    batch = _batch()
    teacher = torch.randn(16, 4)
    loss = model._shift_consistency(batch["source"][0], batch["target"], teacher)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    grads = [p.grad for p in model.model.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert model.domain_norm.alpha_logit.grad is None  # alpha stays a task-only parameter
