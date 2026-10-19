import pytest
import torch

from models.hada_tcformer import AutoDIALFeatureNorm


def _layer(mode="shrinkage", dim=64, groups=4, momentum=0.1):
    return AutoDIALFeatureNorm(dim, n_groups=groups, momentum=momentum, alpha_mode=mode)


def test_shrinkage_weight_follows_shift_relative_to_noise():
    layer = _layer()
    zeros, ones = torch.zeros(3), torch.ones(3)
    # Gap far above the noise -> target statistics; within the noise -> source.
    gap = torch.tensor([10.0, 0.0, 0.1])
    alpha = layer.shrinkage_alpha(zeros, ones, 50, gap, ones, 50)
    assert alpha[0] == pytest.approx(1.0 - 0.04 / 100.0)
    assert alpha[1] == 0.0
    assert alpha[2] == 0.0          # 0.1^2 = 0.01 < noise 2/50 = 0.04
    # More target trials -> less noise -> more trust in the target.
    small = layer.shrinkage_alpha(zeros, ones, 1e6, 0.3 * ones, ones, 20)
    large = layer.shrinkage_alpha(zeros, ones, 1e6, 0.3 * ones, ones, 2000)
    assert (large > small).all()


def test_effective_count_of_running_mean():
    layer = _layer(momentum=0.1)
    source, target = torch.randn(48, 64), torch.randn(48, 64) + 2.0
    for _ in range(400):
        layer.forward_pair(source, target)
    # Steady state of Var(EMA mean) / var = m / (b (2 - m)).
    assert layer.target_inv_count.item() == pytest.approx(0.1 / (48 * 1.9), rel=1e-4)
    # A clear shift (2 standard deviations) is corrected with target statistics.
    assert (layer.alpha() > 0.95).all()


def test_no_shift_keeps_source_statistics():
    torch.manual_seed(0)
    layer = _layer()
    for _ in range(50):
        layer.forward_pair(torch.randn(48, 64), torch.randn(48, 64))
    layer.eval()
    assert layer.alpha().mean() < 0.3
    out = layer(torch.randn(10, 64), "target")
    assert out.shape == (10, 64) and torch.isfinite(out).all()


def test_recalibrated_statistics_use_trial_count():
    layer = _layer()
    layer.forward_pair(torch.randn(48, 64), torch.randn(48, 64))
    layer.set_target_statistics(torch.full((64,), 3.0), torch.ones(64), count=288)
    assert layer.target_inv_count.item() == pytest.approx(1 / 288)
    with pytest.raises(ValueError):
        layer.set_target_statistics(torch.zeros(64), torch.ones(64))


def test_gradients_reach_features_but_no_alpha_parameter():
    layer = _layer()
    source = torch.randn(48, 64, requires_grad=True)
    target = torch.randn(48, 64, requires_grad=True)
    s, t = layer.forward_pair(source, target)
    (s.square().mean() + t.square().mean()).backward()
    assert target.grad is not None and source.grad is not None
    assert not any("alpha" in name for name, _ in layer.named_parameters())


def test_learned_mode_state_is_unchanged():
    layer = _layer(mode="learned")
    assert set(layer.state_dict()) == {
        "alpha_logit", "weight", "bias",
        "source_mean", "source_var", "target_mean", "target_var",
    }
    with pytest.raises(ValueError):
        _layer(mode="other")
