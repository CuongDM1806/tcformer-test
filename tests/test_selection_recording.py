import torch

from models.tcformer import TCFormerModule


def _model():
    torch.manual_seed(0)
    return TCFormerModule(
        n_channels=22, n_classes=4, temp_kernel_lengths=(20, 32, 64),
        sequence_block_types=["mamba_forward", "mamba_backward", "mamba_forward"],
        trans_depth=3, mamba_scan_mode="hoisted",
    ).eval()


def test_recording_is_off_by_default_and_does_not_change_outputs():
    model = _model()
    x = torch.randn(3, 22, 1000)
    reference = model(x)
    assert all(raw is None for raw in model.selection_pre_activations())
    model.record_selection(True)
    recorded = model(x)
    raws = model.selection_pre_activations()
    assert torch.allclose(reference, recorded)
    assert len(raws) == 3 and all(r.shape[0] == 3 and r.shape[2] == model.d_model for r in raws)
    model.record_selection(False)
    assert all(raw is None for raw in model.selection_pre_activations())


def test_backward_block_records_in_forward_time_order():
    model = _model()
    tokens = torch.randn(2, 17, model.d_model)
    block = model.transformer[1]
    assert block.reverse
    block.mixer.record_selection = True
    block(tokens)
    recorded = block.mixer.last_delta_raw
    flipped = torch.flip(block.norm(tokens), dims=(1,))
    values, _ = block.mixer.in_proj(flipped).chunk(2, dim=-1)
    values = block.mixer._shifted_causal_conv(values)
    raw = block.mixer.x_proj(torch.nn.functional.silu(values))[..., :model.d_model]
    assert torch.allclose(recorded, torch.flip(raw, dims=(1,)), atol=1e-6)


def test_delta_shift_moves_the_pre_activation_and_is_not_saved():
    model = _model()
    x = torch.randn(2, 22, 1000)
    model.record_selection(True)
    model(x)
    before = model.selection_pre_activations()[0].clone()
    model.ssm_mixers()[0].delta_shift.fill_(0.5)
    model(x)
    assert torch.allclose(model.selection_pre_activations()[0], before + 0.5, atol=1e-5)
    assert not any("delta_shift" in key for key in model.state_dict())
