import copy

import torch

from models.tcformer import _SelectiveSSMMixer


def _make_pair(d_model=48, d_state=8, d_conv=3):
    torch.manual_seed(7)
    reference = _SelectiveSSMMixer(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        scan_mode="sequential",
    )
    hoisted = copy.deepcopy(reference)
    hoisted.scan_mode = "hoisted"
    hoisted.shift_conv = True
    return reference, hoisted


def test_hoisted_scan_matches_reference_forward_and_backward():
    reference, hoisted = _make_pair()
    x_reference = torch.randn(4, 20, 48, requires_grad=True)
    x_hoisted = x_reference.detach().clone().requires_grad_(True)

    y_reference = reference(x_reference)
    y_hoisted = hoisted(x_hoisted)
    torch.testing.assert_close(y_hoisted, y_reference, rtol=1e-5, atol=1e-6)

    y_reference.square().mean().backward()
    y_hoisted.square().mean().backward()
    torch.testing.assert_close(
        x_hoisted.grad, x_reference.grad, rtol=1e-4, atol=1e-6
    )
    for (name, parameter_reference), (_, parameter_hoisted) in zip(
        reference.named_parameters(), hoisted.named_parameters()
    ):
        torch.testing.assert_close(
            parameter_hoisted.grad,
            parameter_reference.grad,
            rtol=1e-4,
            atol=1e-6,
            msg=name,
        )


def test_hoisted_scan_handles_single_step_sequence():
    reference, hoisted = _make_pair()
    x = torch.randn(2, 1, 48)
    torch.testing.assert_close(hoisted(x), reference(x), rtol=1e-6, atol=1e-7)
