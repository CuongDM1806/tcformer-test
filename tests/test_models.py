"""Model checks on CPU (needs PyTorch): scan equivalence, shapes and parameter counts."""
import pytest

torch = pytest.importorskip("torch")

from ssm_dan import config as cfg                      # noqa: E402
from ssm_dan.data import get_datamodule_cls            # noqa: E402
from ssm_dan.models import get_model_cls               # noqa: E402
from ssm_dan.models.backbone import SelectiveSSMMixer  # noqa: E402

# Trainable parameters (thousands) reported in the paper's efficiency table.
PAPER_PARAMS_K = {("bcic2a", ()): 132.0, ("bcic2a", ("source_only",)): 98.4,
                  ("bcic2b", ()): 128.2, ("bcic2b", ("source_only",)): 94.6,
                  ("zhou2016", ()): 130.4, ("zhou2016", ("source_only",)): 96.8}
WINDOW = {"bcic2a": 1000, "bcic2b": 1000, "zhou2016": 1125, "openbmi": 400}


def build(dataset, variants=()):
    config = cfg.resolve("ssm_dan", dataset, list(variants))
    datamodule_cls = get_datamodule_cls(dataset)
    kwargs = cfg.model_kwargs(config)
    kwargs.update(n_channels=datamodule_cls.channels, n_classes=datamodule_cls.classes)
    return get_model_cls(config["model"])(**kwargs), datamodule_cls


def test_hoisted_scan_matches_sequential_reference():
    torch.manual_seed(0)
    hoisted = SelectiveSSMMixer(48, scan="hoisted")
    sequential = SelectiveSSMMixer(48, scan="sequential")
    sequential.load_state_dict(hoisted.state_dict())
    x = torch.randn(3, 20, 48)
    assert torch.allclose(hoisted(x), sequential(x), atol=1e-6)


@pytest.mark.parametrize("dataset,variants", sorted(PAPER_PARAMS_K))
def test_parameter_count_and_output_shape(dataset, variants):
    model, datamodule_cls = build(dataset, variants)
    count = sum(p.numel() for p in model.parameters())
    assert round(count / 1000, 1) == PAPER_PARAMS_K[(dataset, variants)]
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, datamodule_cls.channels, WINDOW[dataset]))
    assert logits.shape == (2, datamodule_cls.classes)


def test_im_tta_updates_only_batchnorm_affine_parameters():
    model, datamodule_cls = build("bcic2b")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    x = torch.randn(16, datamodule_cls.channels, WINDOW["bcic2b"])
    model.print = lambda *args, **kwargs: None
    model.adapt_to_target([(x, torch.zeros(16, dtype=torch.long))])
    changed = {k for k, v in model.state_dict().items() if k in before and not torch.equal(v, before[k])}
    assert changed, "IM-TTA changed nothing"
    allowed = ("weight", "bias", "target_mean", "target_var")
    for key in changed:
        assert key.endswith(allowed), key
