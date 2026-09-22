import unittest

import torch

from models.lk_block import LKTemporalStage
from models.tcformer import TCFormerModule


def build_model(temporal_mixer="conv"):
    return TCFormerModule(
        n_channels=22,
        n_classes=4,
        F1=32,
        temp_kernel_lengths=(20, 32, 64),
        pool_length_1=8,
        pool_length_2=7,
        D=2,
        dropout_conv=0.4,
        d_group=16,
        tcn_depth=2,
        kernel_length_tcn=4,
        dropout_tcn=0.3,
        use_group_attn=True,
        trans_depth=0,
        sequence_block_types=[],
        temporal_mixer=temporal_mixer,
        lk_kernel=31,
        lk_depth=1,
        lk_expand=2,
        lk_drop_path=0.1,
    )


class LKTemporalTest(unittest.TestCase):
    def test_modes_preserve_shapes(self):
        x = torch.randn(4, 22, 1000)
        temporal_shapes = []
        parameter_counts = {}
        for mode in ("conv", "lk", "conv+lk"):
            model = build_model(mode).eval()
            with torch.no_grad():
                temporal = model.extract_temporal_features(x)
                logits = model(x)
            temporal_shapes.append(tuple(temporal.shape))
            parameter_counts[mode] = sum(p.numel() for p in model.parameters())
            self.assertEqual(tuple(logits.shape), (4, 4))

        self.assertEqual(len(set(temporal_shapes)), 1)
        self.assertLess(parameter_counts["lk"], parameter_counts["conv"])
        self.assertGreater(parameter_counts["conv+lk"], parameter_counts["conv"])

    def test_conv_mode_keeps_baseline_state_keys(self):
        default_model = build_model()
        explicit_conv_model = build_model("conv")
        self.assertEqual(
            set(default_model.state_dict()), set(explicit_conv_model.state_dict())
        )

    def test_group_isolation_and_reparameterization(self):
        torch.manual_seed(0)
        stage = LKTemporalStage(
            48, 3, depth=2, kernel_size=31, dropout=0.0, drop_path_max=0.0
        )
        stage.train()
        for _ in range(3):
            stage(torch.randn(4, 48, 1, 125))

        stage.eval()
        x = torch.randn(4, 48, 1, 125)
        with torch.no_grad():
            reference = stage(x)
            perturbed = x.clone()
            perturbed[:, :16] += 1.0
            changed = stage(perturbed)
        self.assertTrue(torch.equal(reference[:, 16:], changed[:, 16:]))

        stage.reparameterize()
        with torch.no_grad():
            fused = stage(x)
        self.assertTrue(torch.allclose(reference, fused, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
