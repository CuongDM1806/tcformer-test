"""Self-supervised masked-reconstruction pretraining for TCFormer."""

from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .tcformer import TCFormerModule


class ReconstructionDecoder(nn.Module):
    """Lightweight three-stage temporal decoder for EEG reconstruction."""

    def __init__(self, feature_dim: int, n_channels: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(feature_dim, 128, kernel_size=8, stride=4, padding=2),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=4, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.ConvTranspose1d(64, n_channels, kernel_size=8, stride=4, padding=2),
        )

    def forward(self, features: Tensor, output_length: int) -> Tensor:
        reconstruction = self.decoder(features)
        if reconstruction.size(-1) != output_length:
            reconstruction = F.interpolate(
                reconstruction,
                size=output_length,
                mode="linear",
                align_corners=False,
            )
        return reconstruction


class MaskedReconstructionPretrain(pl.LightningModule):
    """Pretrain a TCFormer encoder using contiguous masked time spans."""

    def __init__(
        self,
        encoder: TCFormerModule,
        n_channels: int,
        mask_ratio: float = 0.4,
        mask_span_samples: int = 25,
        frequency_loss_weight: float = 0.0,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1).")
        if mask_span_samples < 1:
            raise ValueError("mask_span_samples must be positive.")
        if frequency_loss_weight < 0.0:
            raise ValueError("frequency_loss_weight must be non-negative.")

        self.encoder = encoder
        self.decoder = ReconstructionDecoder(encoder.feature_dim, n_channels)
        self.mask_token = nn.Parameter(torch.zeros(1, n_channels, 1))
        self.mask_ratio = float(mask_ratio)
        self.mask_span_samples = int(mask_span_samples)
        self.frequency_loss_weight = float(frequency_loss_weight)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.save_hyperparameters(ignore=["encoder"])

    def _time_span_mask(self, x: Tensor) -> Tensor:
        batch_size, _, timepoints = x.shape
        span = min(self.mask_span_samples, timepoints)
        target_count = max(1, int(round(timepoints * self.mask_ratio)))
        mask = torch.zeros(
            batch_size, 1, timepoints, dtype=torch.bool, device=x.device
        )

        # A small loop keeps every masked region contiguous and guarantees at
        # least the requested coverage even when randomly sampled spans overlap.
        for batch_idx in range(batch_size):
            while mask[batch_idx].sum().item() < target_count:
                start = torch.randint(
                    0,
                    timepoints - span + 1,
                    (),
                    device=x.device,
                ).item()
                mask[batch_idx, :, start : start + span] = True
        return mask

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mask = self._time_span_mask(x)
        masked_x = torch.where(mask, self.mask_token.expand_as(x), x)
        temporal_features = self.encoder.extract_temporal_features(masked_x)
        reconstruction = self.decoder(temporal_features, x.size(-1))
        return reconstruction, mask

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        reconstruction, mask = self(x)
        expanded_mask = mask.expand_as(x)
        masked_mse = (reconstruction - x).square()[expanded_mask].mean()

        frequency_loss = x.new_zeros(())
        if self.frequency_loss_weight > 0.0:
            target_spectrum = torch.log1p(torch.fft.rfft(x, dim=-1).abs())
            predicted_spectrum = torch.log1p(
                torch.fft.rfft(reconstruction, dim=-1).abs()
            )
            frequency_loss = F.mse_loss(predicted_spectrum, target_spectrum)

        loss = masked_mse + self.frequency_loss_weight * frequency_loss
        self.log(
            "pretrain_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
        )
        self.log(
            "pretrain_masked_mse",
            masked_mse,
            on_step=False,
            on_epoch=True,
            batch_size=x.size(0),
        )
        if self.frequency_loss_weight > 0.0:
            self.log(
                "pretrain_frequency_loss",
                frequency_loss,
                on_step=False,
                on_epoch=True,
                batch_size=x.size(0),
            )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def save_encoder(self, path, **metadata) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder_state_dict": {
                    key: value.detach().cpu()
                    for key, value in self.encoder.state_dict().items()
                },
                "metadata": metadata,
            },
            path,
        )
