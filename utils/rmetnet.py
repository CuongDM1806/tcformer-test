"""RMETNet, from the authors' public code.

Y. Zhao, D. He, F. Ren, Q. Xia, L. Xu, G. Xie, "RMETNet: A cross-subject motor
imagery EEG signal classification model based on TSLANet and Riemannian
geometry features", PLOS ONE 21(4), 2026.

Source: https://github.com/rokanfeermecer486/RMETNet (commit 1a83f9a)
  * Net / ConvNet / RieInception / classifier: gcn/baseshallow/SLANetMultiLayerRie.py
  * MMDLoss: tools/MMDLoss.py
  * covariance and Riemannian alignment: tools/Rieman.py, bciciv2a/LoaderBCICIV2a.py
The repository imports ``gcn.TSLANet.TSLANet_layer`` and
``gcn.models.layers.Conv2dWithConstraint`` but does not contain them. They are
taken from the TSLANet authors' code (https://github.com/emadeldeen24/TSLANet,
commit ca0e884, Classification/TSLANet_classification.py), with the global
``args.ASB`` / ``args.ICB`` flags passed as the ``a`` / ``i`` arguments that
RMETNet uses (adaptive filter on, as in the TSLANet default), and the usual
max-norm constrained convolution (braindecode).

Interface changes only: input size, channel and class counts are arguments
instead of the hard-coded (22, 1000) / 4, ``squeeze`` keeps the batch axis, and
timm's DropPath / trunc_normal_ are replaced by identical local versions.

The released Riemannian branch (RieInception: 2x2, 4x4 and 6x6 strided
convolutions) only fits 22x22 matrices, so it is used for BCI IV-2a as
released. For other channel counts the paper's dataset-specific multi-scale
rule is used (Table 1 of the paper: four filters per branch, kernels 1x1 and
2x2 for BCI IV-2b, 1x1, 3x3 and 5x5 otherwise), with the released 64-unit
projection.
"""
import numpy as np
import torch
import torch.nn as nn
from pyriemann.utils.mean import mean_riemann
from scipy.linalg import logm


# --------------------------------------------------------------------------
# timm.models.layers :: DropPath, trunc_normal_ (same behaviour)
# --------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


def trunc_normal_(tensor, std=0.02):
    return nn.init.trunc_normal_(tensor, std=std, a=-2.0, b=2.0)


# --------------------------------------------------------------------------
# braindecode-style max-norm convolution (gcn.models.layers, missing upstream)
# --------------------------------------------------------------------------
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


# --------------------------------------------------------------------------
# TSLANet (gcn.TSLANet, missing upstream): Classification/TSLANet_classification.py
# --------------------------------------------------------------------------
class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim, adaptive_filter=True):
        super().__init__()
        self.adaptive_filter = adaptive_filter
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))  # * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape

        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight

        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class TSLANet_layer(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 a=True, i=True):
        super().__init__()
        self.ASB, self.ICB = a, i
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # Check if both ASB and ICB are true
        if self.ICB and self.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is true
        elif self.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif self.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x


# --------------------------------------------------------------------------
# gcn/baseshallow/SLANetMultiLayerRie.py
# --------------------------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(
            self, num_channels: int, sampling_rate: int, num_samples: int,
            dropout_rate: float = 0.5
    ):
        super(ConvNet, self).__init__()

        kernel_size = int(sampling_rate * 0.12)

        pooling_size = 0.3
        hop_size = 0.7
        pooling_kernel_size = int(sampling_rate * pooling_size)
        pooling_stride_size = int(sampling_rate * pooling_size * (1 - hop_size))

        depth = 24
        self.temporal_conv = Conv2dWithConstraint(
            1, depth // 2, kernel_size=[1, kernel_size], padding="same", max_norm=2.0
        )
        self.temporal_conv1 = Conv2dWithConstraint(
            depth // 2, depth, kernel_size=[1, kernel_size], padding="same", max_norm=2.0
        )
        self.spatial_conv = Conv2dWithConstraint(
            depth, depth, kernel_size=[num_channels, 1], padding="valid", max_norm=2.0
        )

        self.bn = nn.BatchNorm2d(depth)
        self.avg_pool = nn.AvgPool2d(
            kernel_size=(1, pooling_kernel_size), stride=(1, pooling_stride_size)
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        # 1032 = 24 x 43 for (22, 1000) in the released code
        n_pooled = (num_samples - pooling_kernel_size) // pooling_stride_size + 1
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(depth * n_pooled, 128),
        )

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.temporal_conv1(x)
        x = self.spatial_conv(x)
        x = torch.square(x)
        x = self.avg_pool(x)
        x = torch.log(torch.clamp(x, min=1e-06))
        x = self.bn(x)
        # x = self.dropout(x)
        x = self.linear(x)

        return x


class RieInception(nn.Module):
    """Released Riemannian branch; valid for 22 x 22 inputs (BCI IV-2a)."""

    def __init__(self, ):
        super(RieInception, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 2, (2, 2), stride=(2, 2), bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(2, 4, (2, 2), stride=(2, 2), bias=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(1, 4, (4, 4), stride=(4, 4), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(1, 4, (6, 6), stride=(6, 6), bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.ConstantPad2d((1, 1, 1, 1), 0,),

        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 5 * 5, 64),
        )

    def forward(self, x):
        o = x
        x1 = self.conv1(o)
        x2 = self.conv2(o)
        x3 = self.conv3(o)
        cat = torch.cat((x1, x2, x3), 1)
        out = self.linear(cat)
        return out


class RieMultiScale(nn.Module):
    """Paper Table 1 rule for channel counts the released branch cannot take:
    F3 = 4 max-norm filters per kernel size, 'same' padding except the last
    ('valid'), flattened, concatenated and projected to 64 units."""

    def __init__(self, num_channels, kernel_sizes, filters=4):
        super().__init__()
        last = len(kernel_sizes) - 1
        self.branches = nn.ModuleList([
            Conv2dWithConstraint(1, filters, k, padding="valid" if b == last else "same",
                                 bias=False, max_norm=2.0)
            for b, k in enumerate(kernel_sizes)
        ])
        n_out = sum(filters * (num_channels if b < last else num_channels - k + 1) ** 2
                    for b, k in enumerate(kernel_sizes))
        self.linear = nn.Sequential(nn.Flatten(), nn.Linear(n_out, 64))

    def forward(self, x):
        cat = torch.cat([branch(x).flatten(1) for branch in self.branches], dim=1)
        return self.linear(cat)


class classifier(nn.Module):
    def __init__(self, inputsize, num_classes=4):
        super(classifier, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(inputsize, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self,
                 num_classes: int = 4,
                 num_channels: int = 22,
                 sampling_rate: int = 250,
                 num_samples: int = 1000):
        super(Net, self).__init__()

        self.backbone = ConvNet(
            num_channels=num_channels, sampling_rate=sampling_rate, num_samples=num_samples
        )
        self.layer_norm = nn.LayerNorm(num_channels)

        h = num_channels
        dpr = [x.item() for x in torch.linspace(0, 0.5, 5)]  # stochastic depth decay rule
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(dim=h, drop=0.5, drop_path=dpr[i], a=True, i=False)
            for i in range(1)]
        )
        if num_channels == 22:
            self.riemann_branch = RieInception()
        elif num_channels == 3:
            self.riemann_branch = RieMultiScale(num_channels, kernel_sizes=[1, 2])
        else:
            self.riemann_branch = RieMultiScale(num_channels, kernel_sizes=[1, 3, 5])
        self.classifier = classifier(inputsize=128 + 64, num_classes=num_classes)

    def forward(self, x, source):
        x = x.squeeze(1).transpose(1, 2)
        for blk in self.tsla_blocks:
            x = blk(x)
        x = x.transpose(1, 2).unsqueeze(1)
        out = self.backbone(x)
        sout = source.unsqueeze(1)
        sout = self.riemann_branch(sout)
        feature = torch.cat((out, sout), dim=1)
        out = self.classifier(feature)

        return feature, out


# --------------------------------------------------------------------------
# tools/MMDLoss.py
# --------------------------------------------------------------------------
class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        L2_distance = torch.cdist(total, total, p=2).pow(2)

        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul ** i) for i in range(self.kernel_num)]

        kernels = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernels)

    def forward(self, source, target):
        kernels = self.gaussian_kernel(source, target)
        batch_size_source = source.size(0)
        batch_size_target = target.size(0)
        XX = kernels[:batch_size_source, :batch_size_source]
        YY = kernels[batch_size_source:, batch_size_source:]
        XY = kernels[:batch_size_source, batch_size_source:]
        YX = kernels[batch_size_source:, :batch_size_source]
        loss = (torch.sum(XX) / (batch_size_source ** 2) +
                torch.sum(YY) / (batch_size_target ** 2) -
                torch.sum(XY) / (batch_size_source * batch_size_target) -
                torch.sum(YX) / (batch_size_source * batch_size_target))
        return loss


# --------------------------------------------------------------------------
# tools/Rieman.py :: compute_covariance_matrix, riemannian_alignment
# --------------------------------------------------------------------------
def compute_covariance_matrix(data):
    """Per-trial covariance (np.cov), data: (n_trials, n_channels, n_samples)."""
    return np.array([np.cov(trial) for trial in data])


def common_reference(source_covs, target_covs):
    """Riemannian means of the source and target domains, and their mean."""
    C_source = mean_riemann(source_covs)
    C_target = mean_riemann(target_covs)
    return mean_riemann(np.stack([C_source, C_target], axis=0))


def align(covs, C_common):
    """logm(C_common^{-1} X) for every covariance X, as in riemannian_alignment."""
    return np.array([np.real(logm(np.linalg.solve(C_common, X))) for X in covs])
