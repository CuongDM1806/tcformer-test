"""
Subject-adaptive TCFormer variant.

This experimental model keeps the TCFormer backbone intact and adds lightweight
subject-conditioned adapters after each Transformer block. The conditioning
vector is inferred from unlabeled EEG trial statistics, so the model can still
run with the existing dataloaders that return only (x, y).
"""

import torch
from torch import nn, Tensor
from einops.layers.torch import Rearrange

from .classification_module import ClassificationModule
from .tcformer import MultiKernelConvBlock, TCNHead, _TransformerBlock, _build_rotary_cache
from utils.latency import measure_latency


class TrialSubjectEncoder(nn.Module):
    """
    Infers a compact subject/style embedding from raw EEG statistics.

    The features are label-free and fixed-size: per-channel mean/std plus a few
    global amplitude descriptors. This makes the adapter usable in LOSO or
    calibration-light settings without needing explicit subject IDs.
    """
    def __init__(self, n_channels: int, emb_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        stat_dim = 2 * n_channels + 4
        self.encoder = nn.Sequential(
            nn.LayerNorm(stat_dim),
            nn.Linear(stat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        channel_mean = x.mean(dim=-1)
        channel_std = x.std(dim=-1, unbiased=False)
        global_mean = x.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        global_std = x.std(dim=(1, 2), unbiased=False, keepdim=False).unsqueeze(-1)
        log_power = torch.log(x.pow(2).mean(dim=(1, 2), keepdim=False).clamp_min(1e-6)).unsqueeze(-1)
        abs_mean = x.abs().mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        stats = torch.cat(
            [channel_mean, channel_std, global_mean, global_std, log_power, abs_mean],
            dim=-1,
        )
        return self.encoder(stats)


class HyperFiLMAdapter(nn.Module):
    """
    Bottleneck adapter modulated by a subject embedding.

    A small hypernetwork maps the inferred subject embedding to FiLM parameters
    gamma/beta. The residual scale is initialized to zero so the branch starts
    as TCFormer and learns adaptation gradually.
    """
    def __init__(
        self,
        d_model: int,
        subject_emb_dim: int = 32,
        bottleneck_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        bottleneck_dim = max(4, d_model // bottleneck_ratio)
        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, d_model)
        self.drop = nn.Dropout(dropout)
        self.hyper = nn.Sequential(
            nn.Linear(subject_emb_dim, 2 * d_model),
            nn.Tanh(),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, subject_emb: Tensor) -> Tensor:
        gamma, beta = self.hyper(subject_emb).chunk(2, dim=-1)
        delta = self.up(self.act(self.down(self.norm(x))))
        delta = delta * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return x + self.residual_scale * self.drop(delta)


class SubjectAdaptiveTCFormerModule(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        F1: int = 16,
        temp_kernel_lengths=(16, 32, 64),
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        D: int = 2,
        dropout_conv: float = 0.3,
        d_group: int = 16,
        tcn_depth: int = 2,
        kernel_length_tcn: int = 4,
        dropout_tcn: float = 0.3,
        use_group_attn: bool = True,
        kv_heads: int = 4,
        q_heads: int = 8,
        trans_dropout: float = 0.4,
        drop_path_max: float = 0.25,
        trans_depth: int = 5,
        subject_emb_dim: int = 32,
        adapter_bottleneck_ratio: int = 4,
        adapter_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_groups = len(temp_kernel_lengths)
        self.d_model = d_group * self.n_groups

        self.subject_encoder = TrialSubjectEncoder(n_channels, subject_emb_dim)
        self.rearrange = Rearrange("b c seq -> b seq c")

        self.conv_block = MultiKernelConvBlock(
            n_channels,
            temp_kernel_lengths,
            F1,
            D,
            pool_length_1,
            pool_length_2,
            dropout_conv,
            d_group,
            use_group_attn,
        )
        self.mix = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.SiLU(),
        )

        drop_rates = torch.linspace(0, 1, trans_depth) ** 2 * drop_path_max
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)
        self.transformer = nn.ModuleList([
            _TransformerBlock(
                self.d_model,
                q_heads,
                kv_heads,
                dropout=trans_dropout,
                drop_path_rate=drop_rates[i].item(),
            )
            for i in range(trans_depth)
        ])
        self.adapters = nn.ModuleList([
            HyperFiLMAdapter(
                self.d_model,
                subject_emb_dim,
                adapter_bottleneck_ratio,
                adapter_dropout,
            )
            for _ in range(trans_depth)
        ])

        self.reduce = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(self.d_model, d_group, kernel_size=1, groups=1, bias=False),
            nn.BatchNorm1d(d_group),
            nn.SiLU(),
        )

        self.tcn_head = TCNHead(
            d_group * (self.n_groups + 1),
            self.n_groups + 1,
            tcn_depth,
            kernel_length_tcn,
            dropout_tcn,
            n_classes,
        )

    def forward(self, x: Tensor) -> Tensor:
        subject_emb = self.subject_encoder(x)
        conv_features = self.conv_block(x)
        _, _, T = conv_features.shape

        tokens = self.rearrange(self.mix(conv_features))
        cos, sin = self._rotary_cache(T, tokens.device)
        for block, adapter in zip(self.transformer, self.adapters):
            tokens = block(tokens, cos, sin)
            tokens = adapter(tokens, subject_emb)

        tran_features = self.reduce(tokens)
        features = torch.cat((conv_features, tran_features), dim=1)
        return self.tcn_head(features)

    def _rotary_cache(self, seq_len: int, device: torch.device):
        head_dim = self.transformer[0].attn.head_dim
        if (self._cos is None) or (self._cos.shape[0] < seq_len):
            cos, sin = _build_rotary_cache(head_dim, seq_len, device)
            self._cos, self._sin = cos.to(device), sin.to(device)
        return self._cos, self._sin


class SubjectAdaptiveTCFormer(ClassificationModule):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        F1: int = 16,
        temp_kernel_lengths: tuple = (16, 32, 64),
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        D: int = 2,
        dropout_conv: float = 0.3,
        d_group: int = 16,
        tcn_depth: int = 2,
        kernel_length_tcn: int = 4,
        dropout_tcn: float = 0.3,
        use_group_attn: bool = True,
        q_heads: int = 8,
        kv_heads: int = 4,
        trans_depth: int = 5,
        trans_dropout: float = 0.4,
        subject_emb_dim: int = 32,
        adapter_bottleneck_ratio: int = 4,
        adapter_dropout: float = 0.1,
        **kwargs,
    ):
        model = SubjectAdaptiveTCFormerModule(
            n_channels=n_channels,
            n_classes=n_classes,
            F1=F1,
            temp_kernel_lengths=temp_kernel_lengths,
            pool_length_1=pool_length_1,
            pool_length_2=pool_length_2,
            D=D,
            dropout_conv=dropout_conv,
            d_group=d_group,
            tcn_depth=tcn_depth,
            kernel_length_tcn=kernel_length_tcn,
            dropout_tcn=dropout_tcn,
            use_group_attn=use_group_attn,
            q_heads=q_heads,
            kv_heads=kv_heads,
            trans_depth=trans_depth,
            trans_dropout=trans_dropout,
            subject_emb_dim=subject_emb_dim,
            adapter_bottleneck_ratio=adapter_bottleneck_ratio,
            adapter_dropout=adapter_dropout,
        )
        super().__init__(model, n_classes, **kwargs)

    @staticmethod
    def benchmark(input_shape, device="cuda:0", warmup=100, runs=500):
        return measure_latency(SubjectAdaptiveTCFormer(22, 4), input_shape, device, warmup, runs)
