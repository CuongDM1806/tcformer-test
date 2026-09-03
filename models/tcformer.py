"""
Model Name: TCFormer - Temporal Convolutional Transformer for EEG-Based Motor Imagery Decoding

Citation:
H. Altaheri, F. Karray, and A.H. Karimi (2025). Temporal Convolutional 
Transformer for EEG Based Motor Imagery Decoding. *Scientific Reports*.

Repository:
Original implementation available at: https://github.com/altaheri/TCFormer

Note:
All default hyperparameters and configurations are consistent with those reported 
in the published paper.
"""

# Core Libraries
import torch
from torch import nn, Tensor

# Utility Libraries
from einops import rearrange
from einops.layers.torch import Rearrange

# Local application-specific imports
from .classification_module import ClassificationModule
from .modules import CausalConv1d, Conv1dWithConstraint
from .channel_group_attention import ChannelGroupAttention
from utils.weight_initialization import glorot_weight_zero_bias
from utils.latency  import measure_latency


# ------------------------------------------------------------------------------- #
class MultiKernelConvBlock(nn.Module):
    """
    Multi-Kernel Convolution Block for EEG Feature Extraction.

    This block applies multiple temporal convolutions with different kernel sizes,
    followed by channel-wise and temporal processing with optional group attention.
    """
    def __init__(
        self,
        n_channels: int,
        temp_kernel_lengths: tuple = (20, 32, 64),
        F1: int = 32,
        D: int = 2,
        pool_length_1: int = 8,
        pool_length_2: int = 7,
        dropout: float = 0.4,
        d_group: int = 16,
        use_group_attn: bool = True,
    ):
        super().__init__()

        # --- 1. one temporal conv per kernel -----------------
        self.rearrange = Rearrange("b c seq -> b 1 c seq")
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad2d((k//2-1, k//2, 0, 0) if k % 2 == 0 else (k//2, k//2, 0, 0), 0),
                nn.Conv2d(1, F1, (1, k), bias=False),
                nn.BatchNorm2d(F1),
                # nn.ELU() 
            )
            for k in temp_kernel_lengths
        ])
            
        # --- 2. shared processing after concatenation --------
        n_groups = len(temp_kernel_lengths)
        self.d_model = d_group * n_groups
        
        # Channel Reduction Stage 1: Grouped Pointwise (1×1) Conv (F1 * n_groups → d_model)
            # Channel Reduction Stage 1 takes more time and did not enhance performance,
            # so we set it as always False.
        self.use_channel_reduction_1  = False
        # self.use_channel_reduction_1  = (self.d_model != F1 * n_groups)
        if self.use_channel_reduction_1:
            self.channel_reduction_1 = nn.Sequential(
                    nn.Conv2d(F1 * n_groups, self.d_model, (1, 1), bias=False, groups=n_groups),
                    nn.BatchNorm2d(self.d_model),
                )
            
        # Depth-wise convolution across EEG channels
        F2 = self.d_model * D if self.use_channel_reduction_1 else F1 * n_groups * D 
        self.channel_DW_conv = nn.Sequential(
            nn.Conv2d(F1 * n_groups, F2, (n_channels, 1), bias=False, groups=F1 * n_groups),
            nn.BatchNorm2d(F2),
            nn.ELU(),
        )
        self.pool1 = nn.AvgPool2d((1, pool_length_1))
        self.drop1 = nn.Dropout(dropout)
        
        # Channel Reduction Stage 2: Grouped Pointwise (1×1) Conv (F2 → d_model)
        self.use_channel_reduction_2 = (self.d_model != F2)
        if self.use_channel_reduction_2:
            self.channel_reduction_2 = nn.Sequential(
                    nn.Conv2d(F2, self.d_model, (1, 1), bias=False, groups=n_groups),
                    nn.BatchNorm2d(self.d_model),
                )

        # Grouped temporal convolution (1 × 16) per group
        self.temporal_conv_2 = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, (1, 16), padding='same',
                       bias=False, groups=n_groups),
            nn.BatchNorm2d(self.d_model),
            nn.ELU(),
        )

        # Enable grouped attention only if multiple groups are used (two temp kernels or more)
        self.use_group_attn = False if n_groups == 1 else use_group_attn
        if self.use_group_attn:
            self.group_attn = ChannelGroupAttention(
                in_channels=self.d_model,
                num_groups=n_groups, 
            )
        
        self.pool2 = nn.AvgPool2d((1, pool_length_2))
        self.drop2 = nn.Dropout(dropout)

        # Initialize weights
        glorot_weight_zero_bias(self)

    def forward(self, x):
        # --- 1. one temporal conv per kernel -----------------
        x = self.rearrange(x)         # (B, 1, C, T)
        feats = [conv(x) for conv in self.temporal_convs] # list of (B, F1, C, T')
        x = torch.cat(feats, dim=1)   # concat on channel dim # [B, F1 x n_groups, C, T]

        # --- 2. shared processing after concatenation --------
        # Channel Reduction Stage 1: (F1 * n_groups → d_model)
        if self.use_channel_reduction_1:
            x = self.channel_reduction_1(x)
        
        # EEG channel depth-wise conv
        x = self.channel_DW_conv(x)                         
        x = self.pool1(x)                                # temporal pooling
        x = self.drop1(x)                                # dropout

        # Channel Reduction Stage 2: (F2 → d_model)
        if self.use_channel_reduction_2:
            x = self.channel_reduction_2(x)
       
        # Grouped Temporal Convolution (1 × 16) applied independently to each group
        x = self.temporal_conv_2(x)                      
        
        # Group attention (optional) 
        if self.use_group_attn:        
            x = x + self.group_attn(x)   # Residual connection 
        
        x = self.pool2(x)                                # temporal pooling
        x = self.drop2(x)                                # dropout
            
        return x.squeeze(2)
# ------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------- #
class TCNBlock(nn.Module):
    def __init__(self, kernel_length: int = 4, n_filters: int = 32, dilation: int = 1,
                 n_groups: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv1 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation, groups=n_groups)
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.nonlinearity1 = nn.ELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(n_filters, n_filters, kernel_size=kernel_length,
                                  dilation=dilation, groups=n_groups)
        self.bn2 = nn.BatchNorm1d(n_filters)
        self.nonlinearity2 = nn.ELU()
        self.drop2 = nn.Dropout(dropout)

        self.nonlinearity3 = nn.ELU()

        nn.init.constant_(self.conv1.bias, 0.0)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, input):
        x = self.drop1(self.nonlinearity1(self.bn1(self.conv1(input))))
        x = self.drop2(self.nonlinearity2(self.bn2(self.conv2(x))))
        x = self.nonlinearity3(input + x)
        return x

class TCN(nn.Module):
    def __init__(self, depth: int = 2, kernel_length: int = 4, n_filters: int = 32,
                 n_groups: int = 1, dropout: float = 0.3):
        super(TCN, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            self.blocks.append(TCNBlock(kernel_length, n_filters, dilation, n_groups, dropout))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class ClassificationHead(nn.Module):
    """
    Maps TCN features to class logits and optionally averages across groups.
    Expected input shape: (batch, d_model, 1)   ← after time-step selection.
    Output shape:        (batch, n_classes)
    """
    def __init__(
        self,
        d_features: int,
        n_groups: int,
        n_classes: int,
        kernel_size: int = 1,
        max_norm: float = 0.25,
    ):
        super().__init__()
        self.n_groups   = n_groups
        self.n_classes  = n_classes

        # self.drop = nn.Dropout(0.3)

        # point-wise (1 × 1) grouped conv = class projection per group
        self.linear = Conv1dWithConstraint(
            in_channels=d_features,
            out_channels=n_classes * n_groups,
            kernel_size=kernel_size,
            groups=n_groups,
            max_norm=max_norm,
        )
        # Zero initialization reproduces the previous equal averaging at the
        # start of training; softmax then learns each group's contribution.
        self.group_weights = nn.Parameter(torch.zeros(n_groups))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_model, Tc)  →  logits: (B, n_classes)
        """
        # (B, n_classes*n_groups, 1) → squeeze last dim
        # x = self.drop(x) 
        x = self.linear(x).squeeze(-1)

        # (B, n_groups, n_classes) -> learned weighted sum over groups
        x = x.view(x.size(0), self.n_groups, self.n_classes)
        weights = torch.softmax(self.group_weights, dim=0)
        x = (x * weights.view(1, self.n_groups, 1)).sum(dim=1)
        return x

class TCNHead(nn.Module):
    def __init__(self, d_features: int = 64, n_groups: int = 1, tcn_depth: int = 2, 
                 kernel_length: int = 4,  dropout_tcn: float = 0.3, n_classes: int = 4):
        super().__init__()
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.tcn = TCN(tcn_depth, kernel_length, d_features, n_groups, dropout_tcn)

        # self.linear = Conv1dWithConstraint(d_model, n_classes*n_groups, kernel_size=1, 
        #                                         groups=n_groups, max_norm=0.25)   
        self.classifier = ClassificationHead(
            d_features=d_features,
            n_groups=n_groups,
            n_classes=n_classes,
        )     
    def extract_temporal_features(self, x):
        """Return the complete TCN sequence before temporal aggregation."""
        return self.tcn(x)

    @staticmethod
    def pool_temporal_features(x):
        """Preserve both the causal final state and whole-trial information."""
        return 0.5 * (x[:, :, -1] + x.mean(dim=-1))

    def extract_features(self, x):
        x = self.extract_temporal_features(x)
        return self.pool_temporal_features(x)

    def classify_features(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return self.classifier(x)

    def forward(self, x):
        return self.classify_features(self.extract_features(x))
# ------------------------------------------------------------------------------- #
    
# ------------------------------------------------------------------------------- #
#  Rotary positional embedding utilities
# Adapted from GPT‑NeoX & LLaMA implementations
def _build_rotary_cache(head_dim: int, seq_len: int, device: torch.device):
    """Return cos & sin tensors of shape (seq_len, head_dim)."""
    theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    seq_idx = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(seq_idx, theta)                 # (seq, dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)             # duplicate for even/odd
    cos, sin = emb.cos(), emb.sin()
    return cos, sin                                     # each: (seq, head_dim)

def _rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):  # q/k: (B, h, T, d)
    def _rotate(x):                                        # half rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)
    q_out = (q * cos) + (_rotate(q) * sin)
    k_out = (k * cos) + (_rotate(k) * sin)
    return q_out, k_out

# ---------------------------------------------------------
#  Grouped‑Query Self‑Attention (GQA) with RoPE
class _GQAttention(nn.Module):
    """Grouped‑Query Attention (num_q_heads >= num_kv_heads)."""
    def __init__(self, d_model: int, num_q_heads: int, num_kv_heads: int, dropout: float = 0.3):
        super().__init__()
        assert d_model % num_q_heads == 0, "d_model must divide num_q_heads"
        assert num_q_heads % num_kv_heads == 0, "q_heads must be multiple of kv_heads"
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.scale = self.head_dim ** -0.5
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(d_model, 2 * num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        _xavier_zero_bias(self)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:  # x (B,T,C)
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim).transpose(1, 2)  # (B, hq, T, d)
        kv = self.kv_proj(x)
        kv = kv.view(B, T, self.num_kv_heads, 2, self.head_dim)
        k, v = kv[..., 0, :].transpose(1, 2), kv[..., 1, :].transpose(1, 2)  # each (B, hk, T, d)
        # replicate k/v across groups
        repeat_factor = self.num_q_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat_factor, dim=1)
        v = v.repeat_interleave(repeat_factor, dim=1)
        # Rotary positional embedding
        q, k = _rope(q, k, cos[:T, :], sin[:T, :])
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B,h,T,T)
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        out = attn @ v                                  # (B,h,T,d)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)

# ---------------------------------------------------------
# from timm.models.layers import DropPath  # for stochastic depth
class DropPath(nn.Module):
    """Implements stochastic depth / DropPath
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    implementation from https://github.com/huggingface/pytorch-image-models/blob/a6e8598aaf90261402f3e9e9a3f12eac81356e9d/timm/models/layers/drop.py#L140
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # shape: (batch_size, 1, 1, ..., 1)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

# ---------------------------------------------------------
#  Transformer encoder block (Pre‑LN, GQA, MLP‑GEGLU)
class _TransformerBlock(nn.Module):
    def __init__(self, d_model: int, q_heads: int, kv_heads: int, mlp_ratio: int = 2, dropout=0.4, drop_path_rate=0.25):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _GQAttention(d_model, q_heads, kv_heads, dropout)
        #self.drop_path = nn.Dropout(dropout)
        self.drop_path   = DropPath(drop_path_rate)  # for stochastic depth
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),    # expands per-token features
            nn.GELU(),                                      # non-linearity
            nn.Linear(mlp_ratio * d_model, d_model),    # compresses back to d_model
            nn.Dropout(dropout),                            # dropout# regularisation
        )
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), cos, sin))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _SelectiveSSMMixer(nn.Module):
    """Small, dependency-free Mamba-style selective state-space mixer.

    The recurrent state is maintained per feature channel.  Input-dependent
    step sizes and B/C projections select what is written to and read from the
    state, while the diagonal negative A parameter keeps the recurrence stable.
    """

    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 3):
        super().__init__()
        if d_state < 1 or d_conv < 1:
            raise ValueError("mamba_d_state and mamba_d_conv must be positive.")
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.local_conv = nn.Conv1d(
            d_model, d_model, d_conv, groups=d_model, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(d_model, d_model + 2 * d_state, bias=False)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .repeat(d_model, 1)
        )
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        values, gate = self.in_proj(x).chunk(2, dim=-1)
        values = self.local_conv(values.transpose(1, 2))[..., :length]
        values = torch.nn.functional.silu(values.transpose(1, 2))

        delta_raw, B, C = torch.split(
            self.x_proj(values), [self.d_model, self.d_state, self.d_state], dim=-1
        )
        delta = torch.nn.functional.softplus(delta_raw).clamp(max=1.0)
        A = -torch.exp(self.A_log).to(dtype=x.dtype)
        state = x.new_zeros(batch, self.d_model, self.d_state)
        outputs = []
        for step in range(length):
            dt = delta[:, step, :].unsqueeze(-1)
            decay = torch.exp(dt * A.unsqueeze(0))
            state = (
                decay * state
                + dt
                * B[:, step, :].unsqueeze(1)
                * values[:, step, :].unsqueeze(-1)
            )
            readout = (state * C[:, step, :].unsqueeze(1)).sum(dim=-1)
            outputs.append(readout + self.D * values[:, step, :])

        y = torch.stack(outputs, dim=1)
        y = y * torch.nn.functional.silu(gate)
        return self.out_proj(y)


class _SelectiveSSMBlock(nn.Module):
    """Pre-norm residual selective SSM, optionally scanning time backwards."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv: int = 3,
        dropout: float = 0.4,
        drop_path_rate: float = 0.0,
        reverse: bool = False,
    ):
        super().__init__()
        self.reverse = reverse
        self.norm = nn.LayerNorm(d_model)
        self.mixer = _SelectiveSSMMixer(d_model, d_state, d_conv)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: Tensor, cos: Tensor = None, sin: Tensor = None) -> Tensor:
        residual = x
        mixed = self.norm(x)
        if self.reverse:
            mixed = torch.flip(mixed, dims=(1,))
        mixed = self.mixer(mixed)
        if self.reverse:
            mixed = torch.flip(mixed, dims=(1,))
        return residual + self.drop_path(self.dropout(mixed))
        
#   helper
def _xavier_zero_bias(module: nn.Module) -> None:
    """Apply Xavier‑uniform + zero bias to every conv/linear inside *module*."""
    for m in module.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
# ------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------- #
class TCFormerModule(nn.Module):
    def __init__(self,                 
            n_channels: int ,
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
            sequence_block_types=None,
            mamba_d_state: int = 8,
            mamba_d_conv: int = 3,
        ):
        super().__init__()
        self.n_classes = n_classes
        self.n_groups = len(temp_kernel_lengths)
        self.d_model = d_group*self.n_groups
        self.feature_dim = d_group * (self.n_groups + 1)

        self.rearrange = Rearrange("b c seq -> b seq c")

        self.conv_block = MultiKernelConvBlock(n_channels, temp_kernel_lengths, F1, D, 
                                               pool_length_1, pool_length_2, dropout_conv, 
                                               d_group, use_group_attn)
        self.mix = nn.Sequential(
            nn.Conv1d(
                in_channels=self.d_model,
                out_channels=self.d_model,
                kernel_size=1,              # across channels only
                groups=1, bias=False),
            nn.BatchNorm1d(self.d_model),
            nn.SiLU()
        )

        # linearly increasing drop path rates from 0 to drop_path_max for deeper layers
        # drop_rates = torch.linspace(0.0, drop_path_max, trans_depth)
        # Quadratically increasing drop path rates from 0 to drop_path_max for deeper layers
        drop_rates = torch.linspace(0, 1, trans_depth) ** 2 * drop_path_max

        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)
        if sequence_block_types is None:
            sequence_block_types = ["transformer"] * trans_depth
        if len(sequence_block_types) != trans_depth:
            raise ValueError("sequence_block_types length must equal trans_depth.")
        valid_block_types = {"transformer", "mamba_forward", "mamba_backward"}
        unknown = set(sequence_block_types) - valid_block_types
        if unknown:
            raise ValueError(f"Unknown sequence block types: {sorted(unknown)}")
        self.sequence_block_types = tuple(sequence_block_types)
        self.head_dim = self.d_model // q_heads
        self.transformer = nn.ModuleList()
        for i, block_type in enumerate(self.sequence_block_types):
            if block_type == "transformer":
                block = _TransformerBlock(
                    self.d_model, q_heads, kv_heads, dropout=trans_dropout,
                    drop_path_rate=drop_rates[i].item()
                )
            else:
                block = _SelectiveSSMBlock(
                    self.d_model,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    dropout=trans_dropout,
                    drop_path_rate=drop_rates[i].item(),
                    reverse=block_type == "mamba_backward",
                )
            self.transformer.append(block)

        self.reduce = nn.Sequential(
            Rearrange("b t c -> b c t"),            # 1. rearrange for Conv1d over channels
            nn.Conv1d(in_channels=self.d_model,     # 2. 1x1 conv over channel dim
                out_channels=d_group,
                kernel_size=1,              
                groups=1, bias=False),
            nn.BatchNorm1d(d_group),
            nn.SiLU(),
        )

        self.tcn_head = TCNHead(self.feature_dim, (self.n_groups+1), tcn_depth,
                                kernel_length_tcn, dropout_tcn, n_classes)

        # # Kaiming (He) init is recommended for Conv layers that precede SiLU / ReLU
        # nn.init.kaiming_normal_(self.reduce[0].weight, nonlinearity="linear") 

    def extract_temporal_features(self, x):  # x: [B, C_electrodes, T]
        conv_features = self.conv_block(x)         
        _, _, T = conv_features.shape

        tokens = self.rearrange(self.mix(conv_features)) 
        cos, sin = self._rotary_cache(T , tokens.device)
        for blk in self.transformer:
            tokens = blk(tokens, cos, sin)
        tran_features = self.reduce(tokens)

        features = torch.cat((conv_features, tran_features), dim=1) 
        return self.tcn_head.extract_temporal_features(features)

    def extract_features(self, x):
        temporal_features = self.extract_temporal_features(x)
        return self.tcn_head.pool_temporal_features(temporal_features)

    def classify_features(self, features):
        return self.tcn_head.classify_features(features)

    def forward(self, x):
        return self.classify_features(self.extract_features(x))
        
    def _rotary_cache(self, seq_len: int, device: torch.device):
        """Build (or reuse) RoPE caches for the current sequence length."""
        head_dim = self.head_dim  # use per-head dimension, not d_model
        if (self._cos is None) or (self._cos.shape[0] < seq_len):
            cos, sin = _build_rotary_cache(head_dim, seq_len, device)
            self._cos, self._sin = cos.to(device), sin.to(device)
        return self._cos, self._sin

class TCFormer(ClassificationModule):
    def __init__(self,
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
            sequence_block_types=None,
            mamba_d_state: int = 8,
            mamba_d_conv: int = 3,
            **kwargs
        ):
        model = TCFormerModule(
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
            q_heads = q_heads, 
            kv_heads = kv_heads, 
            trans_depth = trans_depth,
            trans_dropout = trans_dropout,
            sequence_block_types=sequence_block_types,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
        )
        super().__init__(model, n_classes, **kwargs)
    
    @staticmethod
    def benchmark(input_shape, device="cuda:0", warmup=100, runs=500):
        return measure_latency(TCFormer(22, 4), input_shape, device, warmup, runs)
    

if __name__ == "__main__":
    # Example usage: run benchmark with dummy input shape (batch, channels, time)
    C, T = 22, 1000  # adjust as needed
    print(TCFormer.benchmark((1, C, T)))
