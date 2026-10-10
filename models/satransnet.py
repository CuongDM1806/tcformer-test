"""SATrans-Net (Scientific Reports 2025, doi:10.1038/s41598-025-30806-8).

Network code copied from the authors' code/model.py
(https://github.com/Jasmin-Tianhua/EEG-research_SATrans-Net, commit eaf11e6)
with one equivalence-preserving change taken from the fork used in this project:
the released attention created its q/k LayerNorm lazily during the first
forward pass, after the optimizer was built, so its affine parameters were
never trained; the parameter-free ``F.layer_norm`` below computes the same
function. ``SATransNetBaseline`` wraps ``EEGTransformer`` for the LOSO
pipeline with the authors' settings (heads 8, depth 6, emb 40, dropout 0.25,
AdamW lr 1e-3, betas (0.5, 0.999), batch 72, S&R augmentation).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.backends import cudnn

cudnn.benchmark = False
cudnn.deterministic = True


def numberClassChannel(database_type):
    """Return dataset dimensions without importing visualization utilities."""
    # Added for this repository: an explicit (n_classes, n_channels) pair,
    # used for datasets the released code does not list (Zhou2016).
    if isinstance(database_type, tuple):
        return database_type
    if database_type == "A":
        return 4, 22
    if database_type == "B":
        return 2, 3
    raise ValueError(f"Unsupported database_type: {database_type}")

class PatchEmbeddingCNN(nn.Module):
    def __init__(self, f1=8, kernel_size=64, D=2, pooling_size1=8, pooling_size2=8, dropout_rate=0.3, number_channel=22, emb_size=40):
        super().__init__()
        f2 = D*f1

        self.cnn1 = nn.Sequential(
            # temporal conv kernel size 64=0.25fs
            nn.Conv2d(1, f1, (1, kernel_size), (1, 1), padding='same', bias=False), 
            nn.BatchNorm2d(f1),
            # channel depth-wise conv
            nn.Conv2d(f1, f2, (number_channel, 1), (1, 1), groups=f1, padding='valid', bias=False), # 
            nn.BatchNorm2d(f2),
            nn.ELU(),
            # average pooling 1
            nn.AvgPool2d((1, pooling_size1)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(dropout_rate),
            # spatial conv
            nn.Conv2d(f2, f2, (1, 16), padding='same', bias=False), 
            nn.BatchNorm2d(f2),
            nn.ELU(),

            # average pooling 2 to adjust the length of feature into transformer encoder
            nn.AvgPool2d((1, pooling_size2)),
            nn.Dropout(dropout_rate),         
        )

        self.projection_2 = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )



    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape

        x = self.cnn1(x)
        x = self.projection_2(x)

        return x
    




class Attention(nn.Module):
    def __init__(self, dim, num_heads, topk_ratios=[0.5, 2/3, 3/4, 4/5], dropout=0.3, kernel_size=3, norm_type='layer'):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.topk_ratios = topk_ratios
        self.dropout = dropout

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv1d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv1d(dim * 3, dim * 3, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=dim * 3)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1)
        self.attn_drop = nn.Dropout(dropout)


        self.attn_weights = nn.Parameter(torch.ones(len(topk_ratios)) / len(topk_ratios))

        self.norm_type = norm_type
        if norm_type == 'layer':
            self.norm = None  
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm1d(dim)
        else:
            self.norm = lambda x: F.normalize(x, dim=-1)

    def forward(self, x):
        x = x.permute((0, 2, 1))
        b, c, h = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h -> b head c h', head=self.num_heads)
        k = rearrange(k, 'b (head c) h -> b head c h', head=self.num_heads)
        v = rearrange(v, 'b (head c) h -> b head c h', head=self.num_heads)


        if self.norm_type == 'layer':
            # The released code created LayerNorm modules during the first
            # forward pass, after the optimizer had already been built. Use
            # the equivalent parameter-free operation so checkpoints load
            # cleanly and every model parameter is known at construction.
            q = F.layer_norm(q, q.shape[-1:])
            k = F.layer_norm(k, k.shape[-1:])
        else:
            q = self.norm(q)
            k = self.norm(k)

        _, _, C, _ = q.shape

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        topk_indices = [
            torch.topk(attn, k=int(C * ratio), dim=-1, largest=True)[1]
            for ratio in self.topk_ratios
        ]

        masks = [
            torch.zeros_like(attn, device=x.device).scatter_(-1, index, 1.)
            for index in topk_indices
        ]


        attns = [
            torch.where(mask > 0, attn, torch.full_like(attn, float('-inf'))).softmax(dim=-1)
            for mask in masks
        ]


        if self.training:
            attns = [self.attn_drop(attn) for attn in attns]


        outs = [attn @ v for attn in attns]
        out = sum(out * weight for out, weight in zip(outs, self.attn_weights))

        out = rearrange(out, 'b head c h -> b (head c) h', head=self.num_heads, h=h)
        out = self.project_out(out)
        out = self.attn_drop(out) 
        out = out.permute((0, 2, 1))
        return out

# PointWise FFN
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


#Classification
class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            # nn.Linear(flatten_number, 256),
            nn.Dropout(0.5),
            nn.Linear(flatten_number, n_classes),
           # nn.Softmax(dim=1),
        )

    def forward(self, x):
        out = self.fc(x)
        
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        
        out = self.layernorm(self.drop(res)+x_input)
        return out

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                Attention(emb_size, num_heads), 
                ), emb_size, drop_p),
     
            ResidualAdd(nn.Sequential(
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                ), emb_size, drop_p)
            
            )    
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])




class BranchEEGNetTransformer(nn.Sequential):
    def __init__(self, heads=4, 
                 depth=6, 
                 emb_size=40, 
                 number_channel=22,
                 f1 = 20,
                 kernel_size = 64,
                 D = 2,
                 pooling_size1 = 8,
                 pooling_size2 = 8,
                 dropout_rate = 0.3,
                 **kwargs):
        super().__init__(
            PatchEmbeddingCNN(f1=f1, 
                                 kernel_size=kernel_size,
                                 D=D, 
                                 pooling_size1=pooling_size1, 
                                 pooling_size2=pooling_size2, 
                                 dropout_rate=dropout_rate,
                                 number_channel=number_channel,
                                 emb_size=emb_size),
#             TransformerEncoder(heads, depth, emb_size),
        )



    

        
class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
    def forward(self, x): # x-> [batch, embedding, length]
        x = x + self.encoding[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)        
        
   
    
class EEGTransformer(nn.Module):
    def __init__(self, heads=4, 
                 emb_size=40,
                 depth=6, 
                 database_type='A', 
                 eeg1_f1=20,
                 eeg1_kernel_size=64,
                 eeg1_D=2,
                 eeg1_pooling_size1=8,
                 eeg1_pooling_size2=8,
                 eeg1_dropout_rate=0.3,
                 eeg1_number_channel=22,
                 flatten_eeg1=600,
                 **kwargs):
        super().__init__()
        self.number_class, self.number_channel = numberClassChannel(database_type)
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1
        self.flatten = nn.Flatten()
        
        self.cnn = BranchEEGNetTransformer(heads, depth, emb_size, number_channel=self.number_channel,
                                              f1=eeg1_f1,
                                              kernel_size=eeg1_kernel_size,
                                              D=eeg1_D,
                                              pooling_size1=eeg1_pooling_size1,
                                              pooling_size2=eeg1_pooling_size2,
                                              dropout_rate=eeg1_dropout_rate)
        

        self.position = PositioinalEncoding(emb_size, 100, dropout=0.1)
        self.trans = TransformerEncoder(heads, depth, emb_size)


        
         
        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(self.flatten_eeg1, self.number_class) # FLATTEN_EEGNet + FLATTEN_cnn_module

    def forward(self, x):


        cnn = self.cnn(x)
        # add label 
        cnn = cnn * math.sqrt(self.emb_size)

        features = self.position(cnn)
        features = self.trans(features)

        features = cnn + features

        # features = self.cnn_output(features)
        out = self.classification(self.flatten(features))

        return features, out



from .classification_module import ClassificationModule


class SATransNetBaseline(ClassificationModule):
    """Authors' SATrans-Net inside the shared LOSO training/evaluation pipeline."""

    def __init__(self, n_channels: int, n_classes: int, n_samples: int = 1000,
                 heads: int = 8, emb_size: int = 40, depth: int = 6,
                 eeg1_f1: int = 20, eeg1_kernel_size: int = 64, eeg1_D: int = 2,
                 eeg1_pooling_size1: int = 8, eeg1_pooling_size2: int = 8,
                 eeg1_dropout_rate: float = 0.25, **kwargs):
        # Released dataset codes for BCI IV-2a/2b; any other dataset passes
        # its own (n_classes, n_channels) to the same network.
        database_type = {(4, 22): "A", (2, 3): "B"}.get(
            (n_classes, n_channels), (n_classes, n_channels))
        n_tokens = n_samples // (eeg1_pooling_size1 * eeg1_pooling_size2)
        model = EEGTransformer(
            heads=heads, emb_size=emb_size, depth=depth,
            database_type=database_type,
            eeg1_f1=eeg1_f1, eeg1_kernel_size=eeg1_kernel_size, eeg1_D=eeg1_D,
            eeg1_pooling_size1=eeg1_pooling_size1,
            eeg1_pooling_size2=eeg1_pooling_size2,
            eeg1_dropout_rate=eeg1_dropout_rate,
            eeg1_number_channel=n_channels,
            flatten_eeg1=emb_size * n_tokens,
        )
        super().__init__(model, n_classes, **kwargs)

    def forward(self, x):
        # Pipeline tensors are [B, C, T]; the released network expects
        # [B, 1, C, T] and returns (features, logits).
        _, logits = self.model(x.unsqueeze(1))
        return logits
