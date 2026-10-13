"""Kiến trúc WDAN theo She et al. (TNSRE 2023), mục II-D và Table I.

Feature extractor (Table I):
  Input 1xCxT -> CBAM (1x1) -> lọc m dải con -> Depthwise Conv2D (C,1), d bộ lọc mỗi dải
  -> Variance layer (cửa sổ không chồng lấp w) -> FC 64
Classifier: 2 lớp FC + softmax (mục II-D-3).
Domain discriminator: critic Wasserstein, huấn luyện với gradient penalty (WGAN-GP, mục II-D-2).

GIẢ ĐỊNH (bài không nói): số bộ lọc d, bộ lọc dải con là FIR cố định 251 hệ số,
BatchNorm sau depthwise conv, kích thước lớp ẩn của classifier và critic, dropout.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import firwin

# 8-32 Hz chia thành 6 dải 4 Hz không chồng lấp: 8-12, 12-16, ..., 28-32 Hz
SUB_BANDS = [(8 + 4 * i, 12 + 4 * i) for i in range(6)]


class CBAM(nn.Module):
    """CBAM trên input 1xCxT (một feature map). Theo bài: W0, W1 thuộc R^{1x1},
    attention không gian dùng conv n x n với n = 1, cho ra bản đồ Ms thuộc R^{CxT}."""
    def __init__(self, n=1):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))   # Eq. (1)
        self.spatial = nn.Conv2d(2, 1, n, padding=n // 2)                         # Eq. (2)

    def forward(self, f):                           # f: (B, 1, C, T)
        avg = f.mean(dim=(2, 3))                    # (B, 1)
        mx = f.amax(dim=(2, 3))
        f = f * torch.sigmoid(self.mlp(avg) + self.mlp(mx))[..., None, None]
        pooled = torch.cat([f.mean(1, keepdim=True), f.amax(1, keepdim=True)], dim=1)
        return f * torch.sigmoid(self.spatial(pooled))


class SubBandFilter(nn.Module):
    """Lọc mỗi kênh thành m dải con bằng bộ lọc FIR cố định (không học)."""
    def __init__(self, bands=SUB_BANDS, fs=250, numtaps=251):
        super().__init__()
        taps = np.stack([firwin(numtaps, b, pass_zero=False, fs=fs) for b in bands])
        self.register_buffer("weight", torch.tensor(taps, dtype=torch.float32)[:, None, :])
        self.pad = numtaps // 2

    def forward(self, x):                           # x: (B, 1, C, T)
        B, _, C, T = x.shape
        y = F.conv1d(x.reshape(B * C, 1, T), self.weight, padding=self.pad)
        return y.reshape(B, C, -1, T).permute(0, 2, 1, 3)   # (B, m, C, T)


class VarianceLayer(nn.Module):
    """Phương sai trên các cửa sổ thời gian không chồng lấp kích thước w (Eq. 3-4)."""
    def __init__(self, w):
        super().__init__()
        self.w = w

    def forward(self, x):                           # x: (B, F, T)
        B, Fm, T = x.shape
        T2 = T - T % self.w
        x = x[..., :T2].reshape(B, Fm, T2 // self.w, self.w)
        return x.var(dim=-1, unbiased=False).flatten(1)     # (B, F * T/w)


class FeatureExtractor(nn.Module):
    def __init__(self, n_ch=22, n_times=1000, d=16, win_sec=1.0, fs=250, feat_dim=64):
        super().__init__()
        m = len(SUB_BANDS)
        self.cbam = CBAM(n=1)
        self.subband = SubBandFilter(fs=fs)
        self.spatial = nn.Sequential(
            nn.Conv2d(m, m * d, (n_ch, 1), groups=m, bias=False),
            nn.BatchNorm2d(m * d))
        w = int(round(win_sec * fs))
        self.var = VarianceLayer(w)
        self.fc = nn.Linear(m * d * (n_times // w), feat_dim)
        self.out_dim = feat_dim

    def forward(self, x):                           # x: (B, n_ch, T)
        x = self.cbam(x.unsqueeze(1))               # (B, 1, C, T)
        x = self.subband(x)                         # (B, m, C, T)
        x = self.spatial(x).squeeze(2)              # (B, m*d, T)
        return self.fc(self.var(x))                 # (B, 64)


class Classifier(nn.Module):
    """Hai lớp FC; softmax nằm trong cross-entropy (Eq. 8)."""
    def __init__(self, in_dim=64, hid=64, n_cls=4, drop=0.25):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ELU(), nn.Dropout(drop),
                                 nn.Linear(hid, n_cls))

    def forward(self, f):
        return self.net(f)


class Critic(nn.Module):
    """Domain discriminator f_w: ánh xạ đặc trưng thành một số thực (Eq. 5)."""
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.LeakyReLU(0.2),
                                 nn.Linear(hid, hid), nn.LeakyReLU(0.2),
                                 nn.Linear(hid, 1))

    def forward(self, f):
        return self.net(f)


def gradient_penalty(critic, fs, ft):
    """L_grad = (||grad f_w(h~)||_2 - 1)^2 (Eq. 6), với h~ = {h_s, h_t, điểm ngẫu nhiên
    giữa từng cặp h_s, h_t} như Algorithm 1, dòng 7-8."""
    n = min(len(fs), len(ft))
    fs, ft = fs[:n], ft[:n]
    a = torch.rand(n, 1, device=fs.device)
    h = torch.cat([fs, ft, a * fs + (1 - a) * ft]).requires_grad_(True)
    g = torch.autograd.grad(critic(h).sum(), h, create_graph=True)[0]
    return ((g.norm(2, dim=1) - 1) ** 2).mean()
