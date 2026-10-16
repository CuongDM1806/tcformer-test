"""MSD-DDA (Gong et al., IEEE JBHI 2025) — network and losses.

Network from the authors' release (https://github.com/EllieYLJ/MSD-DDA,
models.py): shared CFE MLP input -> 256 -> 128 -> 64 (Linear, BatchNorm,
LeakyReLU 0.01, Dropout 0.5), one DSFE per source (Linear 64 -> 32, LeakyReLU,
BatchNorm) and one fully connected classifier per source (Table II; the
released file refers to an undefined SLR_layer). Kernels from the authors'
utils.py (guassian_kernel, lsd).

The released training script does not match the paper (linear MMD, column-norm
BNM, label smoothing, top-3 source voting) and cannot run as published, so
the losses follow the paper's equations:
  Eq. (1)  multi-kernel Gaussian MMD between the source and target DSFE features;
  Eq. (11) local subdomain discrepancy with target pseudo-labels of confidence
           >= 0.6, for all classes (the release hard-codes two);
  Eq. (15) L_bnm = -||softmax(C(G(X_t)))||_* / B;
  Eq. (3)-(5) L = L_cls + lambda * (alpha L_mmd + (1-alpha) L_lsd + (1-alpha)/10 L_bnm),
           alpha = sigmoid(100 - K), K the training iteration;
  Eq. (16)-(17) w = softmax(eta / (beta L_mmd + gamma L_cls)) per source and
           the weighted sum of the per-source softmax outputs as prediction.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    L2_distance = torch.cdist(total, total, p=2).pow(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth = torch.clamp(bandwidth, min=1e-6) / (kernel_mul ** (kernel_num // 2))
    return sum(torch.exp(-L2_distance / (bandwidth * kernel_mul ** i)) for i in range(kernel_num))


def mmd(source, target):
    """Eq. (1); the means support unequal source/target batch sizes."""
    n = source.size(0)
    kernels = guassian_kernel(source, target)
    return kernels[:n, :n].mean() + kernels[n:, n:].mean() - 2 * kernels[:n, n:].mean()


def lsd(source, target, source_label, target_label, num_classes):
    """Eq. (6)-(11): mean intra-class minus mean inter-class discrepancy."""
    n = source.size(0)
    kernels = guassian_kernel(source, target)
    XX, YY = kernels[:n, :n], kernels[n:, n:]
    XY, YX = kernels[:n, n:], kernels[n:, :n]
    intra, inter = [], []
    for c1 in range(num_classes):
        s = (source_label == c1).float()
        for c2 in range(num_classes):
            t = (target_label == c2).float()
            ss, tt = torch.outer(s, s), torch.outer(t, t)
            st, ts = torch.outer(s, t), torch.outer(t, s)
            value = ((ss * XX).sum() / (ss.sum() + 1e-5) + (tt * YY).sum() / (tt.sum() + 1e-5)
                     - (st * XY).sum() / (st.sum() + 1e-5) - (ts * YX).sum() / (ts.sum() + 1e-5))
            (intra if c1 == c2 else inter).append(value)
    return torch.stack(intra).mean() - torch.stack(inter).mean()


class CFE(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        layers, dims = [], [input_dim, 256, 128, 64]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out),
                       nn.LeakyReLU(negative_slope=0.01, inplace=True), nn.Dropout(p=0.5)]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class DSFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(nn.Linear(64, 32),
                                    nn.LeakyReLU(negative_slope=0.01, inplace=True))
        self.normalize = nn.BatchNorm1d(32)

    def forward(self, x):
        return self.normalize(self.module(x))


class MSDDDANet(nn.Module):
    def __init__(self, input_dim, number_of_source, number_of_category, pseudo_threshold=0.6):
        super().__init__()
        self.num_classes = number_of_category
        self.pseudo_threshold = pseudo_threshold
        self.sharedNet = CFE(input_dim)
        self.DSFE = nn.ModuleList([DSFE() for _ in range(number_of_source)])
        self.cls_fc_DSC = nn.ModuleList([nn.Linear(32, number_of_category)
                                         for _ in range(number_of_source)])

    def losses(self, data_src, label_src, data_tgt, mark):
        """Losses of the source-target pair `mark` (training mode)."""
        feat_src = self.DSFE[mark](self.sharedNet(data_src))
        feat_tgt = self.DSFE[mark](self.sharedNet(data_tgt))
        pred_src = self.cls_fc_DSC[mark](feat_src)
        prob_tgt = F.softmax(self.cls_fc_DSC[mark](feat_tgt), dim=1)

        cls_loss = F.cross_entropy(pred_src, label_src)
        mmd_loss = mmd(feat_src, feat_tgt)
        max_prob, pseudo_label = prob_tgt.max(dim=1)
        confident = max_prob >= self.pseudo_threshold
        if confident.any():
            lsd_loss = lsd(feat_src, feat_tgt[confident], label_src, pseudo_label[confident],
                           self.num_classes)
        else:
            lsd_loss = feat_src.sum() * 0.0
        bnm_loss = -torch.linalg.matrix_norm(prob_tgt, ord="nuc") / prob_tgt.shape[0]
        return cls_loss, mmd_loss, lsd_loss, bnm_loss

    def forward(self, x, weights):
        """Eq. (16): weighted sum of the per-source softmax outputs."""
        common = self.sharedNet(x)
        probs = [F.softmax(cls(dsfe(common)), dim=1)
                 for dsfe, cls in zip(self.DSFE, self.cls_fc_DSC)]
        return sum(w * p for w, p in zip(weights, probs))


def domain_weights(mmd_losses, cls_losses, eta, beta, gamma):
    """Eq. (17): softmax(eta / (beta * L_mmd + gamma * L_cls)) over sources."""
    scores = eta / np.maximum(beta * np.asarray(mmd_losses) + gamma * np.asarray(cls_losses), 1e-8)
    scores = np.exp(scores - scores.max())
    return scores / scores.sum()


def ea_reference(X):
    """R^{-1/2} of the mean trial covariance (Eq. 18-19)."""
    R = np.mean(X @ X.transpose(0, 2, 1), axis=0) / X.shape[-1]
    w, V = np.linalg.eigh(R)
    return V @ np.diag(np.clip(w, 1e-12, None) ** -0.5) @ V.T


def rcsp_features(X, filters):
    """Eq. (20): f = log(var(W X)) of the RCSP-projected trials."""
    projected = np.einsum("kc,nct->nkt", filters, X)
    return np.log((projected ** 2).mean(axis=-1))
