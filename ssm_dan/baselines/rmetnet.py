"""RMETNet (PLOS ONE 2026) under the chronological LOSO protocol.

The authors' network (``rmetnet_net.py``) and fittvmmd training loop:

* source subjects: training sessions train the model, validation sessions only
  select the checkpoint (best source accuracy);
* target subject: its earlier sessions are the unlabeled target domain of the
  MMD loss (Stage-1 UDA); the evaluation sessions are used only at test time;
* Riemannian features: per-trial covariances of the raw EEG aligned as
  logm(C_common^-1 C), with C_common the Riemannian mean of the source and
  target-adaptation Riemannian means (fixed before training);
* EEG branch: global z-score with the source-training mean and std.

Adam lr 2e-4, betas (0.5, 0.999), weight decay 0.075, batch 64, loss
CE(source) + MMD(source, target), the shorter loader cycled.
"""
import argparse
import itertools
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from ..data import get_datamodule_cls
from ..utils.latency import measure_latency
from .rmetnet_net import MMDLoss, Net, align, common_reference, compute_covariance_matrix
from .sessions import SPLITS, concat
from .tta import fix_random_seed


def split_fold(sessions, target, split):
    """Source training / validation trials and the target's two session groups."""
    Xs, ys, Xv, yv = [], [], [], []
    for subject, arrays in sessions.items():
        if subject == target:
            continue
        X_train, y_train = concat(arrays, split["train"])
        X_val, y_val = concat(arrays, split["test"])
        Xs.append(X_train)
        ys.append(y_train)
        Xv.append(X_val)
        yv.append(y_val)
    X_adapt, _ = concat(sessions[target], split["train"])     # labels never read
    X_test, y_test = concat(sessions[target], split["test"])
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(Xv),
            np.concatenate(yv), X_adapt, X_test, y_test)


def to_tensor(X, dtype=torch.float32):
    return torch.from_numpy(np.ascontiguousarray(X)).to(dtype)


@torch.no_grad()
def predict(model, X, R, device, batch_size=256):
    model.eval()
    logits = []
    for i in range(0, len(X), batch_size):
        _, out = model(X[i:i + batch_size].to(device), R[i:i + batch_size].to(device))
        logits.append(out.cpu())
    return torch.cat(logits)


class RMETNetInference(nn.Module):
    """Raw trial -> prediction incl. covariance, frozen alignment and z-score (latency only).

    logm(C^-1 X) is evaluated as W logm(W X W) W^-1 with W = C^-1/2 (identical).
    """

    def __init__(self, net, C_common, mean, std):
        super().__init__()
        w, V = np.linalg.eigh(C_common)
        self.net = net
        self.register_buffer("W", torch.tensor(V @ np.diag(w ** -0.5) @ V.T, dtype=torch.float32))
        self.register_buffer("W_inv", torch.tensor(V @ np.diag(w ** 0.5) @ V.T, dtype=torch.float32))
        self.mean, self.std = float(mean), float(std)

    def forward(self, x):                                  # x: (1, 1, C, T) raw EEG
        trial = x[:, 0]
        centered = trial - trial.mean(dim=-1, keepdim=True)
        cov = centered @ centered.transpose(1, 2) / (trial.shape[-1] - 1)
        S = self.W @ cov @ self.W
        evals, evecs = torch.linalg.eigh(S)
        log_S = evecs @ torch.diag_embed(torch.log(evals.clamp_min(1e-12))) @ evecs.transpose(1, 2)
        rie = self.W @ log_S @ self.W_inv
        _, out = self.net((x - self.mean) / self.std, rie)
        return out


def train_fold(Xs, Rs, ys, Xa, Ra, Xv, Rv, yv, args, device):
    """Authors' fittvmmd loop with source-validation checkpoint selection."""
    fix_random_seed(args.seed)
    model = Net(num_classes=args.class_num, num_channels=args.chn,
                sampling_rate=args.sample_rate, num_samples=args.time_sample_num).to(device)
    criterion = nn.CrossEntropyLoss()
    mmdloss = MMDLoss().to(device)
    optimizer = optim.Adam(params=list(model.parameters()), lr=args.lr,
                           betas=tuple(args.betas), weight_decay=args.weight_decay)
    train_loader = Data.DataLoader(Data.TensorDataset(Xs, Rs, ys), batch_size=args.batch_size, shuffle=True)
    target_loader = Data.DataLoader(Data.TensorDataset(Xa, Ra), batch_size=args.batch_size, shuffle=True)
    # As in fittvmmd: the shorter loader is cycled, once, before the epoch loop.
    if len(train_loader) > len(target_loader):
        target_iter, train_iter, loop_len = itertools.cycle(target_loader), None, len(train_loader)
    else:
        target_iter, train_iter, loop_len = None, itertools.cycle(train_loader), len(target_loader)

    best_acc, best_state, best_epoch = -1.0, None, -1
    for epoch in range(args.epochs):
        model.train()
        source_batches = train_iter if train_iter is not None else train_loader
        target_batches = target_iter if target_iter is not None else target_loader
        for _, ((source_data, source_rie, source_labels), (target_data, target_rie)) in zip(
                range(loop_len), zip(source_batches, target_batches)):
            source_data, source_rie = source_data.to(device), source_rie.to(device)
            target_data, target_rie = target_data.to(device), target_rie.to(device)
            source_labels = source_labels.to(device)
            source_feature, outputs = model(source_data, source_rie)
            target_feature, _ = model(target_data, target_rie)
            loss = criterion(outputs, source_labels)
            domainloss = mmdloss(source_feature, target_feature)
            total_loss = loss + domainloss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        val_acc = (predict(model, Xv, Rv, device).argmax(1) == yv).float().mean().item()
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if args.log_every and (epoch + 1) % args.log_every == 0:
            print(f"    epoch {epoch + 1}/{args.epochs} | cls={loss.item():.4f} | "
                  f"mmd={domainloss.item():.4f} | val_acc={val_acc * 100:.2f}% "
                  f"(best {best_acc * 100:.2f}% @ {best_epoch + 1})", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    print(f"  best source-validation accuracy {best_acc * 100:.2f}% at epoch {best_epoch + 1}", flush=True)
    return model, best_acc


def run_fold(config, dataset, target, sessions, device):
    module = get_datamodule_cls(dataset)
    hp = dict(config["method_settings"])
    cov_reg = hp.pop("cov_reg", 0.0)
    args = argparse.Namespace(
        chn=module.channels, class_num=module.classes, sample_rate=config["preprocessing"]["sfreq"],
        seed=config["seed"], log_every=config.get("log_every_n_epochs", 0), **hp,
    )
    args.time_sample_num = next(iter(sessions.values()))[0][0].shape[-1]
    Xs, ys, Xv, yv, X_adapt, X_test, y_test = split_fold(sessions, target, SPLITS[dataset])

    # Riemannian features of the raw EEG; reference from source training and
    # unlabeled target-adaptation trials only.
    covs = {k: compute_covariance_matrix(v, cov_reg)
            for k, v in dict(s=Xs, v=Xv, a=X_adapt, t=X_test).items()}
    C_common = common_reference(covs["s"], covs["a"])
    rie = {k: to_tensor(align(c, C_common)) for k, c in covs.items()}
    mean, std = Xs.mean(), Xs.std()
    eeg = {k: to_tensor((v - mean) / std).unsqueeze(1)
           for k, v in dict(s=Xs, v=Xv, a=X_adapt, t=X_test).items()}

    started = time.time()
    model, best_val_acc = train_fold(eeg["s"], rie["s"], to_tensor(ys, torch.long), eeg["a"], rie["a"],
                                     eeg["v"], rie["v"], to_tensor(yv, torch.long), args, device)
    train_minutes = (time.time() - started) / 60
    started = time.time()
    logits = predict(model, eeg["t"], rie["t"], device)
    test_seconds = time.time() - started
    y_hat = logits.argmax(1).numpy()
    record = {
        "test_acc": float(accuracy_score(y_test, y_hat)),
        "test_kappa": float(cohen_kappa_score(y_test, y_hat)),
        "test_loss": float(F.cross_entropy(logits, to_tensor(y_test, torch.long)).item()),
        "confusion_counts": confusion_matrix(y_test, y_hat, labels=list(range(args.class_num))).tolist(),
        "best_val_acc": float(best_val_acc),
        "train_time_min": train_minutes,
        "test_time_s": test_seconds,
        "param_count": int(sum(p.numel() for p in model.parameters())),
    }
    record["cpu_latency_ms"] = measure_latency(
        RMETNetInference(model, C_common, mean, std), (1, 1, args.chn, args.time_sample_num), device="cpu")
    return record
