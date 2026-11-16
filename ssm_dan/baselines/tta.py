"""T-TIME (IEEE TBME 2024) and BFT (IEEE JBHI 2026) under the chronological LOSO protocol.

Both use the authors' implementations (``deeptransfer.py``):

* source subjects: the training sessions train EEGNet source models; the
  validation sessions only select the checkpoint (best source accuracy);
* each source subject is Euclidean-aligned with the reference of its own
  training sessions;
* target subject: the evaluation sessions are presented once, trial by trial
  in recording order, as an unlabeled stream (incremental EA + online
  adaptation); labels are read afterwards for scoring. As an auxiliary
  diagnostic, an independent stream over all target sessions is scored too;
* EA replaces z-scoring, as in the authors' pipeline.

EEGNet (F1 4, D 2, F2 8, kernel fs/2, dropout 0.25), Adam lr 1e-3, batch 32,
100 source epochs. T-TIME: sliding test batch 8, one step, stride 1,
temperature 2, SML ensemble of five seeded EEGNets. BFT: BFT-D, K = 10,
temperature 0.25, ranker trained 20 epochs at lr 1e-3.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from ..data import get_datamodule_cls
from ..utils.latency import measure_latency
from .deeptransfer import (
    BFT_func, EA_reference, ReliabilityRanker, SML_online_ensemble, TTIME, backbone_net,
    clone_model, feature_mask_branches, train_ranker,
)
from .sessions import SPLITS, concat


def fix_random_seed(seed):
    """tl/utils/utils.py :: fix_random_seed, plus the cuDNN flag the scripts set."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def aligned_source(sessions, target, split):
    """EA every source subject with the reference of its own training sessions."""
    Xs, ys, Xv, yv = [], [], [], []
    for subject, arrays in sessions.items():
        if subject == target:
            continue
        X_train, y_train = concat(arrays, split["train"])
        X_val, y_val = concat(arrays, split["test"])
        ref = EA_reference(X_train)
        Xs.append(np.real(np.einsum("cd,ndt->nct", ref, X_train)))
        Xv.append(np.real(np.einsum("cd,ndt->nct", ref, X_val)))
        ys.append(y_train)
        yv.append(y_val)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(Xv), np.concatenate(yv)


def to_loader(X, y, batch_size, shuffle, drop_last, device):
    X = torch.from_numpy(X).to(torch.float32).unsqueeze(1).to(device)
    y = torch.from_numpy(y).to(torch.long).to(device)
    return Data.DataLoader(Data.TensorDataset(X, y), batch_size=batch_size,
                           shuffle=shuffle, drop_last=drop_last)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    for inputs, labels in loader:
        _, outputs = model(inputs)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / total


def train_source_model(seed, train_loader, val_loader, args, device):
    """tl/ttime.py source stage, with best source-validation checkpoint selection."""
    fix_random_seed(seed)
    netF, netC, feature_dim = backbone_net(args.chn, args.time_sample_num, args.sample_rate,
                                           args.class_num)
    netF, netC = netF.to(device), netC.to(device)
    base_network = nn.Sequential(netF, netC)
    criterion = nn.CrossEntropyLoss()
    optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

    best_acc, best_state, best_epoch = -1.0, None, -1
    for epoch in range(args.max_epoch):
        base_network.train()
        for inputs_source, labels_source in train_loader:
            if inputs_source.size(0) == 1:
                continue
            _, outputs_source = base_network(inputs_source)
            classifier_loss = criterion(outputs_source, labels_source)
            optimizer_f.zero_grad()
            optimizer_c.zero_grad()
            classifier_loss.backward()
            optimizer_f.step()
            optimizer_c.step()
        val_acc = evaluate(base_network, val_loader)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in base_network.state_dict().items()}
        if args.log_every and (epoch + 1) % args.log_every == 0:
            print(f"    seed {seed} epoch {epoch + 1}/{args.max_epoch} val_acc={val_acc * 100:.2f}% "
                  f"(best {best_acc * 100:.2f}% @ {best_epoch + 1})", flush=True)
    base_network.load_state_dict(best_state)
    base_network.eval()
    print(f"  source model seed {seed}: best source-validation accuracy "
          f"{best_acc * 100:.2f}% at epoch {best_epoch + 1}", flush=True)
    return base_network, feature_dim, best_acc


class EnsembleForward(nn.Module):
    """Single-trial inference of the T-TIME ensemble (latency only)."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return torch.stack([torch.softmax(m(x)[1], dim=1) for m in self.models]).mean(0)


class BFTForward(nn.Module):
    """Single-trial BFT-D inference: K masked views weighted by the ranker (latency only)."""

    def __init__(self, model, ranker, K, temperature):
        super().__init__()
        self.model, self.ranker = model, ranker
        self.K, self.temperature = K, temperature

    def forward(self, x):
        feat = self.model[0](x)
        views, scores = [], []
        for masked in feature_mask_branches(feat, self.K):
            _, logits = self.model[1](masked)
            views.append(torch.softmax(logits / self.temperature, dim=1))
            scores.append(self.ranker(masked))
        weights = torch.softmax(-torch.stack(scores).squeeze(-1), dim=0)
        return (torch.stack(views) * weights.unsqueeze(-1)).sum(0)


def run_fold(config, dataset, target, sessions, device):
    """Train on the sources, adapt online to the target stream; per-subject record."""
    module = get_datamodule_cls(dataset)
    split = SPLITS[dataset]
    hp = config["method_settings"]
    method = hp["method"]
    args = argparse.Namespace(
        chn=module.channels, class_num=module.classes,
        sample_rate=config["preprocessing"]["sfreq"], lr=hp["lr"],
        batch_size=hp["batch_size"], max_epoch=hp["max_epoch"], align=True,
        log_every=config.get("log_every_n_epochs", 0), **hp[method],
    )
    args.time_sample_num = next(iter(sessions.values()))[0][0].shape[-1]
    seed = config["seed"]

    Xs, ys, Xv, yv = aligned_source(sessions, target, split)
    train_loader = to_loader(Xs, ys, args.batch_size, True, True, device)
    val_loader = to_loader(Xv, yv, args.batch_size * 3, False, False, device)
    X_stream, y_true = concat(sessions[target], split["test"])   # labels only for scoring

    started = time.time()
    if method == "ttime":
        seeds = [seed + m for m in range(args.ensemble_size)]
        source_models, val_accs = [], []
        for model_seed in seeds:
            model, _, val_acc = train_source_model(model_seed, train_loader, val_loader, args, device)
            source_models.append(model)
            val_accs.append(val_acc)
        best_val_acc = float(np.mean(val_accs))
        train_minutes = (time.time() - started) / 60

        def adapt_stream(X):
            preds = []
            for model_seed, source_model in zip(seeds, source_models):
                fix_random_seed(model_seed)
                preds.append(TTIME(X, clone_model(source_model), args, device))
            preds = np.stack(preds)                            # (M, n_trials, n_classes)
            return SML_online_ensemble(preds, args.class_num), preds.mean(0)

        param_count = sum(p.numel() for m in source_models for p in m.parameters())
        latency_model = EnsembleForward(source_models)
    elif method == "bft":
        model, feature_dim, best_val_acc = train_source_model(seed, train_loader, val_loader, args, device)
        fix_random_seed(seed)
        ranker = ReliabilityRanker(feature_dim).to(device)
        train_ranker(model[0], model[1], ranker, train_loader, args, device)
        train_minutes = (time.time() - started) / 60

        def adapt_stream(X):
            probs = BFT_func(X, clone_model(model), ranker, args, device)
            return probs.argmax(1), probs

        param_count = (sum(p.numel() for p in model.parameters())
                       + sum(p.numel() for p in ranker.parameters()))
        latency_model = BFTForward(model, ranker, args.K, args.temperature)
    else:
        raise ValueError(f"Unknown test-time method {method!r}.")

    started = time.time()
    y_hat, probs = adapt_stream(X_stream)
    test_seconds = time.time() - started
    record = {
        "test_acc": float(accuracy_score(y_true, y_hat)),
        "test_kappa": float(cohen_kappa_score(y_true, y_hat)),
        # Negative log-likelihood of the (averaged) online probabilities.
        "test_loss": float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-12))),
        "confusion_counts": confusion_matrix(y_true, y_hat, labels=list(range(args.class_num))).tolist(),
        "best_val_acc": best_val_acc,
        "train_time_min": train_minutes,
        "test_time_s": test_seconds,
        "param_count": int(param_count),
    }

    # Auxiliary diagnostic: a fresh online stream over every target session.
    X_all, y_all = concat(sessions[target], range(len(sessions[target])))
    y_hat_all, probs_all = adapt_stream(X_all)
    record.update({
        "all_sessions_acc": float(accuracy_score(y_all, y_hat_all)),
        "all_sessions_kappa": float(cohen_kappa_score(y_all, y_hat_all)),
        "all_sessions_loss": float(-np.mean(np.log(probs_all[np.arange(len(y_all)), y_all] + 1e-12))),
    })
    record["cpu_latency_ms"] = measure_latency(
        latency_model, (1, 1, args.chn, args.time_sample_num), device="cpu")
    return record
