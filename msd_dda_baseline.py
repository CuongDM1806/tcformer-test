"""Chronological LOSO evaluation of MSD-DDA (BCI IV-2a, IV-2b, Zhou2016).

MSD-DDA (utils/msd_dda.py) is run under the same chronological protocol as
SSM-DAN. Session split per dataset (source training / source validation;
target unlabeled adaptation / evaluation):
BCI IV-2a 1 / 2, BCI IV-2b 1-3 / 4-5, Zhou2016 1-2 / 3.

* source domains: every other subject is one source domain (multi-source, as
  in the paper); its training sessions train its branch, its validation
  sessions are the labeled split used only to select the checkpoint (highest
  source validation accuracy of the weighted joint prediction), as for every
  other method in the paper;
* target domain: the earlier sessions of the target subject are the unlabeled
  target data of the MMD, LSD and BNM losses; the evaluation sessions are used
  only at test time, and their labels only for scoring;
* features (paper Section IV-B): 8-30 Hz band-pass, CAR, Euclidean alignment
  and RCSP log-variance features. CAR is applied by projecting each trial on
  an orthonormal basis of the zero-sum subspace (rank C-1), where EA is well
  defined. EA reference per subject: the source subjects' training sessions,
  the target's unlabeled adaptation sessions (evaluation trials are only
  transformed). RCSP: one-vs-rest CSP (binary CSP for two classes) on
  Ledoit-Wolf regularised class covariances, fitted on the labeled source
  training trials only, keeping n_pairs filters from each end; the features
  are standardised with the source training statistics. ICA is not applied:
  the paper does not report its component rejection rule;
* windows: the paper's per-dataset setting.

Training follows the paper: Adam lr 1e-4, 200 epochs, batch 18, lambda 0.3,
pseudo-label threshold 0.6, alpha = sigmoid(100 - K) with K the iteration
(one iteration = one minibatch of every source domain, as in the authors'
script), domain weights of Eq. (17) with eta 2, beta 10, gamma 1 recomputed
from the epoch's mean losses.
"""
import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path

if os.environ.get("MPLBACKEND", "").startswith("module://matplotlib_inline"):
    os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import yaml
from sklearn.covariance import ledoit_wolf
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from rmetnet_baseline import to_tensor
from tta_baselines import DATASETS, _concat, fix_random_seed, load_sessions
from utils.latency import measure_latency
from utils.metrics import write_summary
from utils.msd_dda import MSDDDANet, domain_weights, ea_reference, rcsp_features
from utils.plotting import plot_confusion_matrix

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def car_basis(n_channels):
    """Orthonormal basis (C-1, C) of the zero-sum subspace: CAR + rank reduction."""
    return scipy.linalg.null_space(np.ones((1, n_channels))).T


def _class_cov(X):
    samples = X.transpose(1, 0, 2).reshape(X.shape[1], -1).T
    return ledoit_wolf(samples)[0]


def fit_rcsp(X, y, n_classes, n_pairs):
    """Regularised CSP filters (rows), one-vs-rest for more than two classes."""
    problems = [(0, [1])] if n_classes == 2 else [
        (c, [o for o in range(n_classes) if o != c]) for c in range(n_classes)]
    covs = {c: _class_cov(X[y == c]) for c in range(n_classes)}
    filters = []
    for c, rest in problems:
        Ca = covs[c]
        Cb = np.mean([covs[o] for o in rest], axis=0)
        _, W = scipy.linalg.eigh(Ca, Ca + Cb)          # eigenvalues ascending
        idx = list(range(n_pairs)) + list(range(W.shape[1] - n_pairs, W.shape[1]))
        filters.append(W[:, idx].T)
    return np.concatenate(filters)


def prepare_fold(sessions, target, split, args):
    """RCSP feature arrays of every source domain and of the target subject."""
    n_channels = next(iter(sessions.values()))[0][0].shape[1]
    B = car_basis(n_channels)

    def reduce(X):
        return np.einsum("kc,nct->nkt", B, X)

    sources = []
    for subject, arrays in sessions.items():
        if subject == target:
            continue
        X_train, y_train = _concat(arrays, split["train"])
        X_val, y_val = _concat(arrays, split["test"])
        X_train, X_val = reduce(X_train), reduce(X_val)
        R = ea_reference(X_train)
        sources.append(dict(X=R @ X_train, y=y_train, Xv=R @ X_val, yv=y_val))

    X_adapt, _ = _concat(sessions[target], split["train"])     # labels never read
    X_test, y_test = _concat(sessions[target], split["test"])
    X_adapt, X_test = reduce(X_adapt), reduce(X_test)
    R_target = ea_reference(X_adapt)
    X_adapt, X_test = R_target @ X_adapt, R_target @ X_test

    # Filters from each end of the spectrum, at most half the CAR rank.
    n_pairs = max(1, min(args.rcsp_pairs, (n_channels - 1) // 2))
    filters = fit_rcsp(np.concatenate([s["X"] for s in sources]),
                       np.concatenate([s["y"] for s in sources]), args.class_num, n_pairs)
    for s in sources:
        s["f"], s["fv"] = rcsp_features(s["X"], filters), rcsp_features(s["Xv"], filters)
    f_adapt, f_test = rcsp_features(X_adapt, filters), rcsp_features(X_test, filters)

    train_features = np.concatenate([s["f"] for s in sources])
    mean, scale = train_features.mean(0), train_features.std(0) + 1e-8
    for s in sources:
        s["f"], s["fv"] = (s["f"] - mean) / scale, (s["fv"] - mean) / scale
    f_adapt, f_test = (f_adapt - mean) / scale, (f_test - mean) / scale
    transform = dict(B=B, R_target=R_target, filters=filters, mean=mean, scale=scale)
    return sources, f_adapt, f_test, y_test, transform


@torch.no_grad()
def predict(model, F_, weights, device, batch_size=512):
    model.eval()
    return torch.cat([model(F_[i:i + batch_size].to(device), weights).cpu()
                      for i in range(0, len(F_), batch_size)])


class MSDDDAInference(nn.Module):
    """Single raw trial -> prediction, including CAR, the target EA reference,
    the RCSP projection, log-variance and standardisation (latency only)."""

    def __init__(self, model, transform, weights):
        super().__init__()
        self.model, self.weights = model, [float(w) for w in weights]
        spatial = transform["filters"] @ transform["R_target"] @ transform["B"]
        self.register_buffer("spatial", torch.tensor(spatial, dtype=torch.float32))
        self.register_buffer("mean", torch.tensor(transform["mean"], dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(transform["scale"], dtype=torch.float32))

    def forward(self, x):                                  # x: (1, 1, C, T) filtered EEG
        projected = self.spatial @ x[:, 0]
        features = (torch.log((projected ** 2).mean(-1)) - self.mean) / self.scale
        return self.model(features, self.weights)


def train_fold(sources, f_adapt, args, device):
    fix_random_seed(args.seed)
    n_sources = len(sources)
    model = MSDDDANet(input_dim=f_adapt.shape[1], number_of_source=n_sources,
                      number_of_category=args.class_num,
                      pseudo_threshold=args.pseudo_threshold).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # drop_last: BatchNorm cannot train on a final batch of one trial.
    source_loaders = [Data.DataLoader(Data.TensorDataset(to_tensor(s["f"]),
                                                         to_tensor(s["y"], torch.long)),
                                      batch_size=args.batch_size, shuffle=True, drop_last=True)
                      for s in sources]
    target_loader = Data.DataLoader(Data.TensorDataset(to_tensor(f_adapt)),
                                    batch_size=args.batch_size, shuffle=True, drop_last=True)
    Fv = to_tensor(np.concatenate([s["fv"] for s in sources]))
    yv = to_tensor(np.concatenate([s["yv"] for s in sources]), torch.long)

    def next_batch(i, loader, iters):
        try:
            return next(iters[i])
        except StopIteration:
            iters[i] = iter(loader)
            return next(iters[i])

    source_iters = [iter(l) for l in source_loaders]
    target_iters = [iter(target_loader)]
    steps_per_epoch = max(len(l) for l in source_loaders)
    weights = np.full(n_sources, 1.0 / n_sources)
    best_acc, best_state, best_weights, best_epoch = -1.0, None, None, -1
    iteration = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_cls = [[] for _ in range(n_sources)]
        epoch_mmd = [[] for _ in range(n_sources)]
        for _ in range(steps_per_epoch):
            iteration += 1
            # Eq. (5): alpha = e^(100-K) / (1 + e^(100-K)).
            alpha = 1.0 / (1.0 + math.exp(min(max(iteration - 100.0, -80.0), 80.0)))
            for mark in range(n_sources):
                source_data, source_label = next_batch(mark, source_loaders[mark], source_iters)
                (target_data,) = next_batch(0, target_loader, target_iters)
                cls_loss, mmd_loss, lsd_loss, bnm_loss = model.losses(
                    source_data.to(device), source_label.to(device), target_data.to(device), mark)
                dynamic = alpha * mmd_loss + (1 - alpha) * lsd_loss + (1 - alpha) / 10 * bnm_loss
                total_loss = cls_loss + args.lambda_adapt * dynamic

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                epoch_cls[mark].append(cls_loss.item())
                epoch_mmd[mark].append(mmd_loss.item())

        weights = domain_weights([np.mean(v) for v in epoch_mmd], [np.mean(v) for v in epoch_cls],
                                 args.weight_eta, args.weight_beta, args.weight_gamma)
        val_acc = (predict(model, Fv, weights, device).argmax(1) == yv).float().mean().item()
        if val_acc > best_acc:
            best_acc, best_epoch, best_weights = val_acc, epoch, weights.copy()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if args.log_every and (epoch + 1) % args.log_every == 0:
            print(f"    epoch {epoch + 1}/{args.epochs} | alpha={alpha:.3f} | "
                  f"cls={np.mean(epoch_cls):.4f} | mmd={np.mean(epoch_mmd):.4f} | "
                  f"lsd={lsd_loss.item():.4f} | bnm={bnm_loss.item():.4f} | "
                  f"val_acc={val_acc * 100:.2f}% (best {best_acc * 100:.2f}% @ {best_epoch + 1})",
                  flush=True)

    model.load_state_dict(best_state)
    model.eval()
    print(f"  best source-validation accuracy {best_acc * 100:.2f}% at epoch {best_epoch + 1}; "
          f"domain weights {np.round(best_weights, 3).tolist()}", flush=True)
    return model, best_weights


def run(config, gpu_id, dataset_name="bcic2a"):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    split = DATASETS[dataset_name]
    module = split["module"]
    preprocessing = config["preprocessing"][dataset_name]
    args = argparse.Namespace(class_num=module.classes, seed=config["seed"],
                              log_every=config.get("log_every_n_epochs", 0),
                              **config["msd_dda"])
    print(f"MSD-DDA | {dataset_name} | {args.epochs} epochs", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = (Path(__file__).resolve().parent /
                  f"results/MSD-DDA_{dataset_name}_loso_seed-{config['seed']}_GPU{gpu_id}_{timestamp}")
    (result_dir / "confmats").mkdir(parents=True, exist_ok=True)
    with open(result_dir / "config.yaml", "w") as f:
        yaml.dump({"method": "msd_dda", "dataset": dataset_name, **config}, f,
                  default_flow_style=False)

    sessions = load_sessions(dataset_name, preprocessing)
    n_channels, n_samples = next(iter(sessions.values()))[0][0].shape[1:]

    subject_ids = config["subject_ids"]
    if subject_ids == "all":
        subject_ids = module.all_subject_ids
    test_accs, test_losses, test_kappas = [], [], []
    train_times, test_times, response_times, confmats = [], [], [], []

    for target in subject_ids:
        print(f"\n>>> MSD-DDA | target subject {target}", flush=True)
        st_train = time.time()
        sources, f_adapt, f_test, y_test, transform = prepare_fold(sessions, target, split, args)
        print(f"  {len(sources)} source domains, "
              f"source train {sum(len(s['y']) for s in sources)} trials, "
              f"source val {sum(len(s['yv']) for s in sources)} trials, "
              f"target adaptation (unlabeled) {len(f_adapt)}, target test {len(f_test)}, "
              f"RCSP features {f_adapt.shape[1]}", flush=True)
        model, weights = train_fold(sources, f_adapt, args, device)
        train_times.append((time.time() - st_train) / 60)

        st_test = time.time()
        probs = predict(model, to_tensor(f_test), weights, device)
        test_times.append(time.time() - st_test)
        y_hat = probs.argmax(1).numpy()
        loss = F.nll_loss(torch.log(probs + 1e-8), to_tensor(y_test, torch.long)).item()
        acc = accuracy_score(y_test, y_hat)
        kappa = cohen_kappa_score(y_test, y_hat)
        test_accs.append(acc)
        test_kappas.append(kappa)
        test_losses.append(loss)
        cm = confusion_matrix(y_test, y_hat, labels=list(range(args.class_num)))
        confmats.append(cm)
        plot_confusion_matrix(
            cm, save_path=result_dir / f"confmats/confmat_subject_{target}.png",
            class_names=module.class_names,
            title=f"Confusion Matrix – Subject {target}",
        )
        param_count = sum(p.numel() for p in model.parameters())
        response_times.append(measure_latency(
            MSDDDAInference(model, transform, weights), (1, 1, n_channels, n_samples),
            device="cpu"))
        print(f"\nTARGET SUBJECT {target} {split['label']} RESULT | acc={acc * 100:.2f}% | "
              f"loss={loss:.4f} | kappa={kappa:.4f} | test_time={test_times[-1]:.2f}s\n",
              flush=True)

    write_summary(result_dir, "MSD-DDA", f"{dataset_name}_loso", list(subject_ids), param_count,
                  test_accs, test_losses, test_kappas, train_times, test_times,
                  response_times, primary_test_label=split["label"])
    plot_confusion_matrix(
        np.mean(confmats, axis=0), save_path=result_dir / "confmats/avg_confusion_matrix.png",
        class_names=module.class_names, title="Average Confusion Matrix",
    )
    print((result_dir / "results.txt").read_text(encoding="utf-8"), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="bcic2a")
    parser.add_argument("--gpu_id", type=int, default=0)
    cli = parser.parse_args()
    with open(CONFIG_DIR / "msd_dda.yaml") as f:
        config = yaml.safe_load(f)
    run(config, cli.gpu_id, cli.dataset)


if __name__ == "__main__":
    main()
