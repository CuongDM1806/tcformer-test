"""Chronological LOSO evaluation of RMETNet (BCI IV-2a, IV-2b, Zhou2016).

RMETNet is the authors' implementation (utils/rmetnet.py) run under the same
chronological protocol as SSM-DAN. Session split per dataset
(source training / source validation; target unlabeled adaptation / evaluation):
BCI IV-2a 1 / 2, BCI IV-2b 1-3 / 4-5, Zhou2016 1-2 / 3.

* source subjects: the training sessions train the model, the validation
  sessions are the labeled split used only to select the checkpoint (highest
  source validation accuracy), as for every other method in the paper;
* target subject: the earlier sessions are the unlabeled target domain of the
  MMD loss (Stage 1 UDA in the paper's protocol); the evaluation sessions are
  used only at test time, and their labels only for scoring;
* Riemannian features: per-trial covariances (np.cov) of the raw EEG, aligned
  as logm(C_common^{-1} C) with C_common the Riemannian mean of the source
  (training sessions) and target (unlabeled adaptation sessions) Riemannian
  means, as in the authors' riemannian_alignment. The reference is fixed before
  training; validation and evaluation trials are only transformed;
* EEG branch: global z-score with the mean and standard deviation of the source
  training data, as in the authors' loader;
* preprocessing (windows, Zhou2016 8-30 Hz filter) is the paper's per-dataset
  setting.

Training follows the authors' main script and fittvmmd loop: Adam lr 2e-4,
betas (0.5, 0.999), weight decay 0.075, batch 64, 1000 epochs, loss
CE(source) + MMD(source features, target features), the shorter loader cycled.
In the authors' script the MMD target domain and the model-selection split are
the labeled test subject; here they are replaced by the protocol's unlabeled
target adaptation sessions and the source validation sessions.
"""
import argparse
import itertools
import os
import time
from datetime import datetime
from pathlib import Path

if os.environ.get("MPLBACKEND", "").startswith("module://matplotlib_inline"):
    os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from tta_baselines import DATASETS, _concat, fix_random_seed, load_sessions
from utils.latency import measure_latency
from utils.metrics import write_summary
from utils.plotting import plot_confusion_matrix
from utils.rmetnet import MMDLoss, Net, align, common_reference, compute_covariance_matrix

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def split_fold(sessions, target, split):
    """Source training / validation trials and the target's two session groups."""
    Xs, ys, Xv, yv = [], [], [], []
    for subject, arrays in sessions.items():
        if subject == target:
            continue
        X_train, y_train = _concat(arrays, split["train"])
        X_val, y_val = _concat(arrays, split["test"])
        Xs.append(X_train)
        ys.append(y_train)
        Xv.append(X_val)
        yv.append(y_val)
    X_adapt, _ = _concat(sessions[target], split["train"])     # labels never read
    X_test, y_test = _concat(sessions[target], split["test"])
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
    """Single raw trial -> prediction, including the covariance, the frozen
    Riemannian alignment and the z-score (latency only). logm(C^{-1} X) is
    evaluated as W logm(W X W) W^{-1} with W = C^{-1/2}, which is identical."""

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

    train_loader = Data.DataLoader(Data.TensorDataset(Xs, Rs, ys),
                                   batch_size=args.batch_size, shuffle=True)
    target_loader = Data.DataLoader(Data.TensorDataset(Xa, Ra),
                                    batch_size=args.batch_size, shuffle=True)
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
    print(f"  best source-validation accuracy {best_acc * 100:.2f}% at epoch {best_epoch + 1}",
          flush=True)
    return model


def run(config, gpu_id, dataset_name="bcic2a"):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    split = DATASETS[dataset_name]
    module = split["module"]
    preprocessing = config["preprocessing"][dataset_name]
    hp = dict(config["rmetnet"])
    hp["epochs"] = hp.pop("epochs_by_dataset", {}).get(dataset_name, hp["epochs"])
    cov_reg = hp.pop("cov_reg_by_dataset", {}).get(dataset_name, 0.0)
    args = argparse.Namespace(
        chn=module.channels, class_num=module.classes, sample_rate=preprocessing["sfreq"],
        seed=config["seed"], log_every=config.get("log_every_n_epochs", 0), **hp,
    )
    print(f"RMETNet | {dataset_name} | {args.epochs} epochs", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = (Path(__file__).resolve().parent /
                  f"results/RMETNet_{dataset_name}_loso_seed-{config['seed']}_GPU{gpu_id}_{timestamp}")
    (result_dir / "confmats").mkdir(parents=True, exist_ok=True)
    with open(result_dir / "config.yaml", "w") as f:
        yaml.dump({"method": "rmetnet", "dataset": dataset_name, **config}, f,
                  default_flow_style=False)

    sessions = load_sessions(dataset_name, preprocessing)
    args.time_sample_num = next(iter(sessions.values()))[0][0].shape[-1]

    subject_ids = config["subject_ids"]
    if subject_ids == "all":
        subject_ids = module.all_subject_ids
    test_accs, test_losses, test_kappas = [], [], []
    train_times, test_times, response_times, confmats = [], [], [], []

    for target in subject_ids:
        print(f"\n>>> RMETNet | target subject {target}", flush=True)
        Xs, ys, Xv, yv, X_adapt, X_test, y_test = split_fold(sessions, target, split)

        # Riemannian features from the raw EEG; reference fixed from source
        # training and unlabeled target adaptation trials only.
        covs = {k: compute_covariance_matrix(v, cov_reg) for k, v in
                dict(s=Xs, v=Xv, a=X_adapt, t=X_test).items()}
        C_common = common_reference(covs["s"], covs["a"])
        rie = {k: to_tensor(align(c, C_common)) for k, c in covs.items()}

        # Global z-score with the source training statistics.
        mean, std = Xs.mean(), Xs.std()
        eeg = {k: to_tensor((v - mean) / std).unsqueeze(1)
               for k, v in dict(s=Xs, v=Xv, a=X_adapt, t=X_test).items()}
        print(f"  source train {tuple(Xs.shape)}, source val {tuple(Xv.shape)}, "
              f"target adaptation (unlabeled) {tuple(X_adapt.shape)}, "
              f"target test {tuple(X_test.shape)}", flush=True)

        st_train = time.time()
        model = train_fold(eeg["s"], rie["s"], to_tensor(ys, torch.long), eeg["a"], rie["a"],
                           eeg["v"], rie["v"], to_tensor(yv, torch.long), args, device)
        train_times.append((time.time() - st_train) / 60)

        st_test = time.time()
        logits = predict(model, eeg["t"], rie["t"], device)
        test_times.append(time.time() - st_test)
        y_hat = logits.argmax(1).numpy()
        loss = F.cross_entropy(logits, to_tensor(y_test, torch.long)).item()
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
        latency_model = RMETNetInference(model, C_common, mean, std)
        response_times.append(measure_latency(
            latency_model, (1, 1, args.chn, args.time_sample_num), device="cpu"))
        print(f"\nTARGET SUBJECT {target} {split['label']} RESULT | acc={acc * 100:.2f}% | "
              f"loss={loss:.4f} | kappa={kappa:.4f} | test_time={test_times[-1]:.2f}s\n",
              flush=True)

    write_summary(result_dir, "RMETNet", f"{dataset_name}_loso", list(subject_ids), param_count,
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
    with open(CONFIG_DIR / "rmetnet.yaml") as f:
        config = yaml.safe_load(f)
    run(config, cli.gpu_id, cli.dataset)


if __name__ == "__main__":
    main()
