"""Chronological LOSO evaluation of T-TIME and BFT (BCI IV-2a, IV-2b, Zhou2016).

Both methods are the authors' implementations (utils/deeptransfer_tta.py) run
under the same chronological protocol as SSM-DAN. Session split per dataset
(source training / source validation = target evaluation sessions):
BCI IV-2a 1 / 2, BCI IV-2b 1-3 / 4-5, Zhou2016 1-2 / 3.

* source subjects: the training sessions train the EEGNet source models, the
  validation sessions are the labeled split used only to select the checkpoint
  (highest source validation accuracy), as for every other method in the paper;
* each source subject is Euclidean-aligned with the reference of its own
  training sessions (the authors align every source subject independently);
* target subject: the evaluation sessions are presented once, trial by trial
  in recording order, as an unlabeled stream. Incremental EA, the online
  adaptation and the predictions use the EEG only; labels are read afterwards
  for scoring. Earlier target sessions are not used (test-time methods).
* preprocessing (windows, Zhou2016 8-30 Hz filter) is the paper's per-dataset
  setting; EA replaces z-scoring, as in the authors' pipeline.
* the T-TIME ensemble is SML: multiclass (one-vs-rest) for BCI IV-2a and
  Zhou2016, binary for BCI IV-2b, as in the authors' ttime_ensemble.py.

Settings follow the authors' scripts and papers: EEGNet (F1 4, D 2, F2 8,
kernel fs/2, dropout 0.25), Adam lr 1e-3, batch 32, 100 source epochs; T-TIME
with sliding test batch 8, one step, stride 1, temperature 2 and the complete
SML ensemble of five independently seeded EEGNets ("T-TIME (5)" in the paper);
BFT with the BFT-D variant, K = 10, temperature 0.25, a ranker trained for 20
epochs at lr 1e-3, and BN statistics refreshed on a sliding batch of 8.
"""
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

if os.environ.get("MPLBACKEND", "").startswith("module://matplotlib_inline"):
    os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from datamodules.bcic4_2a import BCICIV2a, _get_2a_train_test_sessions
from datamodules.bcic4_2b import BCICIV2b, _get_ordered_sessions
from datamodules.zhou2016 import Zhou2016LOSO, _get_three_sessions
from datamodules.base import BaseDataModule
from utils.deeptransfer_tta import (
    BFT_func, EA_reference, ReliabilityRanker, SML_online_ensemble, TTIME,
    backbone_net, clone_model, feature_mask_branches, train_ranker,
)
from utils.latency import measure_latency
from utils.load_bcic4 import load_bcic4
from utils.load_zhou2016 import load_zhou2016
from utils.metrics import write_summary
from utils.plotting import plot_confusion_matrix

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def fix_random_seed(seed):
    """tl/utils/utils.py :: fix_random_seed, plus the cuDNN flag the scripts set."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Chronological split of each dataset, identical to its LOSO datamodule:
# source training sessions / source validation (= target evaluation) sessions.
DATASETS = {
    "bcic2a": dict(module=BCICIV2a, train=[0], test=[1], label="SESSION 2"),
    "bcic2b": dict(module=BCICIV2b, train=[0, 1, 2], test=[3, 4], label="SESSIONS 4-5"),
    "zhou2016": dict(module=Zhou2016LOSO, train=[0, 1], test=[2], label="SESSION 3"),
}


def _subject_sessions(dataset_name, windows, subject):
    by_subject = windows.split("subject")
    subject_ds = by_subject.get(str(subject), by_subject.get(subject))
    if dataset_name == "bcic2a":
        return list(_get_2a_train_test_sessions(subject_ds))
    if dataset_name == "bcic2b":
        return _get_ordered_sessions(subject_ds)
    return _get_three_sessions(subject_ds)


def load_sessions(dataset_name, preprocessing):
    """Return {subject: [(X, y) per session]} in chronological session order."""
    module = DATASETS[dataset_name]["module"]
    if dataset_name == "zhou2016":
        windows = load_zhou2016(module.all_subject_ids, preprocessing)
    else:
        windows = load_bcic4(subject_ids=module.all_subject_ids,
                             dataset=dataset_name[-2:], preprocessing_dict=preprocessing)
    sessions = {}
    for subject in module.all_subject_ids:
        arrays = []
        for session in _subject_sessions(dataset_name, windows, subject):
            X, y = BaseDataModule._dataset_to_arrays(session)
            arrays.append((X.astype(np.float64), y.astype(np.int64)))
        sessions[subject] = arrays
    return sessions


def _concat(arrays, idx):
    return (np.concatenate([arrays[i][0] for i in idx]),
            np.concatenate([arrays[i][1] for i in idx]))


def aligned_source(sessions, target, split):
    """EA every source subject with the reference of its own training sessions."""
    Xs, ys, Xv, yv = [], [], [], []
    for subject, arrays in sessions.items():
        if subject == target:
            continue
        X_train, y_train = _concat(arrays, split["train"])
        X_val, y_val = _concat(arrays, split["test"])
        ref = EA_reference(X_train)
        Xs.append(np.real(np.einsum("cd,ndt->nct", ref, X_train)))
        Xv.append(np.real(np.einsum("cd,ndt->nct", ref, X_val)))
        ys.append(y_train)
        yv.append(y_val)
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(Xv), np.concatenate(yv))


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
    """tl/ttime.py :: train_target source stage, with best source-val selection."""
    fix_random_seed(seed)
    netF, netC, feature_dim = backbone_net(
        args.chn, args.time_sample_num, args.sample_rate, args.class_num)
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
            print(f"    seed {seed} epoch {epoch + 1}/{args.max_epoch} "
                  f"val_acc={val_acc * 100:.2f}% (best {best_acc * 100:.2f}% "
                  f"@ {best_epoch + 1})", flush=True)

    base_network.load_state_dict(best_state)
    base_network.eval()
    print(f"  source model seed {seed}: best source-validation accuracy "
          f"{best_acc * 100:.2f}% at epoch {best_epoch + 1}", flush=True)
    return base_network, feature_dim


class EnsembleForward(nn.Module):
    """Single-trial inference of the T-TIME ensemble (latency only)."""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return torch.stack(
            [torch.softmax(m(x)[1], dim=1) for m in self.models]).mean(0)


class BFTForward(nn.Module):
    """Single-trial BFT-D inference: K masked views weighted by the ranker."""

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


def run(config, method, gpu_id, dataset_name="bcic2a"):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    split = DATASETS[dataset_name]
    module = split["module"]
    preprocessing = config["preprocessing"][dataset_name]
    hp = config[method]
    args = argparse.Namespace(
        chn=module.channels, class_num=module.classes,
        sample_rate=preprocessing["sfreq"], lr=config["lr"],
        batch_size=config["batch_size"], max_epoch=config["max_epoch"],
        align=True, log_every=config.get("log_every_n_epochs", 0), **hp,
    )

    model_name = {"ttime": "T-TIME", "bft": "BFT"}[method]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = (Path(__file__).resolve().parent /
                  f"results/{model_name}_{dataset_name}_loso_seed-{config['seed']}_GPU{gpu_id}_{timestamp}")
    (result_dir / "confmats").mkdir(parents=True, exist_ok=True)
    with open(result_dir / "config.yaml", "w") as f:
        yaml.dump({"method": method, "dataset": dataset_name, **config}, f, default_flow_style=False)

    sessions = load_sessions(dataset_name, preprocessing)
    args.time_sample_num = next(iter(sessions.values()))[0][0].shape[-1]

    subject_ids = config["subject_ids"]
    if subject_ids == "all":
        subject_ids = module.all_subject_ids
    test_accs, test_losses, test_kappas = [], [], []
    train_times, test_times, response_times, confmats = [], [], [], []

    for target in subject_ids:
        print(f"\n>>> {model_name} | target subject {target}", flush=True)
        Xs, ys, Xv, yv = aligned_source(sessions, target, split)
        train_loader = to_loader(Xs, ys, args.batch_size, True, True, device)
        val_loader = to_loader(Xv, yv, args.batch_size * 3, False, False, device)
        # Target evaluation sessions, in recording order, labels kept aside.
        X_stream, y_true = _concat(sessions[target], split["test"])

        st_train = time.time()
        if method == "ttime":
            seeds = [config["seed"] + m for m in range(args.ensemble_size)]
            source_models = []
            for seed in seeds:
                model, _ = train_source_model(seed, train_loader, val_loader, args, device)
                source_models.append(model)
            train_times.append((time.time() - st_train) / 60)

            st_test = time.time()
            preds = []
            for seed, source_model in zip(seeds, source_models):
                fix_random_seed(seed)
                preds.append(TTIME(X_stream, clone_model(source_model), args, device))
            preds = np.stack(preds)               # (M, n_trials, n_classes)
            y_hat = SML_online_ensemble(preds, args.class_num)
            probs = preds.mean(0)
            test_times.append(time.time() - st_test)
            param_count = sum(p.numel() for m in source_models for p in m.parameters())
            latency_model = EnsembleForward(source_models)
        else:
            model, feature_dim = train_source_model(
                config["seed"], train_loader, val_loader, args, device)
            fix_random_seed(config["seed"])
            ranker = ReliabilityRanker(feature_dim).to(device)
            train_ranker(model[0], model[1], ranker, train_loader, args, device)
            train_times.append((time.time() - st_train) / 60)

            st_test = time.time()
            probs = BFT_func(X_stream, clone_model(model), ranker, args, device)
            y_hat = probs.argmax(1)
            test_times.append(time.time() - st_test)
            param_count = (sum(p.numel() for p in model.parameters())
                           + sum(p.numel() for p in ranker.parameters()))
            latency_model = BFTForward(model, ranker, args.K, args.temperature)

        acc = accuracy_score(y_true, y_hat)
        kappa = cohen_kappa_score(y_true, y_hat)
        # Negative log-likelihood of the (averaged) online probabilities.
        loss = float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true] + 1e-12)))
        test_accs.append(acc)
        test_kappas.append(kappa)
        test_losses.append(loss)
        cm = confusion_matrix(y_true, y_hat, labels=list(range(args.class_num)))
        confmats.append(cm)
        plot_confusion_matrix(
            cm, save_path=result_dir / f"confmats/confmat_subject_{target}.png",
            class_names=module.class_names,
            title=f"Confusion Matrix – Subject {target}",
        )
        response_times.append(measure_latency(
            latency_model, (1, 1, args.chn, args.time_sample_num), device="cpu"))
        latency_model.to(device)
        print(f"\nTARGET SUBJECT {target} {split['label']} RESULT | acc={acc * 100:.2f}% | "
              f"loss={loss:.4f} | kappa={kappa:.4f} | test_time={test_times[-1]:.2f}s\n",
              flush=True)

    write_summary(result_dir, model_name, f"{dataset_name}_loso", list(subject_ids), param_count,
                  test_accs, test_losses, test_kappas, train_times, test_times,
                  response_times, primary_test_label=split["label"])
    plot_confusion_matrix(
        np.mean(confmats, axis=0), save_path=result_dir / "confmats/avg_confusion_matrix.png",
        class_names=module.class_names, title="Average Confusion Matrix",
    )
    print((result_dir / "results.txt").read_text(encoding="utf-8"), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ttime", "bft"], required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="bcic2a")
    parser.add_argument("--gpu_id", type=int, default=0)
    cli = parser.parse_args()
    with open(CONFIG_DIR / "tta_baselines.yaml") as f:
        config = yaml.safe_load(f)
    run(config, cli.method, cli.gpu_id, cli.dataset)


if __name__ == "__main__":
    main()
