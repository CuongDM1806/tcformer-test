"""RMETNet reproduced under the authors' own cross-subject protocol.

This script exists only to check whether the RMETNet numbers published in
PLOS ONE (BCI IV-2a: 71.39%, kappa 0.59; BCI IV-2b: 80.93%, kappa 0.54) can be
approached with the same network when it is evaluated the way the authors'
released code evaluates it. It is NOT the protocol of the SSM-DAN paper (see
rmetnet_baseline.py for that).

What the authors' code does (maingcn-mmdloss-2a-matrics.py, strategy 2;
bciciv2a/LoaderBCICIV2a.py :: BCICIV2Amatdata; tools/utilstrva.py :: fittvmmd):

* data: one session group per subject. For BCI IV-2a the ``gdf`` data are
  generated from the A0xT.gdf files only (session T); for BCI IV-2b the paper
  keeps the three feedback sessions (sessions 3-5, 480 trials). Window 0.5-4.5 s
  after the cue (paper), no extra filtering;
* source: the other subjects. The 80/20 train/valid split of BCICIV2Amatdata is
  merged again by strategy 2, so the model trains on all source trials;
* target: the held-out subject's trials are at the same time
    - the target domain of the MMD loss (zip of the source loader with the
      cycled test loader; the labels are not used by the MMD),
    - the target domain of the Riemannian alignment (C_common from the source
      and *test* Riemannian means),
    - the "validation" set evaluated after every epoch, with its labels;
* z-score: global mean and standard deviation of the source trials;
* optimisation: Adam lr 2e-4, betas (0.5, 0.999), weight decay 0.075, batch 64,
  loss CE + MMD; 500 epochs as reported in the paper (the script's default is
  1000); random_state 44 (the script's default);
* reported accuracy: the mean of the five highest per-epoch test accuracies
  over all epochs (``sorted(val_acc, reverse=True)[:5]``); reported kappa: from
  the confusion matrix of the last epoch.

Reported here per subject: that "top-5" accuracy (the authors' number), the
best single epoch, and the last epoch (no selection at all), so the effect of
the selection rule is visible.
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

from rmetnet_baseline import predict, to_tensor
from tta_baselines import DATASETS, _concat, fix_random_seed, load_sessions
from utils.metrics import write_summary
from utils.rmetnet import MMDLoss, Net, align, common_reference, compute_covariance_matrix

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def split_fold(sessions, target, session_idx):
    """All source subjects' trials, and the target subject's trials."""
    Xs, ys = [], []
    for subject, arrays in sessions.items():
        if subject == target:
            continue
        X, y = _concat(arrays, session_idx)
        Xs.append(X)
        ys.append(y)
    X_test, y_test = _concat(sessions[target], session_idx)
    return np.concatenate(Xs), np.concatenate(ys), X_test, y_test


def train_and_track(Xs, Rs, ys, Xt, Rt, yt, args, device):
    """fittvmmd: CE + MMD(source, test) every batch, test scored every epoch."""
    fix_random_seed(args.seed)
    model = Net(num_classes=args.class_num, num_channels=args.chn,
                sampling_rate=args.sample_rate, num_samples=args.time_sample_num).to(device)
    criterion = nn.CrossEntropyLoss()
    mmdloss = MMDLoss().to(device)
    optimizer = optim.Adam(params=list(model.parameters()), lr=args.lr,
                           betas=tuple(args.betas), weight_decay=args.weight_decay)

    train_loader = Data.DataLoader(Data.TensorDataset(Xs, Rs, ys),
                                   batch_size=args.batch_size, shuffle=True)
    test_loader = Data.DataLoader(Data.TensorDataset(Xt, Rt),
                                  batch_size=args.batch_size, shuffle=True)
    # As in fittvmmd: the shorter loader is cycled, once, before the epoch loop.
    if len(train_loader) > len(test_loader):
        test_iter, train_iter, loop_len = itertools.cycle(test_loader), None, len(train_loader)
    else:
        test_iter, train_iter, loop_len = None, itertools.cycle(train_loader), len(test_loader)

    epoch_accs, last_pred = [], None
    for epoch in range(args.epochs):
        model.train()
        source_batches = train_iter if train_iter is not None else train_loader
        target_batches = test_iter if test_iter is not None else test_loader
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

        # fittvmmd's "validation" loop on the labeled test subject.
        last_pred = predict(model, Xt, Rt, device).argmax(1)
        acc = (last_pred == yt).float().mean().item()
        epoch_accs.append(acc)
        if args.log_every and (epoch + 1) % args.log_every == 0:
            print(f"    epoch {epoch + 1}/{args.epochs} | cls={loss.item():.4f} | "
                  f"mmd={domainloss.item():.4f} | test_acc={acc * 100:.2f}% "
                  f"(best so far {max(epoch_accs) * 100:.2f}%)", flush=True)
    return np.array(epoch_accs), last_pred.numpy()


def run(config, gpu_id, dataset_name):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    module = DATASETS[dataset_name]["module"]
    preprocessing = config["preprocessing"][dataset_name]
    session_idx = config["sessions"][dataset_name]
    hp = dict(config["rmetnet"])
    args = argparse.Namespace(chn=module.channels, class_num=module.classes,
                              sample_rate=preprocessing["sfreq"], seed=config["seed"],
                              log_every=config.get("log_every_n_epochs", 0), **hp)
    print(f"RMETNet (authors' protocol) | {dataset_name} | sessions {session_idx} | "
          f"{args.epochs} epochs | seed {args.seed}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = (Path(__file__).resolve().parent /
                  f"results/RMETNet-authors-protocol_{dataset_name}_seed-{args.seed}_GPU{gpu_id}_{timestamp}")
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "config.yaml", "w") as f:
        yaml.dump({"dataset": dataset_name, **config}, f, default_flow_style=False)

    sessions = load_sessions(dataset_name, preprocessing)
    args.time_sample_num = next(iter(sessions.values()))[0][0].shape[-1]

    subject_ids = config["subject_ids"]
    if subject_ids == "all":
        subject_ids = module.all_subject_ids
    top5, best, last, kappas, losses, train_times, test_times, rows = [], [], [], [], [], [], [], []

    for target in subject_ids:
        print(f"\n>>> RMETNet (authors' protocol) | target subject {target}", flush=True)
        Xs, ys, Xt, yt = split_fold(sessions, target, session_idx)

        # Riemannian alignment with the test subject as target domain.
        cov_s, cov_t = compute_covariance_matrix(Xs), compute_covariance_matrix(Xt)
        C_common = common_reference(cov_s, cov_t)
        Rs, Rt = to_tensor(align(cov_s, C_common)), to_tensor(align(cov_t, C_common))
        mean, std = Xs.mean(), Xs.std()
        Es = to_tensor((Xs - mean) / std).unsqueeze(1)
        Et = to_tensor((Xt - mean) / std).unsqueeze(1)
        print(f"  source {tuple(Xs.shape)}, target/test {tuple(Xt.shape)}", flush=True)

        st = time.time()
        accs, last_pred = train_and_track(Es, Rs, to_tensor(ys, torch.long),
                                          Et, Rt, to_tensor(yt, torch.long), args, device)
        train_times.append((time.time() - st) / 60)
        test_times.append(0.0)

        acc_top5 = float(np.mean(np.sort(accs)[::-1][:5]))
        kappa_last = cohen_kappa_score(yt, last_pred)
        top5.append(acc_top5)
        best.append(float(accs.max()))
        last.append(float(accs[-1]))
        kappas.append(kappa_last)
        losses.append(float("nan"))
        rows.append(f"Subject {target}: top-5 mean {acc_top5 * 100:.2f}% | best epoch "
                    f"{accs.max() * 100:.2f}% (epoch {int(accs.argmax()) + 1}) | last epoch "
                    f"{accs[-1] * 100:.2f}% | last-epoch kappa {kappa_last:.4f}")
        print("  " + rows[-1], flush=True)

    param_count = sum(p.numel() for p in Net(num_classes=args.class_num, num_channels=args.chn,
                                             sampling_rate=args.sample_rate,
                                             num_samples=args.time_sample_num).parameters())
    # results.txt: the authors' reported quantities (top-5 accuracy, last-epoch kappa).
    write_summary(result_dir, "RMETNet (authors' protocol, top-5 epochs on test)",
                  f"{dataset_name}_authors_protocol", list(subject_ids), param_count,
                  top5, losses, kappas, train_times, test_times, [0.0] * len(top5),
                  primary_test_label="TARGET")
    summary = [
        "RMETNet under the authors' protocol (target = test subject, see docstring)",
        f"dataset {dataset_name}, sessions {session_idx}, epochs {args.epochs}, seed {args.seed}",
        "",
        *rows,
        "",
        f"Mean top-5 accuracy (authors' reported metric): {np.mean(top5) * 100:.2f} +- {np.std(top5) * 100:.2f}",
        f"Mean best-epoch accuracy:                      {np.mean(best) * 100:.2f} +- {np.std(best) * 100:.2f}",
        f"Mean last-epoch accuracy (no selection):       {np.mean(last) * 100:.2f} +- {np.std(last) * 100:.2f}",
        f"Mean last-epoch kappa (authors' reported):     {np.mean(kappas):.3f} +- {np.std(kappas):.3f}",
    ]
    (result_dir / "authors_protocol_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bcic2a", "bcic2b"], default="bcic2a")
    parser.add_argument("--gpu_id", type=int, default=0)
    cli = parser.parse_args()
    with open(CONFIG_DIR / "rmetnet_authors_protocol.yaml") as f:
        config = yaml.safe_load(f)
    run(config, cli.gpu_id, cli.dataset)


if __name__ == "__main__":
    main()
