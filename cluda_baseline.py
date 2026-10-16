"""Chronological LOSO evaluation of CLUDA (BCI IV-2a, IV-2b, Zhou2016).

CLUDA is the authors' implementation (utils/cluda.py) run under the same
chronological protocol as SSM-DAN. Session split per dataset
(source training / source validation; target unlabeled adaptation / evaluation):
BCI IV-2a 1 / 2, BCI IV-2b 1-3 / 4-5, Zhou2016 1-2 / 3.

* source domain: the training sessions of all other subjects train the model;
  their validation sessions are the labeled split used only to select the
  checkpoint (highest source validation accuracy), as for every other method
  in the paper (the authors also select on the source validation split);
* target domain: the earlier sessions of the target subject are the unlabeled
  target data of the contrastive, nearest-neighbour and adversarial losses;
  the evaluation sessions are used only at test time, and their labels only
  for scoring;
* input: global z-score with the mean and standard deviation of the source
  training data; each trial is a C-variate sequence (N x L x C) as in the
  authors' SensorDataset;
* preprocessing (windows, Zhou2016 8-30 Hz filter) is the paper's per-dataset
  setting.

Training follows the authors' main/train.py and the CLUDA step of
main/algorithms.py with their configuration for multivariate sensor data
(HAR/HHAR in the README): TCN kernel 7, dilation 3, 64 channels, dropout 0.2,
MLP hidden 128, queue 8192, momentum 0.99, batch 32, 10000 steps, Adam lr 1e-4,
betas (0.5, 0.99), weight decay 1e-4, loss weights src/trg contrastive 0.05,
nearest-neighbour 0.1, discriminator 1, prediction 1, GRL coefficient
2 / (1 + exp(-10 step / 1000)) - 1, evaluation every 100 steps. CLUDA classifies the last TCN output step; the
HAR configuration has 3 levels (receptive field 157 samples) for 128-sample
windows, so the depth is raised to 5 levels (1453 samples) to span the
1000-sample EEG trials; with 3 levels the model only sees the last 0.6 s.
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
import torch.nn.functional as F
import torch.utils.data as Data
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

from rmetnet_baseline import split_fold, to_tensor
from tta_baselines import DATASETS, fix_random_seed, load_sessions
from utils.cluda import CLUDA_NN, get_augmenter, receptive_field
from utils.latency import measure_latency
from utils.metrics import write_summary
from utils.plotting import plot_confusion_matrix

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@torch.no_grad()
def predict(model, X, device, batch_size=256):
    """X: (N, L, C) sequences -> class logits."""
    model.eval()
    logits = [model.predict(X[i:i + batch_size].to(device)).cpu()
              for i in range(0, len(X), batch_size)]
    return torch.cat(logits)


class CLUDAInference(nn.Module):
    """Single raw trial (1, 1, C, T) -> prediction, including the z-score (latency only)."""

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.mean, self.std = float(mean), float(std)

    def forward(self, x):
        return self.model.predict(((x[:, 0] - self.mean) / self.std).transpose(1, 2))


def train_fold(Xs, ys, Xa, Xv, yv, args, device):
    """Authors' train.py loop with source-validation checkpoint selection."""
    fix_random_seed(args.seed)
    model = CLUDA_NN(num_inputs=Xs.shape[-1], output_dim=args.class_num,
                     num_channels=args.num_channels_TCN, mlp_hidden_dim=args.hidden_dim_MLP,
                     use_batch_norm=args.use_batch_norm, kernel_size=args.kernel_size_TCN,
                     stride=args.stride_TCN, dilation_factor=args.dilation_factor_TCN,
                     dropout=args.dropout, K=args.queue_size, m=args.momentum).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay, betas=(0.5, 0.99))
    augmenter = get_augmenter(Xs.shape[1], Xs.shape[2])
    criterion_CL = nn.CrossEntropyLoss()

    # The model does not support incomplete batches (queue pointer).
    source_loader = Data.DataLoader(Data.TensorDataset(Xs, ys), batch_size=args.batch_size,
                                    shuffle=True, drop_last=True)
    target_loader = Data.DataLoader(Data.TensorDataset(Xa), batch_size=args.batch_size,
                                    shuffle=True, drop_last=True)

    def augment(x):
        return augmenter(x, torch.ones_like(x, dtype=torch.long))[0]

    count_step, started = 0, time.time()
    best_acc, best_state, best_step = -1.0, None, -1
    while count_step < args.num_steps:
        target_iter = iter(target_loader)
        for seq_src, label_src in source_loader:
            try:
                (seq_trg,) = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                (seq_trg,) = next(target_iter)
            model.train()
            seq_src, label_src, seq_trg = seq_src.to(device), label_src.to(device), seq_trg.to(device)

            p = float(count_step) / 1000
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            (output_s, target_s, output_t, target_t, output_ts, target_ts,
             output_disc, target_disc, pred_s) = model(
                augment(seq_src), augment(seq_src), augment(seq_trg), augment(seq_trg), alpha)

            loss_s = criterion_CL(output_s, target_s)
            loss_t = criterion_CL(output_t, target_t)
            loss_ts = criterion_CL(output_ts, target_ts)
            loss_disc = F.binary_cross_entropy(output_disc, target_disc)
            src_cls_loss = F.cross_entropy(pred_s, label_src)
            loss = (args.weight_loss_src * loss_s + args.weight_loss_trg * loss_t
                    + args.weight_loss_ts * loss_ts + args.weight_loss_disc * loss_disc
                    + args.weight_loss_pred * src_cls_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count_step += 1

            if count_step % args.checkpoint_freq == 0:
                val_acc = (predict(model, Xv, device).argmax(1) == yv).float().mean().item()
                if val_acc > best_acc:
                    best_acc, best_step = val_acc, count_step
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if args.log_every and count_step % args.log_every == 0:
                    print(f"    step {count_step}/{args.num_steps} | pred={src_cls_loss.item():.4f} | "
                          f"cl_s={loss_s.item():.4f} | cl_t={loss_t.item():.4f} | "
                          f"nn_ts={loss_ts.item():.4f} | disc={loss_disc.item():.4f} | "
                          f"val_acc={val_acc * 100:.2f}% (best {best_acc * 100:.2f}% "
                          f"@ {best_step}) | {(time.time() - started) / 60:.1f} min", flush=True)
            if count_step >= args.num_steps:
                break

    model.load_state_dict(best_state)
    model.eval()
    print(f"  best source-validation accuracy {best_acc * 100:.2f}% at step {best_step}",
          flush=True)
    return model


def run(config, gpu_id, dataset_name="bcic2a"):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    split = DATASETS[dataset_name]
    module = split["module"]
    preprocessing = config["preprocessing"][dataset_name]
    hp = dict(config["cluda"])
    hp["num_channels_TCN"] = [int(c) for c in str(hp["num_channels_TCN"]).split("-")]
    args = argparse.Namespace(class_num=module.classes, seed=config["seed"],
                              log_every=config.get("log_every_n_steps", 0), **hp)
    print(f"CLUDA | {dataset_name} | {args.num_steps} steps", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_dir = (Path(__file__).resolve().parent /
                  f"results/CLUDA_{dataset_name}_loso_seed-{config['seed']}_GPU{gpu_id}_{timestamp}")
    (result_dir / "confmats").mkdir(parents=True, exist_ok=True)
    with open(result_dir / "config.yaml", "w") as f:
        yaml.dump({"method": "cluda", "dataset": dataset_name, **config}, f,
                  default_flow_style=False)

    sessions = load_sessions(dataset_name, preprocessing)
    n_channels, n_samples = next(iter(sessions.values()))[0][0].shape[1:]
    # CLUDA classifies the last TCN output step, so its receptive field must
    # span the whole trial (the authors' sensor configurations do).
    field = receptive_field(len(args.num_channels_TCN), args.kernel_size_TCN,
                            args.dilation_factor_TCN)
    print(f"  TCN receptive field {field} samples, trial length {n_samples}", flush=True)
    assert field >= n_samples, "TCN receptive field shorter than the trial"

    subject_ids = config["subject_ids"]
    if subject_ids == "all":
        subject_ids = module.all_subject_ids
    test_accs, test_losses, test_kappas = [], [], []
    train_times, test_times, response_times, confmats = [], [], [], []

    for target in subject_ids:
        print(f"\n>>> CLUDA | target subject {target}", flush=True)
        Xs, ys, Xv, yv, X_adapt, X_test, y_test = split_fold(sessions, target, split)

        # Global z-score with the source training statistics; N x L x C sequences.
        mean, std = Xs.mean(), Xs.std()
        seq = {k: to_tensor((v - mean) / std).transpose(1, 2)
               for k, v in dict(s=Xs, v=Xv, a=X_adapt, t=X_test).items()}
        print(f"  source train {tuple(Xs.shape)}, source val {tuple(Xv.shape)}, "
              f"target adaptation (unlabeled) {tuple(X_adapt.shape)}, "
              f"target test {tuple(X_test.shape)}", flush=True)

        st_train = time.time()
        model = train_fold(seq["s"], to_tensor(ys, torch.long), seq["a"], seq["v"],
                           to_tensor(yv, torch.long), args, device)
        train_times.append((time.time() - st_train) / 60)

        st_test = time.time()
        logits = predict(model, seq["t"], device)
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
        # Inference parameters only (the momentum key encoder, projector and
        # discriminator are training-time modules).
        param_count = sum(p.numel() for m in (model.encoder_q, model.predictor)
                          for p in m.parameters())
        response_times.append(measure_latency(
            CLUDAInference(model, mean, std), (1, 1, n_channels, n_samples), device="cpu"))
        print(f"\nTARGET SUBJECT {target} {split['label']} RESULT | acc={acc * 100:.2f}% | "
              f"loss={loss:.4f} | kappa={kappa:.4f} | test_time={test_times[-1]:.2f}s\n",
              flush=True)

    write_summary(result_dir, "CLUDA", f"{dataset_name}_loso", list(subject_ids), param_count,
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
    with open(CONFIG_DIR / "cluda.yaml") as f:
        config = yaml.safe_load(f)
    run(config, cli.gpu_id, cli.dataset)


if __name__ == "__main__":
    main()
