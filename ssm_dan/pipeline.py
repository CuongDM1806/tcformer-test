"""Train and evaluate one LOSO fold (one target subject).

Lightning methods (SSM-DAN, its variants and the end-to-end baselines):

1. Stage 1: train on the sources (and, with domain adaptation, the unlabeled
   earlier target sessions); keep the checkpoint with the best accuracy on the
   labeled source validation sessions. Target labels are never read.
2. Optionally score the same weights without IM-TTA (``no_tta_*``).
3. Stage 2 (SSM-DAN): label-free adaptation on the evaluation trials.
4. Score the evaluation session(s) (``test_*``) and the auxiliary all-session
   view (``all_sessions_*``); measure single-trial CPU latency.
"""
import gc
import os
import time
from pathlib import Path

if os.environ.get("MPLBACKEND", "").startswith("module://matplotlib_inline"):
    os.environ["MPLBACKEND"] = "Agg"   # notebook backends are absent in subprocesses

import numpy as np
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from . import config as cfg
from .data import get_datamodule_cls
from .models import get_model_cls
from .utils.latency import measure_latency
from .utils.plotting import plot_confusion_matrix, plot_curve
from .utils.seed import seed_everything


class MetricsCallback(Callback):
    """Collect per-epoch train/validation metrics and print them every n epochs."""

    def __init__(self, log_every_n_epochs=0):
        super().__init__()
        self.log_every_n_epochs = max(0, int(log_every_n_epochs))
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def _collect(self, trainer, names):
        for name in names:
            if name in trainer.callback_metrics:
                self.history[name].append(trainer.callback_metrics[name].cpu().item())

    def on_train_epoch_end(self, trainer, pl_module):
        self._collect(trainer, ("train_loss", "train_acc"))

    def on_validation_epoch_end(self, trainer, pl_module):
        self._collect(trainer, ("val_loss", "val_acc"))
        epoch = trainer.current_epoch + 1
        if self.log_every_n_epochs and (epoch % self.log_every_n_epochs == 0 or epoch == trainer.max_epochs):
            values = []
            for name in ("train_loss", "train_acc", "val_loss", "val_acc"):
                if name in trainer.callback_metrics:
                    value = trainer.callback_metrics[name].detach().cpu().item()
                    values.append(f"{name}={value * 100:.2f}%" if name.endswith("acc") else f"{name}={value:.4f}")
            pl_module.print(f"Epoch {epoch}/{trainer.max_epochs} | " + " | ".join(values))


def _trainer(config, gpu, callbacks):
    if torch.cuda.is_available():
        accelerator, devices = "auto", [gpu]
    else:
        accelerator, devices = "cpu", 1
    return Trainer(
        max_epochs=config["max_epochs"],
        accumulate_grad_batches=config["preprocessing"].get("accumulate_grad_batches", 1),
        devices=devices,
        accelerator=accelerator,
        strategy="auto",
        num_sanity_val_steps=0,
        logger=False,
        enable_progress_bar=config.get("enable_progress_bar", False),
        enable_checkpointing=True,
        callbacks=callbacks,
    )


def _scores(result):
    return {"acc": float(result["test_acc"]), "loss": float(result["test_loss"]),
            "kappa": float(result["test_kappa"])}


def run_lightning_fold(config, subject_id, out_dir: Path, gpu=0):
    datamodule_cls = get_datamodule_cls(config["dataset"])
    model_cls = get_model_cls(config["model"])
    kwargs = cfg.model_kwargs(config)
    kwargs.update(n_channels=datamodule_cls.channels, n_classes=datamodule_cls.classes)
    if kwargs.get("osc_branch", "none") != "none":
        kwargs.setdefault("osc_sfreq", config["preprocessing"]["sfreq"])   # states are tuned in Hz

    seed_everything(config["seed"])
    metrics = MetricsCallback(config.get("log_every_n_epochs", 10))
    checkpoint_name = f"S{subject_id:02d}"
    for stale in (out_dir / "checkpoints").glob(f"{checkpoint_name}*.ckpt"):
        stale.unlink()   # a rerun must not pick up an older checkpoint
    best = ModelCheckpoint(dirpath=out_dir / "checkpoints", filename=checkpoint_name, monitor="val_acc",
                           mode="max", save_top_k=1, save_last=False, auto_insert_metric_name=False)
    trainer = _trainer(config, gpu, [metrics, best])
    datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)
    model = model_cls(**kwargs)
    param_count = sum(p.numel() for p in model.parameters())

    # ---------------- Stage 1 ----------------
    started = time.time()
    trainer.fit(model, datamodule=datamodule)
    train_minutes = (time.time() - started) / 60
    if not best.best_model_path:
        raise RuntimeError(f"No source-validation checkpoint was saved for subject {subject_id}.")
    model.load_state_dict(torch.load(best.best_model_path, map_location="cpu", weights_only=False)["state_dict"])
    best_val_acc = float(best.best_model_score)
    print(f"Best source-validation checkpoint | subject {subject_id} | val_acc={best_val_acc * 100:.2f}%",
          flush=True)

    # Reuse the prepared loaders (trainer.test(datamodule=...) would reload every subject).
    test_loader = datamodule.test_dataloader()
    all_loader = datamodule.all_target_dataloader() if datamodule.all_target_dataset is not None else None
    datamodule.dataset = datamodule.train_dataset = datamodule.val_dataset = None
    datamodule.target_dataset = datamodule.all_target_dataset = None
    gc.collect()

    record = {"subject_id": subject_id, "best_val_acc": best_val_acc, "train_time_min": train_minutes,
              "param_count": int(param_count)}

    # ---------------- same weights without IM-TTA ----------------
    if (config.get("evaluate_without_im_tta") and hasattr(model, "adapt_to_target")
            and getattr(model, "im_tta_steps", 0) > 0):
        reference = model_cls(**kwargs)
        reference.load_state_dict(model.state_dict())
        reference.to(next(model.parameters()).device)
        reference.trainer = trainer          # adapt_to_target prints through the trainer
        reference.im_tta_steps = 0
        reference.adapt_to_target(test_loader)   # AutoDIAL target statistics are still re-estimated
        no_tta = _scores(trainer.test(reference, dataloaders=test_loader, verbose=False)[0])
        record.update({f"no_tta_{k}": v for k, v in no_tta.items()})
        print(f"TARGET SUBJECT {subject_id} WITHOUT IM-TTA | acc={no_tta['acc'] * 100:.2f}% | "
              f"kappa={no_tta['kappa']:.4f}", flush=True)
        del reference
        gc.collect()

    # ---------------- Stage 2 and evaluation ----------------
    if hasattr(model, "adapt_to_target"):
        model.adapt_to_target(test_loader)    # labels in the loader are never read
    if hasattr(model, "autodial_alpha") and model.autodial_alpha() is not None:
        record["autodial_alpha"] = model.autodial_alpha()
    interpretation = getattr(getattr(model, "model", None), "interpretation", None)
    if callable(interpretation) and interpretation() is not None:
        record["oscillatory_ssm"] = interpretation()
    started = time.time()
    test = _scores(trainer.test(model, dataloaders=test_loader, verbose=False)[0])
    record["test_time_s"] = time.time() - started
    record.update({f"test_{k}": v for k, v in test.items()})
    record["confusion_counts"] = model.test_counts.numpy().astype(int).tolist()
    confusion_pct = model.test_confmat.numpy().copy()
    print(f"TARGET SUBJECT {subject_id} {datamodule_cls.primary_test_label} | acc={test['acc'] * 100:.2f}% | "
          f"kappa={test['kappa']:.4f} | loss={test['loss']:.4f}", flush=True)

    if all_loader is not None:
        aux = _scores(trainer.test(model, dataloaders=all_loader, verbose=False)[0])
        record.update({f"all_sessions_{k}": v for k, v in aux.items()})
        print(f"TARGET SUBJECT {subject_id} {datamodule_cls.auxiliary_test_label} (auxiliary) | "
              f"acc={aux['acc'] * 100:.2f}%", flush=True)

    sample_x, _ = datamodule.test_dataset[0]
    record["cpu_latency_ms"] = measure_latency(model, (1, *sample_x.shape), device="cpu")

    figures = out_dir / "figures"
    plot_confusion_matrix(confusion_pct, save_path=figures / f"S{subject_id:02d}_confusion.png",
                          class_names=datamodule_cls.class_names, title=f"Subject {subject_id}")
    history = metrics.history
    for name in ("loss", "acc"):
        if history[f"train_{name}"] and history[f"val_{name}"]:
            plot_curve(history[f"train_{name}"], history[f"val_{name}"], name.title(), subject_id,
                       figures / f"S{subject_id:02d}_{name}.png")
    record["class_names"] = list(datamodule_cls.class_names)
    return record


def run_baseline_fold(config, subject_id, sessions, gpu=0):
    """RMETNet / T-TIME / BFT (own training loops, no Lightning)."""
    from .baselines import rmetnet, tta

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    runner = {"rmetnet": rmetnet, "tta": tta}[config["runner"]]
    record = runner.run_fold(config, config["dataset"], subject_id, sessions, device)
    record["subject_id"] = subject_id
    record["class_names"] = list(get_datamodule_cls(config["dataset"]).class_names)
    return record


def plot_baseline_confusion(record, out_dir: Path):
    counts = np.asarray(record["confusion_counts"], dtype=float)
    percent = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None) * 100.0
    plot_confusion_matrix(percent, save_path=out_dir / "figures" / f"S{record['subject_id']:02d}_confusion.png",
                          class_names=record["class_names"], title=f"Subject {record['subject_id']}")
