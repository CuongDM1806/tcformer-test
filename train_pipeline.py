import gc
import os, time, yaml
import shutil

# Colab exports its notebook-only matplotlib backend to subprocesses. The
# isolated Python 3.10 environment does not contain that backend, so use a
# headless backend before PyTorch Lightning/TorchMetrics imports matplotlib.
if os.environ.get("MPLBACKEND", "").startswith("module://matplotlib_inline"):
    os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
# from torchviz import make_dot  # optional for graph visualization

from utils.plotting import plot_confusion_matrix, plot_curve
from utils.metrics  import MetricsCallback, write_summary
from utils.latency  import measure_latency
from utils.misc     import visualize_model_graph, show_gpu_info

from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_model_cls import get_model_cls
from utils.seed import seed_everything

# Set visible GPUs
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Define the path to the configuration directory
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def _write_no_tta_summary(result_dir, completed_ids, no_tta_accs, no_tta_kappas):
    if not no_tta_accs:
        return
    lines = [
        "Same trained models scored without IM-TTA (AutoDIAL target statistics "
        "re-estimated on the evaluation trials, as in the paper's variant).",
        f"Subject IDs: {completed_ids}",
    ]
    lines += [
        f"Subject {subject_id} => Acc: {acc:.4f}, Kappa: {kappa:.4f}"
        for subject_id, acc, kappa in zip(
            completed_ids, no_tta_accs, no_tta_kappas
        )
    ]
    lines += [
        f"Average Accuracy without TTA: {np.mean(no_tta_accs) * 100:.2f} "
        f"± {np.std(no_tta_accs) * 100:.2f}",
        f"Average Kappa without TTA:    {np.mean(no_tta_kappas):.3f} "
        f"± {np.std(no_tta_kappas):.3f}",
    ]
    (result_dir / "results_without_tta.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _copy_incremental_results(result_dir):
    """Expose cumulative fold results at a stable Kaggle output path."""
    snapshot_path = os.environ.get("INCREMENTAL_RESULTS_PATH", "").strip()
    if not snapshot_path:
        return
    snapshot = Path(snapshot_path)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result_dir / "results.txt", snapshot)
    no_tta_path = result_dir / "results_without_tta.txt"
    if no_tta_path.exists():
        shutil.copy2(
            no_tta_path,
            snapshot.with_name(f"{snapshot.stem}_without_tta{snapshot.suffix}"),
        )
    print(f">>> Incremental fold results saved to {snapshot}", flush=True)

# Main training and testing pipeline
def train_and_test(config):
     # Create result and checkpoints directories
    model_name = config["model"]
    dataset_name = config["dataset_name"]
    seed = config["seed"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M") # Format: YYYYMMDD_HHMM (e.g., 20250517_1530)
    result_dir = ( 
        Path(__file__).resolve().parent / 
        f"results/{model_name}_{dataset_name}_seed-{seed}_aug-{config['preprocessing']['interaug']}"
        f"_GPU{config['gpu_id']}_{timestamp}"
    )   
    result_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "confmats", "curves"]: (result_dir / sub).mkdir(parents=True, exist_ok=True)

    # Save config to the result directory
    with open(result_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Retrieve model and datamodule classes
    model_cls = get_model_cls(model_name)
    datamodule_cls = get_datamodule_cls(dataset_name)

    config["model_kwargs"]["n_channels"] = datamodule_cls.channels
    config["model_kwargs"]["n_classes"] = datamodule_cls.classes

    # Parse subject IDs from config
    subj_cfg = config["subject_ids"]
    subject_ids = datamodule_cls.all_subject_ids if subj_cfg == "all" else \
                  [subj_cfg] if isinstance(subj_cfg, int) else \
                  subj_cfg
  
    # Initialize containers for tracking metrics across subjects
    test_accs, test_losses, test_kappas = [], [], []
    all_sessions_accs, all_sessions_losses, all_sessions_kappas = [], [], []
    train_times, test_times, all_sessions_test_times, response_times = [], [], [], []
    all_confmats, all_sessions_confmats = [], []
    no_tta_accs, no_tta_kappas = [], []

    # Optional wall-clock budget (hours, env MAX_RUN_HOURS, e.g. a Kaggle
    # session): no new fold is started when it would likely not finish; the
    # completed folds are summarized as usual.
    max_run_hours = float(os.environ.get("MAX_RUN_HOURS", "0") or 0)
    run_started = time.time()
    fold_durations, completed_ids = [], []

    # Loop through each subject ID for training and testing   
    for subject_id in subject_ids:
        if max_run_hours > 0 and fold_durations:
            elapsed_hours = (time.time() - run_started) / 3600
            next_fold_hours = 1.15 * float(np.mean(fold_durations)) / 3600
            if elapsed_hours + next_fold_hours > max_run_hours:
                remaining = [s for s in subject_ids if s not in completed_ids]
                print(f"\n>>> Time budget: {elapsed_hours:.2f} h used of {max_run_hours:.2f} h, "
                      f"next fold needs ~{next_fold_hours:.2f} h. Stopping before subject "
                      f"{subject_id}. Completed targets: {completed_ids}. "
                      f"Remaining targets: {remaining}", flush=True)
                break
        fold_started = time.time()
        print(f"\n>>> Training on subject: {subject_id}")

        # Set seed for reproducibility
        seed_everything(config["seed"])
        metrics_callback = MetricsCallback(
            log_every_n_epochs=config.get("log_every_n_epochs", 0)
        )
        best_checkpoint = ModelCheckpoint(
            dirpath=result_dir / "checkpoints",
            filename=(
                f"subject_{subject_id}_best"
                "-{epoch:03d}-{val_acc:.4f}"
            ),
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            save_last=False,
            auto_insert_metric_name=False,
        )
   
        # Initialize PyTorch Lightning Trainer
        trainer = Trainer(
            max_epochs=config["max_epochs"],
            accumulate_grad_batches=config["preprocessing"].get(
                "accumulate_grad_batches", 1
            ),
            devices = -1 if config.get("gpu_id", 0) == -1 else \
                [config.get("gpu_id", 0)],
            num_sanity_val_steps=0,
            accelerator="auto",
            strategy = "auto" if config.get("gpu_id", 0) != -1 
                else DDPStrategy(find_unused_parameters=True), 
            logger=False,
            enable_progress_bar=config.get("enable_progress_bar", True),
            enable_checkpointing=True,
            callbacks=[metrics_callback, best_checkpoint]
        )

        # Instantiate datamodule and model
        datamodule = datamodule_cls(config["preprocessing"], subject_id=subject_id)
        model = model_cls(**config["model_kwargs"], max_epochs=config["max_epochs"])

        # Optionally visualize the model graph
        # visualize_model_graph(model, model_name=model_name)
        
        # Count total number of model parameters
        param_count = sum(p.numel() for p in model.parameters())

        # ---------------- TRAIN ----------------
        st_train = time.time()
        trainer.fit(model, datamodule=datamodule)
        train_times.append((time.time() - st_train) / 60) # minutes

        # Select the epoch using labeled source validation only. The held-out
        # target labels are never consulted for checkpoint selection.
        if not best_checkpoint.best_model_path:
            raise RuntimeError(
                f"No best validation checkpoint was produced for subject {subject_id}."
            )
        checkpoint = torch.load(
            best_checkpoint.best_model_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(checkpoint["state_dict"])
        best_val_acc = float(best_checkpoint.best_model_score)
        print(
            f"Loaded best source-validation checkpoint | "
            f"subject={subject_id} | val_acc={best_val_acc * 100:.2f}% | "
            f"path={best_checkpoint.best_model_path}",
            flush=True,
        )

        # ---------------- TEST -----------------
        # Passing the datamodule to trainer.test() makes Lightning invoke
        # prepare_data/setup again. For LOSO this reloads every subject and can
        # exhaust Colab RAM. Reuse the already prepared test loader instead.
        test_loader = datamodule.test_dataloader()
        all_sessions_loader = (
            datamodule.all_target_dataloader()
            if datamodule.all_target_dataset is not None
            else None
        )
        datamodule.dataset = None
        datamodule.train_dataset = None
        datamodule.val_dataset = None
        datamodule.target_dataset = None
        datamodule.all_target_dataset = None
        gc.collect()

        # Same trained model scored without IM-TTA (paper's variant), before
        # the full adaptation; labels are read only by the scoring itself.
        if (config.get("score_without_tta", True) and hasattr(model, "adapt_to_target")
                and getattr(model, "im_tta_steps", 0) > 0):
            no_tta = score_without_tta(model, trainer, test_loader)
            no_tta_accs.append(float(no_tta["test_acc"]))
            no_tta_kappas.append(float(no_tta["test_kappa"]))
            print(f"\nTARGET SUBJECT {subject_id} WITHOUT TEST-TIME ADAPTATION | "
                  f"acc={no_tta_accs[-1] * 100:.2f}% | kappa={no_tta_kappas[-1]:.4f}\n",
                  flush=True)

        # Optional source-free IM-TTA. The method consumes only target EEG;
        # labels carried by the test loader are deliberately ignored.
        if hasattr(model, "adapt_to_target"):
            model.adapt_to_target(test_loader)

        st_test = time.time()
        test_results = trainer.test(model, dataloaders=test_loader)
        test_duration = time.time() - st_test
        test_times.append(test_duration)

        subject_result = test_results[0]
        subject_acc = float(subject_result["test_acc"])
        subject_loss = float(subject_result["test_loss"])
        subject_kappa = float(subject_result["test_kappa"])
        primary_test_label = getattr(
            datamodule_cls, "primary_test_label", "SESSION 2"
        )
        auxiliary_test_label = getattr(
            datamodule_cls, "auxiliary_test_label", "SESSIONS 1+2"
        )
        print(
            f"\nTARGET SUBJECT {subject_id} {primary_test_label} RESULT | "
            f"acc={subject_acc * 100:.2f}% | "
            f"loss={subject_loss:.4f} | "
            f"kappa={subject_kappa:.4f} | "
            f"test_time={test_duration:.2f}s\n",
            flush=True,
        )
        print(
            f">>> Target subject {subject_id} {primary_test_label.lower()} | "
            f"acc={test_results[0]['test_acc'] * 100:.2f}% | "
            f"loss={test_results[0]['test_loss']:.4f} | "
            f"kappa={test_results[0]['test_kappa']:.4f}"
        )

        # Some datasets expose an additional combined-session evaluation view.
        # PhysioNet has one imagery test view, so this branch is skipped.
        primary_confmat = model.test_confmat.numpy().copy()
        if all_sessions_loader is not None:
            st_all_sessions_test = time.time()
            all_sessions_results = trainer.test(
                model, dataloaders=all_sessions_loader
            )
            all_sessions_duration = time.time() - st_all_sessions_test
            all_sessions_test_times.append(all_sessions_duration)

            all_sessions_result = all_sessions_results[0]
            all_sessions_acc = float(all_sessions_result["test_acc"])
            all_sessions_loss = float(all_sessions_result["test_loss"])
            all_sessions_kappa = float(all_sessions_result["test_kappa"])
            all_sessions_accs.append(all_sessions_acc)
            all_sessions_losses.append(all_sessions_loss)
            all_sessions_kappas.append(all_sessions_kappa)
            all_sessions_confmats.append(model.test_confmat.numpy().copy())

            print(
                f"\nTARGET SUBJECT {subject_id} {auxiliary_test_label} RESULT "
                f"(AUXILIARY) | acc={all_sessions_acc * 100:.2f}% | "
                f"loss={all_sessions_loss:.4f} | "
                f"kappa={all_sessions_kappa:.4f} | "
                f"test_time={all_sessions_duration:.2f}s\n",
                flush=True,
            )

        # ---------------- LATENCY --------------
        # Deduce input shape from one sample of the test dataset
        sample_x, _ = datamodule.test_dataset[0]
        input_shape = (1, *sample_x.shape)  # prepend batch dim
        # device_str = "cuda:" + str(config["gpu_id"]) if config["gpu_id"] != -1 else "cpu"
        device_str = "cpu"
        lat_ms = measure_latency(model, input_shape, device=device_str)
        response_times.append(lat_ms)  # convert to seconds for summary helper

        # ---------------- METRICS --------------
        test_accs.append(subject_acc)
        test_losses.append(subject_loss)
        test_kappas.append(subject_kappa)

        # compute & store this subject's confusion matrix
        # The [C × C] tensor is inside the LightningModule:
        cm = primary_confmat
        all_confmats.append(cm)

        # plot per-subject if requested
        if config.get("plot_cm_per_subject", False):
            plot_confusion_matrix(
                cm, save_path=result_dir / f"confmats/confmat_subject_{subject_id}.png",
                class_names=datamodule_cls.class_names,
                title=f"Confusion Matrix – Subject {subject_id}",
            )
            if all_sessions_loader is not None:
                plot_confusion_matrix(
                    all_sessions_confmats[-1],
                    save_path=(
                        result_dir
                        / f"confmats/confmat_subject_{subject_id}_sessions_1_2.png"
                    ),
                    class_names=datamodule_cls.class_names,
                    title=(
                        f"Confusion Matrix - Subject {subject_id} "
                        f"{auxiliary_test_label.title()} (Auxiliary)"
                    ),
                )

        # Plot and save loss and accuracy curves if available
        if metrics_callback.train_loss and metrics_callback.val_loss:
            plot_curve(metrics_callback.train_loss, metrics_callback.val_loss,
                        "Loss", subject_id, result_dir / f"curves/subject_{subject_id}_loss.png")
        if metrics_callback.train_acc and metrics_callback.val_acc:
            plot_curve(metrics_callback.train_acc, metrics_callback.val_acc,
                        "Accuracy", subject_id, result_dir / f"curves/subject_{subject_id}_acc.png")
        if metrics_callback.train_loss and metrics_callback.val_all_sessions_loss:
            plot_curve(
                metrics_callback.train_loss,
                metrics_callback.val_all_sessions_loss,
                "Loss (target sessions 1+2)",
                subject_id,
                result_dir / f"curves/subject_{subject_id}_all_sessions_loss.png",
            )
        if metrics_callback.train_acc and metrics_callback.val_all_sessions_acc:
            plot_curve(
                metrics_callback.train_acc,
                metrics_callback.val_all_sessions_acc,
                "Accuracy (target sessions 1+2)",
                subject_id,
                result_dir / f"curves/subject_{subject_id}_all_sessions_acc.png",
            )

        # Optionally save the trained model's weights
        if config.get("save_checkpoint", False):
            ckpt_path = result_dir / f"checkpoints/subject_{subject_id}_model.ckpt"
            trainer.save_checkpoint(ckpt_path)

        completed_ids.append(subject_id)
        fold_durations.append(time.time() - fold_started)
        # Persist cumulative metrics immediately. OpenBMI jobs can approach the
        # Kaggle time limit, so a later interrupted fold must not erase the
        # results of folds that already completed.
        _write_no_tta_summary(
            result_dir, completed_ids, no_tta_accs, no_tta_kappas
        )
        write_summary(
            result_dir, model_name, dataset_name, completed_ids, param_count,
            test_accs, test_losses, test_kappas, train_times, test_times,
            response_times, all_sessions_accs=all_sessions_accs,
            all_sessions_losses=all_sessions_losses,
            all_sessions_kappas=all_sessions_kappas,
            all_sessions_test_times=all_sessions_test_times,
        )
        _copy_incremental_results(result_dir)
        print(f">>> Fold of target {subject_id} took {fold_durations[-1] / 60:.1f} min "
              f"| run elapsed {(time.time() - run_started) / 3600:.2f} h", flush=True)
   
    # Summarize and save final results (completed folds only)
    _write_no_tta_summary(
        result_dir, completed_ids, no_tta_accs, no_tta_kappas
    )
    subject_ids = completed_ids
    write_summary(result_dir, model_name, dataset_name, subject_ids, param_count,
        test_accs, test_losses, test_kappas, train_times, test_times, response_times,
        all_sessions_accs=all_sessions_accs,
        all_sessions_losses=all_sessions_losses,
        all_sessions_kappas=all_sessions_kappas,
        all_sessions_test_times=all_sessions_test_times)
    
    # plot the average if requested
    if config.get("plot_cm_average", True) and all_confmats:
        avg_cm = np.mean(np.stack(all_confmats), axis=0)
        plot_confusion_matrix(
            avg_cm, save_path=result_dir / "confmats/avg_confusion_matrix.png",
            class_names= datamodule_cls.class_names,
            title="Average Confusion Matrix",
        )     
    if config.get("plot_cm_average", True) and all_sessions_confmats:
        avg_all_sessions_cm = np.mean(np.stack(all_sessions_confmats), axis=0)
        plot_confusion_matrix(
            avg_all_sessions_cm,
            save_path=result_dir / "confmats/avg_confusion_matrix_sessions_1_2.png",
            class_names=datamodule_cls.class_names,
            title="Average Confusion Matrix - Target Sessions 1+2 (Auxiliary)",
        )


# Command-line argument parsing
def parse_arguments():
    """Parses command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="tcformer",
        help = "Name of the model to use. Options:\n"
               "tcformer, atcnet, d-atcnet, atcnet_2_0, eegnet, shallownet, basenet\n"
                "eegtcnet, eegconformer, tsseffnet, eegdeformer, sst_dpn, ctnet, mscformer"
    )        
    parser.add_argument("--dataset", type=str, default="bcic2a", 
        help="Name of the dataset to use."
                        "Options: bcic2a, bcic2b, hgd, physionet, zhou2016, openbmi, reh_mi, bcic3"
    )
    parser.add_argument("--loso", action="store_true", default=False, 
        help="Enable subject-independent (LOSO) mode"
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID to use")
    
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed value (overrides config if specified)")
    parser.add_argument("--interaug", action="store_true", 
                        help="Enable inter-trial augmentation (overrides config if specified)")
    parser.add_argument("--no_interaug", action="store_true", 
                        help="Disable inter-trial augmentation (overrides config if specified)")
    return parser.parse_args()

def score_without_tta(model, trainer, test_loader):
    """Score the "without test-time adaptation" variant of the same model.

    As in the paper's variant, only the AutoDIAL target statistics are
    re-estimated on the unlabeled evaluation trials (no IM-TTA). The target
    statistics are restored afterwards, so the subsequent full adaptation is
    exactly the one of a run without this extra score.
    """
    norm = getattr(model, "domain_norm", None)
    saved = None if norm is None else (norm.target_mean.clone(), norm.target_var.clone())
    im_tta_steps = model.im_tta_steps
    model.im_tta_steps = 0
    try:
        model.adapt_to_target(test_loader)
        return trainer.test(model, dataloaders=test_loader)[0]
    finally:
        model.im_tta_steps = im_tta_steps
        if saved is not None:
            norm.target_mean.copy_(saved[0])
            norm.target_var.copy_(saved[1])


# ----------------------------------------------
# Main function to run the training and testing pipeline
# ----------------------------------------------
def run():
    # show_gpu_info()     # Uncomment to display GPU info
    args = parse_arguments()
     
    # load config
    config_path = os.path.join(CONFIG_DIR, f"{args.model}.yaml") 
    with open(config_path) as f:    
        config = yaml.safe_load(f)

    # Adjust training parameters based on LOSO setting
    if args.loso:
        config["dataset_name"] = args.dataset + "_loso" 
        if args.dataset == "hgd":
            config["max_epochs"] = config["max_epochs_loso_hgd"]
        elif args.dataset == "zhou2016":
            config["max_epochs"] = config.get(
                "max_epochs_loso_zhou2016", config["max_epochs_loso"]
            )
        elif args.dataset == "openbmi":
            config["max_epochs"] = config.get(
                "max_epochs_loso_openbmi", config["max_epochs_loso"]
            )
        else:
            config["max_epochs"] = config["max_epochs_loso"]
        config["model_kwargs"]["warmup_epochs"] = config["model_kwargs"]["warmup_epochs_loso"]
    else:
        if config.get("requires_loso", False):
            raise ValueError(
                f"{config['model']} requires --loso to provide an unlabeled target domain."
            )
        config["dataset_name"] = args.dataset
        config["max_epochs"] = config["max_epochs_2b"] if args.dataset == "bcic2b" else config["max_epochs"]

    config["preprocessing"] = config["preprocessing"][args.dataset]
    dataset_model_overrides = config["preprocessing"].pop("model_overrides", {})
    if dataset_model_overrides:
        config["model_kwargs"].update(dataset_model_overrides)
        print(
            f"Dataset-specific model overrides ({args.dataset}): "
            f"{dataset_model_overrides}",
            flush=True,
        )
    config["preprocessing"]["z_scale"] = config["preprocessing"].get(
        "z_scale", config["z_scale"]
    )
    config["preprocessing"]["domain_adaptation"] = config.get("domain_adaptation", False)
    # Override interaug if specified
    if args.interaug:
        config["preprocessing"]["interaug"] = True
    elif args.no_interaug:
        config["preprocessing"]["interaug"] = False
    else:
        # A dataset block may disable augmentation (Zhou2016 runs without it).
        config["preprocessing"]["interaug"] = config["preprocessing"].get(
            "interaug", config["interaug"]
        )
    config.pop("interaug", None)

    config["gpu_id"] = args.gpu_id
    # Override seed if specified
    if args.seed is not None:
        config["seed"] = args.seed
    config["preprocessing"]["seed"] = config["seed"]

    # set to True to plot confusion matrices
    config["plot_cm_per_subject"] = True # set to True to plot per-subject confusion matrices
    config["plot_cm_average"]     = True # set to True to plot average confusion matrix

    train_and_test(config)

if __name__ == "__main__":
    run()
