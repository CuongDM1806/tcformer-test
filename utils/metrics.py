
from pytorch_lightning import Callback
import numpy as np

# Custom Callback to track train/val loss and accuracy each epoch
class MetricsCallback(Callback):
    """Custom PyTorch Lightning callback to track training and validation metrics."""
    def __init__(self):
        super().__init__()
        self.train_loss, self.val_loss = [], []
        self.train_acc, self.val_acc = [], []
        self.val_all_sessions_loss, self.val_all_sessions_acc = [], []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.train_loss.append(metrics["train_loss"].cpu().item())
        if "train_acc" in metrics:
            self.train_acc.append(metrics["train_acc"].cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "val_loss" in metrics:
            self.val_loss.append(metrics["val_loss"].cpu().item())
        if "val_acc" in metrics:
            self.val_acc.append(metrics["val_acc"].cpu().item())
        if "val_all_sessions_loss" in metrics:
            self.val_all_sessions_loss.append(
                metrics["val_all_sessions_loss"].cpu().item()
            )
        if "val_all_sessions_acc" in metrics:
            self.val_all_sessions_acc.append(
                metrics["val_all_sessions_acc"].cpu().item()
            )


# Helper to write summary results to a text file
def write_summary(result_dir, model_name, dataset_name, subject_ids,
                   param_count, test_accs, test_losses, test_kappas,
                   train_times, test_times, response_times,
                   all_sessions_accs=None, all_sessions_losses=None,
                   all_sessions_kappas=None, all_sessions_test_times=None,
                   all_sessions_test_label="Sessions 1+2",
                   primary_test_label="Session 2"):
    avg_test_acc = float(np.mean(test_accs))
    std_test_acc = float(np.std(test_accs))
    avg_test_kappa = float(np.mean(test_kappas))   # 🆕  average κ
    std_test_kappa = float(np.std(test_kappas))
    avg_test_loss = float(np.mean(test_losses))
    std_test_loss = float(np.std(test_losses))

    total_train_time = float(np.sum(train_times))
    avg_response_time = float(np.mean(response_times))   # milliseconds
    has_all_sessions = bool(all_sessions_accs)
    if has_all_sessions:
        avg_all_sessions_acc = float(np.mean(all_sessions_accs))
        std_all_sessions_acc = float(np.std(all_sessions_accs))
        avg_all_sessions_loss = float(np.mean(all_sessions_losses))
        std_all_sessions_loss = float(np.std(all_sessions_losses))
        avg_all_sessions_kappa = float(np.mean(all_sessions_kappas))
        std_all_sessions_kappa = float(np.std(all_sessions_kappas))


    with open(result_dir / "results.txt", "w", encoding="utf-8") as f:
        f.write(f"Results for model: {model_name}\n")
        f.write(f"#Params: {param_count}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Subject IDs: {subject_ids}\n\n")
        f.write("Results for each subject:\n")

        for i, subject_id in enumerate(subject_ids):
            f.write(
                f"Subject {subject_id} => Train Time: {train_times[i]:.2f}m, "
                f"{primary_test_label} Test Time: {test_times[i]:.2f}s, "
                f"{primary_test_label} Acc: {test_accs[i]:.4f}, "
                f"{primary_test_label} Loss: {test_losses[i]:.4f}, "
                f"{primary_test_label} Kappa: {test_kappas[i]:.4f}"
            )
            if has_all_sessions:
                f.write(
                    f", {all_sessions_test_label} Time: "
                    f"{all_sessions_test_times[i]:.2f}s, "
                    f"{all_sessions_test_label} Acc: {all_sessions_accs[i]:.4f}, "
                    f"{all_sessions_test_label} Loss: {all_sessions_losses[i]:.4f}, "
                    f"{all_sessions_test_label} Kappa: "
                    f"{all_sessions_kappas[i]:.4f}"
                )
            f.write("\n")

        f.write("\n--- Summary Statistics ---\n")
        f.write(f"Average Test Accuracy: {avg_test_acc * 100:.2f} ± {std_test_acc * 100:.2f}\n")
        f.write(f"Average Test Kappa:    {avg_test_kappa:.3f} ± {std_test_kappa:.3f}\n")
        f.write(f"Average Test Loss:     {avg_test_loss:.3f} ± {std_test_loss:.3f}\n")
        f.write(f"Total Training Time: {total_train_time:.2f} min\n")
        f.write(f"Average Response Time: {avg_response_time:.2f} ms\n")
        if has_all_sessions:
            f.write(f"\n--- Auxiliary Target {all_sessions_test_label} ---\n")
            f.write(
                f"Average {all_sessions_test_label} Accuracy: "
                f"{avg_all_sessions_acc * 100:.2f} ± "
                f"{std_all_sessions_acc * 100:.2f}\n"
            )
            f.write(
                f"Average {all_sessions_test_label} Kappa:    "
                f"{avg_all_sessions_kappa:.3f} ± "
                f"{std_all_sessions_kappa:.3f}\n"
            )
            f.write(
                f"Average {all_sessions_test_label} Loss:     "
                f"{avg_all_sessions_loss:.3f} ± "
                f"{std_all_sessions_loss:.3f}\n"
            )

    print("\n=== Summary ===")
    print(f"Average Test Accuracy: {avg_test_acc * 100:.2f} ± {std_test_acc * 100:.2f}")
    print(f"Average Test Kappa:    {avg_test_kappa:.3f} ± {std_test_kappa:.3f}")
    print(f"Average Test Loss:     {avg_test_loss:.3f} ± {std_test_loss:.3f}")
    print(f"Total Training Time: {total_train_time:.2f} min")
    print(f"Average Response Time: {avg_response_time:.2f} ms")
    if has_all_sessions:
        print(f"\n=== Auxiliary Target {all_sessions_test_label} ===")
        print(
            f"Average {all_sessions_test_label} Accuracy: "
            f"{avg_all_sessions_acc * 100:.2f} ± "
            f"{std_all_sessions_acc * 100:.2f}"
        )
        print(
            f"Average {all_sessions_test_label} Kappa:    "
            f"{avg_all_sessions_kappa:.3f} ± "
            f"{std_all_sessions_kappa:.3f}"
        )
        print(
            f"Average {all_sessions_test_label} Loss:     "
            f"{avg_all_sessions_loss:.3f} ± "
            f"{std_all_sessions_loss:.3f}"
        )
