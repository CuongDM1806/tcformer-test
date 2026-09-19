"""Fold-wise self-supervised pretraining for the TCFormer encoder."""

import argparse
import gc
import inspect
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from pytorch_lightning import Trainer
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from models.masked_reconstruction_pretrain import MaskedReconstructionPretrain
from models.tcformer import TCFormerModule
from utils.get_datamodule_cls import get_datamodule_cls
from utils.seed import seed_everything


CONFIG_PATH = Path(__file__).resolve().parent / "configs/hada_tcformer.yaml"


class EEGOnlyDataset(Dataset):
    """Discard labels and expose only EEG tensors to the pretraining stage."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if isinstance(item, (tuple, list)):
            return item[0]
        return item


def _encoder_kwargs(model_kwargs: dict, n_channels: int, n_classes: int) -> dict:
    parameters = inspect.signature(TCFormerModule.__init__).parameters
    allowed = set(parameters) - {"self", "n_channels", "n_classes"}
    filtered = {key: value for key, value in model_kwargs.items() if key in allowed}
    filtered.update(n_channels=n_channels, n_classes=n_classes)
    return filtered


def _subject_ids(config: dict, datamodule_cls, requested):
    if requested:
        unknown = sorted(set(requested) - set(datamodule_cls.all_subject_ids))
        if unknown:
            raise ValueError(f"Unsupported subject IDs: {unknown}")
        return requested
    configured = config["subject_ids"]
    if configured == "all":
        return datamodule_cls.all_subject_ids
    return [configured] if isinstance(configured, int) else configured


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bcic2a", choices=["bcic2a", "physionet"])
    parser.add_argument("--subject_ids", type=int, nargs="+")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--mask_span_ms", type=float, default=100.0)
    parser.add_argument("--frequency_loss_weight", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--output_dir", type=Path, default=Path("pretrained_encoders"))
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip folds whose encoder checkpoint already exists.",
    )
    parser.add_argument(
        "--source_only",
        action="store_true",
        help="Exclude unlabeled target EEG from pretraining.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    datamodule_cls = get_datamodule_cls(f"{args.dataset}_loso")
    subjects = _subject_ids(config, datamodule_cls, args.subject_ids)
    preprocessing = deepcopy(config["preprocessing"][args.dataset])
    preprocessing.pop("model_overrides", None)
    preprocessing["z_scale"] = preprocessing.get("z_scale", config["z_scale"])
    preprocessing["interaug"] = False
    preprocessing["domain_adaptation"] = False
    preprocessing["seed"] = config["seed"]
    if args.num_workers is not None:
        preprocessing["num_workers"] = args.num_workers

    batch_size = args.batch_size or preprocessing["batch_size"]
    num_workers = preprocessing.get("num_workers", 2)
    mask_span_samples = max(
        1, int(round(args.mask_span_ms * preprocessing["sfreq"] / 1000.0))
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in subjects:
        checkpoint_path = output_dir / (
            f"{args.dataset}_target-{subject_id:03d}_encoder.pt"
        )
        if args.skip_existing and checkpoint_path.is_file():
            print(f"Skipping existing checkpoint: {checkpoint_path}", flush=True)
            continue
        seed_everything(config["seed"])
        print(
            f"\n>>> SSL pretraining fold target={subject_id} | "
            f"dataset={args.dataset} | source_only={args.source_only}",
            flush=True,
        )
        datamodule = datamodule_cls(preprocessing, subject_id=subject_id)
        datamodule.prepare_data()
        datamodule.setup("fit")
        if datamodule.train_dataset is None:
            raise RuntimeError("LOSO datamodule did not create a source train dataset.")

        datasets = [EEGOnlyDataset(datamodule.train_dataset)]
        target_samples = 0
        if not args.source_only:
            if datamodule.target_dataset is None:
                raise RuntimeError("LOSO datamodule did not expose unlabeled target EEG.")
            datasets.append(EEGOnlyDataset(datamodule.target_dataset))
            target_samples = len(datamodule.target_dataset)
        pretrain_dataset = ConcatDataset(datasets)
        train_loader = DataLoader(
            pretrain_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=True,
            **({"prefetch_factor": 2} if num_workers > 0 else {}),
        )

        encoder = TCFormerModule(
            **_encoder_kwargs(
                config["model_kwargs"],
                n_channels=datamodule_cls.channels,
                n_classes=datamodule_cls.classes,
            )
        )
        pretrainer = MaskedReconstructionPretrain(
            encoder=encoder,
            n_channels=datamodule_cls.channels,
            mask_ratio=args.mask_ratio,
            mask_span_samples=mask_span_samples,
            frequency_loss_weight=args.frequency_loss_weight,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        trainer = Trainer(
            max_epochs=args.epochs,
            accelerator="auto",
            devices=-1 if args.gpu_id == -1 else [args.gpu_id],
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(pretrainer, train_dataloaders=train_loader)

        pretrainer.save_encoder(
            checkpoint_path,
            dataset=args.dataset,
            target_subject=subject_id,
            source_only=args.source_only,
            source_samples=len(datamodule.train_dataset),
            target_unlabeled_samples=target_samples,
            mask_ratio=args.mask_ratio,
            mask_span_samples=mask_span_samples,
            frequency_loss_weight=args.frequency_loss_weight,
        )
        print(f"Saved pretrained encoder: {checkpoint_path}", flush=True)

        del trainer, pretrainer, encoder, train_loader, pretrain_dataset, datamodule
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
