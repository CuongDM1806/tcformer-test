"""Fingerprint one fold of one paper run, on this repository or on a development branch.

Run inside a checkout (``--side new`` for this repository, ``--side legacy`` for
the development branch that produced the paper result):

    python probe.py --side new    --label bcic2a/no_ra --fixture paper_configs.json --out new.json
    python probe.py --side legacy --label bcic2a/no_ra --fixture paper_configs.json --out old.json

For target subject 1 it records SHA-256 fingerprints of the four data splits,
of the initial weights, of the weights after two short CPU training epochs and of
the test logits after Stage-2 adaptation. Identical fingerprints mean that the
two code bases perform the same computation with the same random numbers.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

NEW_RUNS = {
    "full": ("ssm_dan", []), "no_ra": ("ssm_dan", ["no_ra"]), "no_da": ("ssm_dan", ["no_da"]),
    "no_tta": ("ssm_dan", ["no_tta"]), "per_session_ra": ("ssm_dan", ["per_session_ra"]),
    "source_only": ("ssm_dan", ["source_only"]), "eegconformer": ("eegconformer", []),
    "atcnet": ("atcnet", []), "tcformer": ("tcformer", []), "eegencoder": ("eegencoder", []),
    "satransnet": ("satransnet", []),
}


def digest(arrays):
    h = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array.detach().cpu().numpy() if torch.is_tensor(array) else array)
        h.update(str(array.dtype).encode() + str(array.shape).encode())
        h.update(array.tobytes())
    return h.hexdigest()[:16]


def state_digest(module):
    state = {k.replace("model.transformer.", "model.ssm_blocks."): v
             for k, v in module.state_dict().items()}
    return digest([state[k] for k in sorted(state)])


def dataset_tensors(dataset):
    return list(dataset.tensors) if dataset is not None else []


def build(side, label, fixture):
    dataset, run = label.split("/")
    if side == "legacy":
        from utils.get_datamodule_cls import get_datamodule_cls
        from utils.get_model_cls import get_model_cls
        from utils.seed import seed_everything
        config = json.loads(json.dumps(fixture["runs"][label]["config"]))
        datamodule_cls = get_datamodule_cls(f"{dataset}_loso")
        model_cls = get_model_cls(config["model"])
        kwargs = dict(config["model_kwargs"], max_epochs=config["max_epochs"])
        preprocessing = config["preprocessing"]
        max_epochs = config["max_epochs"]
    else:
        from ssm_dan import config as cfg
        from ssm_dan.data import get_datamodule_cls
        from ssm_dan.models import get_model_cls
        from ssm_dan.utils.seed import seed_everything
        method, variants = ("ssm_dan", ["source_only_head"]) if label == "openbmi/source_only" else NEW_RUNS[run]
        config = cfg.resolve(method, dataset, variants)
        datamodule_cls = get_datamodule_cls(dataset)
        model_cls = get_model_cls(config["model"])
        kwargs = cfg.model_kwargs(config)
        preprocessing = config["preprocessing"]
        max_epochs = config["max_epochs"]
    kwargs.update(n_channels=datamodule_cls.channels, n_classes=datamodule_cls.classes)
    return datamodule_cls, model_cls, kwargs, preprocessing, max_epochs, seed_everything


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=["new", "legacy"], required=True)
    parser.add_argument("--label", required=True, help="e.g. bcic2a/no_ra (a key of the fixture)")
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--steps", type=int, default=6, help="training steps per epoch (two epochs)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    fixture = json.loads(Path(args.fixture).read_text())

    from pytorch_lightning import Trainer

    datamodule_cls, model_cls, kwargs, preprocessing, _, seed_everything = build(
        args.side, args.label, fixture)
    torch.set_num_threads(4)
    seed_everything(0)
    # Two short epochs: the learning rate is zero during the first warm-up epoch.
    trainer = Trainer(accelerator="cpu", devices=1, max_epochs=2, limit_train_batches=args.steps,
                      limit_val_batches=0, num_sanity_val_steps=0, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False)
    datamodule = datamodule_cls(preprocessing, subject_id=args.subject)
    model = model_cls(**kwargs)
    report = {"label": args.label, "side": args.side,
              "params": int(sum(p.numel() for p in model.parameters())),
              "init": state_digest(model)}
    trainer.fit(model, datamodule=datamodule)
    report["data"] = {name: digest(dataset_tensors(getattr(datamodule, f"{name}_dataset", None)))
                      for name in ("train", "val", "target", "test")}
    report["trained"] = state_digest(model)

    test_loader = datamodule.test_dataloader()
    if hasattr(model, "adapt_to_target"):
        model.adapt_to_target(test_loader)
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(batch[0]) for batch in test_loader])
        labels = torch.cat([batch[1] for batch in test_loader])
    report["test_logits"] = digest([logits])
    report["test_acc_after_steps"] = float((logits.argmax(1) == labels).float().mean())
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
