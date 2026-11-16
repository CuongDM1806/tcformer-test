"""Networks trained by the LOSO pipeline (imported lazily: baselines have extra dependencies)."""
import importlib

MODELS = {
    "SSMDAN": ("ssm_dan.models.ssm_dan", "SSMDAN"),
    "SSMSourceOnly": ("ssm_dan.models.backbone", "SSMSourceOnly"),
    "EEGConformer": ("ssm_dan.models.baselines.eegconformer", "EEGConformer"),
    "ATCNet": ("ssm_dan.models.baselines.atcnet", "ATCNet"),
    "TCFormer": ("ssm_dan.models.baselines.tcformer", "TCFormer"),
    "EEGEncoder": ("ssm_dan.models.baselines.eegencoder", "EEGEncoderBaseline"),
    "SATransNet": ("ssm_dan.models.baselines.satransnet", "SATransNetBaseline"),
}


def get_model_cls(name: str):
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(MODELS)}.")
    module, cls = MODELS[name]
    return getattr(importlib.import_module(module), cls)
