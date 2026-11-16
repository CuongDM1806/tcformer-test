"""LOSO data modules of the four MI-EEG datasets."""

# Target subjects of each dataset (available without importing the data stack).
SUBJECT_IDS = {
    "bcic2a": list(range(1, 10)),
    "bcic2b": list(range(1, 10)),
    "zhou2016": list(range(1, 5)),
    "openbmi": list(range(1, 55)),
}

DATASETS = {
    "bcic2a": ("ssm_dan.data.bcic2a", "BCICIV2aLOSO"),
    "bcic2b": ("ssm_dan.data.bcic2b", "BCICIV2bLOSO"),
    "zhou2016": ("ssm_dan.data.zhou2016", "Zhou2016LOSO"),
    "openbmi": ("ssm_dan.data.openbmi", "OpenBMILOSO"),
}


def get_datamodule_cls(name: str):
    """Data-module class of a dataset (imported lazily)."""
    import importlib

    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}.")
    module, cls = DATASETS[name]
    return getattr(importlib.import_module(module), cls)
