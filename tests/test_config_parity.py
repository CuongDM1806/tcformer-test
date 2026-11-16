"""Every run reported in the paper resolves to the configuration it was run with.

tests/fixtures/paper_configs.json holds the effective configuration of each
reported run, reconstructed from the development branch and Kaggle notebook
that produced it (or the config.yaml saved with its results). This test
resolves the same run from configs/ and compares every setting that changes
the computation. Needs only PyYAML:  python -m pytest tests/test_config_parity.py
"""
import json
from pathlib import Path

import pytest

from ssm_dan import config as cfg

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "paper_configs.json").read_text())

# paper label -> (method, variants)
RUNS = {
    "full": ("ssm_dan", []),
    "no_ra": ("ssm_dan", ["no_ra"]),
    "no_da": ("ssm_dan", ["no_da"]),
    "no_tta": ("ssm_dan", ["no_tta"]),
    "per_session_ra": ("ssm_dan", ["per_session_ra"]),
    "source_only": ("ssm_dan", ["source_only"]),
    "eegconformer": ("eegconformer", []),
    "atcnet": ("atcnet", []),
    "tcformer": ("tcformer", []),
    "eegencoder": ("eegencoder", []),
    "satransnet": ("satransnet", []),
}
OPENBMI_RUNS = {"source_only": ("ssm_dan", ["source_only_head"])}

MODEL_NAMES = {"HADATCFormer": "SSMDAN", "FullMambaSourceOnly": "SSMSourceOnly",
               "TCFormerOriginal": "TCFormer", "EEGEncoderBaseline": "EEGEncoder",
               "SATransNetBaseline": "SATransNet"}
# Defaults of the development code when a key was absent from a config.
LEGACY_ADAPTATION_DEFAULTS = {
    "feature_alignment": "none", "autodial_alpha_init": 0.5, "domain_norm_momentum": 0.1,
    "target_im_weight": 0.0, "recalibrate_target_statistics": True, "aligner_hidden_dim": 128,
    "domain_hidden_dim": 128, "adaptation_dropout": 0.3, "adversarial_weight": 1.0, "mmd_weight": 0.5,
    "temporal_mmd_weight": 0.1, "im_tta_steps": 0, "im_tta_lr": 1e-4, "im_tta_diversity_weight": 1.0,
}
# Preprocessing keys read by the data modules, with the development-code defaults.
PREPROCESSING_DEFAULTS = {
    "sfreq": None, "low_cut": None, "high_cut": None, "start": 0.0, "stop": None, "trial_duration": None,
    "z_scale": None, "interaug": False, "batch_size": None, "riemannian_alignment": False,
    "ra_reference": "calibration", "common_average_reference": None, "domain_adaptation": False,
    # cpu_count() // 2 on the 4-vCPU Kaggle machines that ran every experiment.
    "num_workers": 2, "test_num_workers": 0, "accumulate_grad_batches": 1,
}
# Settings that do not change the computation of the variant they appear in.
UNUSED_WITHOUT_TARGET_BATCHES = ("target_im_weight", "adversarial_weight", "mmd_weight", "temporal_mmd_weight",
                                 "autodial_alpha_init", "domain_norm_momentum")
IGNORED_MODEL_KWARGS = {"warmup_epochs_loso", "light_adaptation_factor", "log_every_n_batches", "max_epochs"}


def _ssm_backbone(kwargs):
    """Map the development names of the SSM backbone to the current ones."""
    kwargs = dict(kwargs)
    types = kwargs.pop("sequence_block_types", None)
    if types is not None:
        assert len(types) == kwargs.pop("trans_depth")
        kwargs["ssm_directions"] = [t.replace("mamba_", "") for t in types]
        kwargs["ssm_dropout"] = kwargs.pop("trans_dropout")
        kwargs["ssm_d_state"] = kwargs.pop("mamba_d_state", 8)
        kwargs["ssm_d_conv"] = kwargs.pop("mamba_d_conv", 3)
        kwargs["ssm_scan"] = kwargs.pop("mamba_scan_mode")
        kwargs.pop("q_heads", None)   # attention heads: unused by an all-SSM stack
        kwargs.pop("kv_heads", None)
    return kwargs


def _normalize(config, legacy):
    model = MODEL_NAMES.get(config["model"], config["model"])
    kwargs = {k: v for k, v in config["model_kwargs"].items() if k not in IGNORED_MODEL_KWARGS}
    if model in ("SSMDAN", "SSMSourceOnly"):
        kwargs = _ssm_backbone(kwargs)
    if model == "SSMDAN":
        defaults = LEGACY_ADAPTATION_DEFAULTS if legacy else {}
        kwargs = {**defaults, **kwargs}
        if not config["domain_adaptation"]:
            for key in UNUSED_WITHOUT_TARGET_BATCHES:
                kwargs.pop(key, None)
    pre = config["preprocessing"]
    preprocessing = {k: pre.get(k, d) for k, d in PREPROCESSING_DEFAULTS.items()}
    if not preprocessing["riemannian_alignment"]:
        preprocessing.pop("ra_reference")
    return {"model": model, "max_epochs": config["max_epochs"], "seed": config["seed"],
            "domain_adaptation": bool(config["domain_adaptation"]), "model_kwargs": kwargs,
            "preprocessing": preprocessing}


def _current(label):
    dataset, run = label.split("/")
    method, variants = (OPENBMI_RUNS if dataset == "openbmi" and run in OPENBMI_RUNS else RUNS)[run]
    config = cfg.resolve(method, dataset, variants, seed=0)
    return {"model": config["model"], "max_epochs": config["max_epochs"], "seed": config["seed"],
            "domain_adaptation": config["domain_adaptation"], "model_kwargs": cfg.model_kwargs(config),
            "preprocessing": config["preprocessing"]}


@pytest.mark.parametrize("label", sorted(FIXTURE["runs"]))
def test_paper_run_config(label):
    legacy = _normalize(FIXTURE["runs"][label]["config"], legacy=True)
    current = _normalize(_current(label), legacy=False)
    assert current == legacy, FIXTURE["runs"][label]["source"]


@pytest.mark.parametrize("dataset", ["bcic2a", "bcic2b", "zhou2016"])
def test_rmetnet_config(dataset):
    legacy = FIXTURE["own_loop"]["rmetnet"]["config"]
    current = cfg.resolve("rmetnet", dataset)
    hp = dict(legacy["rmetnet"])
    expected = {
        "epochs": hp.pop("epochs_by_dataset", {}).get(dataset, hp["epochs"]),
        "cov_reg": hp.pop("cov_reg_by_dataset", {}).get(dataset, 0.0),
        "batch_size": hp["batch_size"], "lr": hp["lr"], "betas": hp["betas"],
        "weight_decay": hp["weight_decay"],
    }
    assert current["method_settings"] == expected
    for key, value in legacy["preprocessing"][dataset].items():
        assert current["preprocessing"][key] == value, key


@pytest.mark.parametrize("method", ["ttime", "bft"])
@pytest.mark.parametrize("dataset", ["bcic2a", "bcic2b", "zhou2016"])
def test_tta_config(method, dataset):
    legacy = FIXTURE["own_loop"]["tta"]["config"]
    current = cfg.resolve(method, dataset)
    settings = current["method_settings"]
    assert settings["method"] == method
    assert settings[method] == legacy[method]
    for key in ("lr", "batch_size", "max_epoch"):
        assert settings[key] == legacy[key], key
    for key, value in legacy["preprocessing"][dataset].items():
        assert current["preprocessing"][key] == value, key
