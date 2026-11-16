"""Run configuration: dataset protocol + method + variants + command-line overrides.

A run is resolved from

1. ``configs/datasets.yaml[dataset]``: protocol and preprocessing shared by every method;
2. ``configs/methods/<method>.yaml``: model, optimization and method defaults,
   then its ``datasets[dataset]`` block (per-dataset settings of the method);
3. the requested ``variants`` of the method, applied in order;
4. ``--set key.path=value`` overrides.

Every block may contain ``preprocessing``, ``model_kwargs``,
``adaptation_kwargs``, ``method_settings`` (merged key by key) and the scalars
``model``, ``max_epochs``, ``domain_adaptation``, ``evaluate_without_im_tta``.
"""
import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
MERGED = ("preprocessing", "model_kwargs", "adaptation_kwargs", "method_settings")
SCALARS = ("model", "runner", "max_epochs", "domain_adaptation", "evaluate_without_im_tta",
           "log_every_n_epochs", "enable_progress_bar")


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def available_methods():
    return sorted(p.stem for p in (CONFIG_DIR / "methods").glob("*.yaml"))


def _apply(config, block, where):
    unknown = set(block) - set(MERGED) - set(SCALARS) - {"description"}
    if unknown:
        raise KeyError(f"{where}: unknown keys {sorted(unknown)}")
    for key in MERGED:
        if block.get(key) is not None:
            config.setdefault(key, {}).update(copy.deepcopy(block[key]))
    for key in SCALARS:
        if key in block:
            config[key] = block[key]


def parse_value(text):
    """YAML scalar parsing for --set values (numbers, booleans, null, lists)."""
    return yaml.safe_load(text)


def set_path(config, dotted, value):
    keys = dotted.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def run_name(method, variants):
    return "-".join([method, *variants]) if variants else method


def resolve(method, dataset, variants=(), overrides=(), seed=0, subjects=None):
    """The complete configuration of one run (one method, dataset, seed)."""
    datasets = load_yaml(CONFIG_DIR / "datasets.yaml")
    if dataset not in datasets:
        raise KeyError(f"Unknown dataset {dataset!r}; choose from {sorted(datasets)}.")
    method_path = CONFIG_DIR / "methods" / f"{method}.yaml"
    if not method_path.is_file():
        raise KeyError(f"Unknown method {method!r}; choose from {available_methods()}.")
    spec = load_yaml(method_path)

    config = {"method": method, "dataset": dataset, "variants": list(variants), "seed": int(seed),
              "runner": "lightning", "domain_adaptation": False, "evaluate_without_im_tta": False}
    _apply(config, datasets[dataset], f"datasets.yaml[{dataset}]")
    _apply(config, {k: v for k, v in spec.items() if k not in ("datasets", "variants")}, method_path.name)
    supported = spec.get("datasets", {})
    if dataset not in supported:
        raise KeyError(f"{method} has no settings for {dataset}; supported: {sorted(supported)}.")
    _apply(config, supported[dataset] or {}, f"{method_path.name}: datasets.{dataset}")

    declared = spec.get("variants", {})
    for variant in variants:
        if variant not in declared:
            raise KeyError(f"{method} has no variant {variant!r}; choose from {sorted(declared)}.")
        block = dict(declared[variant] or {})
        only = block.pop("datasets_only", None)
        if only and dataset not in only:
            raise ValueError(f"Variant {variant!r} applies only to {only}.")
        _apply(config, block, f"{method_path.name}: variants.{variant}")

    for item in overrides:
        key, _, value = item.partition("=")
        if not _:
            raise ValueError(f"--set expects key=value, got {item!r}")
        set_path(config, key.strip(), parse_value(value))

    # The data modules read these from the preprocessing block.
    config["preprocessing"]["domain_adaptation"] = bool(config["domain_adaptation"])
    config["preprocessing"]["seed"] = config["seed"]
    config.setdefault("model_kwargs", {})["max_epochs"] = config.get("max_epochs")
    config["subjects"] = subjects
    config["name"] = run_name(method, variants)
    return config


def model_kwargs(config):
    """Constructor kwargs of the network: model_kwargs (+ adaptation_kwargs for SSM-DAN)."""
    kwargs = copy.deepcopy(config.get("model_kwargs", {}))
    if config.get("model") == "SSMDAN":
        kwargs.update(copy.deepcopy(config.get("adaptation_kwargs", {})))
    return kwargs


def parse_subjects(text, all_ids):
    """'all', '3', '1-9', '1,4,7-9' -> list of subject ids."""
    if text in (None, "", "all"):
        return list(all_ids)
    subjects = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            first, last = part.split("-")
            subjects.extend(range(int(first), int(last) + 1))
        elif part:
            subjects.append(int(part))
    unknown = sorted(set(subjects) - set(all_ids))
    if unknown:
        raise ValueError(f"Subjects {unknown} are not in this dataset ({all_ids[0]}..{all_ids[-1]}).")
    return subjects
