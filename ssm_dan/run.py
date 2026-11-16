"""Command-line entry point: one method (+ variants) on one dataset, LOSO.

    python -m ssm_dan.run --method ssm_dan --dataset bcic2a
    python -m ssm_dan.run --method ssm_dan --variant no_ra --dataset zhou2016 --subjects 1-2
    python -m ssm_dan.run --method eegconformer --dataset bcic2b --gpu 1
    python -m ssm_dan.run --method ssm_dan --dataset bcic2a --subjects 1 --set max_epochs=2   # smoke test
    python -m ssm_dan.run --list

Finished folds are skipped (their ``subjects/SXX.json`` exists) unless --overwrite.
"""
import argparse
import sys
import time

from . import config as cfg
from . import results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ssm_dan.run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="ssm_dan", help="config in configs/methods/ (default: ssm_dan)")
    parser.add_argument("--dataset", default="bcic2a", help="bcic2a | bcic2b | zhou2016 | openbmi")
    parser.add_argument("--variant", default="", help="comma-separated variants of the method, e.g. no_ra")
    parser.add_argument("--subjects", default="all", help="target subjects: all | 3 | 1-9 | 1,4,7-9")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config entry, e.g. preprocessing.batch_size=32 (repeatable)")
    parser.add_argument("--results", default=None, help="results root (default: ./results or $SSM_DAN_RESULTS)")
    parser.add_argument("--overwrite", action="store_true", help="re-run folds that already have results")
    parser.add_argument("--time-budget-h", type=float, default=None,
                        help="do not start a fold that would end after this many hours")
    parser.add_argument("--list", action="store_true", help="list methods and variants, then exit")
    return parser.parse_args(argv)


def list_methods():
    for method in cfg.available_methods():
        spec = cfg.load_yaml(cfg.CONFIG_DIR / "methods" / f"{method}.yaml")
        variants = ", ".join(spec.get("variants", {})) or "-"
        datasets = ", ".join(spec.get("datasets", {}))
        print(f"{method:14s} datasets: {datasets:34s} variants: {variants}")


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        list_methods()
        return 0
    variants = [v.strip() for v in args.variant.split(",") if v.strip()]
    config = cfg.resolve(args.method, args.dataset, variants, args.overrides, seed=args.seed)

    from .data import SUBJECT_IDS
    subjects = cfg.parse_subjects(args.subjects, SUBJECT_IDS[args.dataset])
    out_dir = results.prepare(results.run_dir(config, args.results), config)
    todo = [s for s in subjects if args.overwrite or not results.is_done(out_dir, s)]
    print(f"Run {config['name']} | dataset {args.dataset} | seed {args.seed} | output {out_dir}\n"
          f"Targets {subjects} | to train {todo}", flush=True)
    if not todo:
        print(results.summarize(out_dir) or "", flush=True)
        return 0

    from . import pipeline

    started, fold_hours = time.monotonic(), []
    sessions = None
    if config["runner"] != "lightning":
        from .baselines.sessions import load_sessions
        sessions = load_sessions(args.dataset, config["preprocessing"])
    for subject in todo:
        if args.time_budget_h is not None and fold_hours:
            left = args.time_budget_h - (time.monotonic() - started) / 3600
            if max(fold_hours) > left:
                print(f"Time budget: {left:.1f} h left, a fold takes ~{max(fold_hours):.1f} h; "
                      f"stopping before subject {subject}.", flush=True)
                break
        fold_started = time.monotonic()
        print(f"\n>>> {config['name']} | {args.dataset} | target subject {subject}", flush=True)
        if config["runner"] == "lightning":
            record = pipeline.run_lightning_fold(config, subject, out_dir, gpu=args.gpu)
        else:
            record = pipeline.run_baseline_fold(config, subject, sessions, gpu=args.gpu)
            pipeline.plot_baseline_confusion(record, out_dir)
        record.update(method=config["method"], variants=config["variants"], dataset=args.dataset,
                      seed=args.seed)
        results.write_subject(out_dir, record)
        fold_hours.append((time.monotonic() - fold_started) / 3600)
        print(results.summarize(out_dir), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
