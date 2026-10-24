"""Task 2 runner:  python -m task2_distillation.run [--seeds 0 1 2]

Writes results/task2/{all_runs.csv, summary.csv, sweep_dev.csv, best.json}. best.json (T, alpha chosen on DEV)
is read by Task 3 when task3.temperature / task3.alpha are null.
"""
from __future__ import annotations

from src.pipeline import base_parser, setup
from src.utils import ROOT, ensure_dir, save_json
from .distill import run_distillation, select_best


def main(argv=None):
    args = base_parser("Task 2: knowledge distillation").parse_args(argv)
    cfg, base = setup(args)
    seeds = args.seeds if args.seeds is not None else cfg["task2"]["seeds"]
    out = ensure_dir(ROOT / (args.out or "results/task2"))

    df = run_distillation(base, cfg, seeds)
    df.to_csv(out / "all_runs.csv", index=False)

    best = select_best(df)
    save_json(best, out / "best.json")
    save_json({"seeds": seeds, "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}, out / "run_config.json")

    sweep = (df[df["model"] == "student_distilled"].groupby(["T", "alpha"])[["dev_macro_f1", "test_macro_f1"]]
             .agg(["mean", "std"]).round(4))
    sweep.to_csv(out / "sweep_dev.csv")

    chosen = df[(df["model"] == "student_distilled") & (df["T"] == best["T"]) & (df["alpha"] == best["alpha"])].copy()
    chosen["model"] = "student_distilled(best_on_dev)"
    final = df[df["model"].isin(["teacher", "student_hard", "baseline_charLR"])]
    import pandas as pd
    comp = pd.concat([final, chosen])
    summary = comp.groupby("model")[["test_macro_f1", "test_accuracy", "test_ece", "params", "size_kb", "latency_ms"]].agg(["mean", "std"]).round(4)
    summary.to_csv(out / "summary.csv")

    print(f"\nbest (T, alpha) on dev: {best}")
    print("\n=== Task 2: TEST results (mean/std over seeds) ===")
    print(summary.to_string())
    return df


if __name__ == "__main__":
    main()
