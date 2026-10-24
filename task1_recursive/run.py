"""Task 1 runner:  python -m task1_recursive.run [--seeds 0 1 2] [--set data.source=synthetic]

Compares three variants over seeds:
  verified         : confidence + independent-model agreement, dev gating with rollback  (the Task 1 system)
  confidence_only  : confidence threshold only, dev gating with rollback
  naive            : confidence threshold only, no rollback (shows error reinforcement risk)
Baselines: seed-only teacher, char-n-gram LR on seed only, and a full-label teacher (upper reference).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import make_split
from src.eval import macro_f1, scores
from src.models import CharLR, train_model
from src.pipeline import base_parser, setup
from src.utils import ROOT, ensure_dir, save_json
from .selftrain import run_selftrain

VARIANTS = {
    "verified": dict(require_agreement=True, use_rollback=True),
    "confidence_only": dict(require_agreement=False, use_rollback=True),
    "naive": dict(require_agreement=False, use_rollback=False),
}


def main(argv=None):
    args = base_parser("Task 1: verified recursive self-training").parse_args(argv)
    cfg, base = setup(args)
    seeds = args.seeds if args.seeds is not None else cfg["task1"]["seeds"]
    out = ensure_dir(ROOT / (args.out or "results/task1"))
    nb = cfg["eval"]["ece_bins"]
    Xt, Xv = base.X["teacher"], base.X["verifier"]

    rounds, finals = [], []
    for seed in seeds:
        split = make_split(base, seed, cfg["data"]["seed_per_class"])
        y_seed = base.y["train"][split.seed_idx]

        lr = CharLR(cfg["models"]["verifier_lr_C"], base.n_classes).fit(Xv["train"][split.seed_idx], y_seed)
        finals.append({"seed": seed, "variant": "baseline_charLR_seed_only", **{f"test_{k}": v for k, v in scores(lr.proba(Xv["test"]), base.y["test"], nb).items()}})
        full = train_model("teacher", Xt["train"], base.y["train"], cfg, seed)
        finals.append({"seed": seed, "variant": "reference_full_labels", **{f"test_{k}": v for k, v in scores(full.proba(Xt["test"]), base.y["test"], nb).items()}})

        for name, flags in VARIANTS.items():
            res = run_selftrain(base, split, cfg, seed, **flags)
            for h in res.history:
                rounds.append({"seed": seed, "variant": name, **h})
            for tag, model in (("round0_teacher", res.teacher_round0), ("final_teacher", res.teacher_final)):
                s = scores(model.proba(Xt["test"]), base.y["test"], nb)
                finals.append({"seed": seed, "variant": f"{name}:{tag}", **{f"test_{k}": v for k, v in s.items()},
                               "n_accepted": int(len(res.accepted_idx)),
                               "final_pl_precision": float((res.accepted_y == base.y["train"][res.accepted_idx]).mean()) if len(res.accepted_idx) else np.nan})

    rdf, fdf = pd.DataFrame(rounds), pd.DataFrame(finals)
    rdf.to_csv(out / "rounds.csv", index=False)
    fdf.to_csv(out / "final_test.csv", index=False)
    summary = fdf.groupby("variant")[["test_macro_f1", "test_accuracy", "test_ece"]].agg(["mean", "std"]).round(4)
    summary.to_csv(out / "summary.csv")
    save_json({"seeds": seeds, "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}, out / "run_config.json")
    print("\n=== Task 1: final TEST results (mean/std over seeds) ===")
    print(summary.to_string())
    print(f"\nper-round log: {out / 'rounds.csv'}")
    return rdf, fdf


if __name__ == "__main__":
    main()
