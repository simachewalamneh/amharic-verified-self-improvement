"""Diagnostic for Task 1: what class mix do accepted pseudo-labels have, and which class does the final teacher lose?

python -m task1_recursive.class_drift [--seeds 0 1 2]
Writes results/task1/class_drift.csv (accepted pseudo-label counts per class, pool gold counts, per-class test F1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import make_split
from src.pipeline import base_parser, setup
from src.utils import ROOT, ensure_dir
from .run import VARIANTS
from .selftrain import run_selftrain


def main(argv=None):
    args = base_parser("Task 1 class-drift diagnostic").parse_args(argv)
    cfg, base = setup(args)
    seeds = args.seeds if args.seeds is not None else cfg["task1"]["seeds"]
    out = ensure_dir(ROOT / "results" / "task1")
    Xt, y_te = base.X["teacher"], base.y["test"]
    rows = []
    for seed in seeds:
        split = make_split(base, seed, cfg["data"]["seed_per_class"])
        pool_gold = np.bincount(base.y["train"][split.pool_idx], minlength=base.n_classes)
        for name in ("verified", "naive"):
            res = run_selftrain(base, split, cfg, seed, log=lambda *a, **k: None, **VARIANTS[name])
            acc = np.bincount(res.accepted_y, minlength=base.n_classes) if len(res.accepted_y) else np.zeros(base.n_classes, int)
            for tag, model in (("round0", res.teacher_round0), ("final", res.teacher_final)):
                f1s = f1_score(y_te, model.proba(Xt["test"]).argmax(1), average=None, labels=range(base.n_classes))
                row = {"seed": seed, "variant": name, "teacher": tag}
                row.update({f"f1_{base.label_names[c]}": float(f1s[c]) for c in range(base.n_classes)})
                if tag == "final":
                    row.update({f"accepted_{base.label_names[c]}": int(acc[c]) for c in range(base.n_classes)})
                    row.update({f"pool_gold_{base.label_names[c]}": int(pool_gold[c]) for c in range(base.n_classes)})
                rows.append(row)
            print(f"seed {seed} {name}: accepted per class {dict(zip(base.label_names, acc.tolist()))}", flush=True)
    pd.DataFrame(rows).to_csv(out / "class_drift.csv", index=False)


if __name__ == "__main__":
    main()
