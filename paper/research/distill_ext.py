"""Distillation extension: linear AND non-linear students, two supervision conditions, 10 seeds, paired statistics.

python -m paper.research.distill_ext [--seeds 0 ... 9]
Conditions:
  full     : teacher trained on all 1,311 gold labels (Task 2 setting); students see all train inputs and gold labels.
  lowlabel : teacher trained on the 80-label seed set; students see seed + pool inputs, hard labels only on the seed set.
Students  : linear (Task 2 student) and a small MLP (64 hidden units) on the same 2,000-feature view.
(T, alpha) is chosen per condition and student family by mean DEV macro-F1 over seeds.
Writes results/research/distill_ext_runs.csv and distill_ext_summary.csv. Tasks 1-3 are not modified.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.data import make_split
from src.eval import scores
from src.models import train_model
from src.pipeline import base_parser, setup
from src.utils import ROOT, ensure_dir
from .stats import paired


def main(argv=None):
    args = base_parser("Distillation extension").parse_args(argv)
    cfg, base = setup(args)
    seeds = args.seeds if args.seeds is not None else list(range(10))
    Ts, As = cfg["task2"]["temperatures"], cfg["task2"]["alphas"]
    nb = cfg["eval"]["ece_bins"]
    cfg_mlp = copy.deepcopy(cfg)
    cfg_mlp["models"]["student"] = dict(kind="mlp", hidden=64, dropout=0.3, epochs=60, lr=0.002, weight_decay=0.0001, batch_size=32)
    fams = {"linear": cfg, "mlp64": cfg_mlp}
    Xt, Xs = base.X["teacher"], base.X["student"]
    ytr, ydev, yte = base.y["train"], base.y["dev"], base.y["test"]
    rows = []
    for seed in seeds:
        split = make_split(base, seed, cfg["data"]["seed_per_class"])
        for cond in ("full", "lowlabel"):
            if cond == "full":
                tidx, ty = np.arange(len(ytr)), ytr
                sidx, sy = np.arange(len(ytr)), ytr
            else:
                tidx, ty = split.seed_idx, ytr[split.seed_idx]
                sidx = np.concatenate([split.seed_idx, split.pool_idx])
                sy = np.full(len(sidx), -1, dtype=np.int64)
                sy[: len(split.seed_idx)] = ytr[split.seed_idx]
            teacher = train_model("teacher", Xt["train"][tidx], ty, cfg, seed)
            tl = teacher.logits(Xt["train"][sidx])
            t_s = scores(teacher.proba(Xt["test"]), yte, nb)
            rows.append({"condition": cond, "seed": seed, "family": "teacher", "model": "teacher", "T": np.nan, "alpha": np.nan,
                         "dev_f1": scores(teacher.proba(Xt["dev"]), ydev, nb)["macro_f1"], "test_f1": t_s["macro_f1"], "test_ece": t_s["ece"]})
            for fam, c in fams.items():
                hard_idx = sidx[sy >= 0]
                m = train_model("student", Xs["train"][hard_idx], sy[sy >= 0], c, seed)
                rows.append({"condition": cond, "seed": seed, "family": fam, "model": "hard", "T": np.nan, "alpha": np.nan,
                             "dev_f1": scores(m.proba(Xs["dev"]), ydev, nb)["macro_f1"],
                             "test_f1": scores(m.proba(Xs["test"]), yte, nb)["macro_f1"], "test_ece": scores(m.proba(Xs["test"]), yte, nb)["ece"]})
                for T in Ts:
                    for a in As:
                        m = train_model("student", Xs["train"][sidx], sy, c, seed, teacher_logits=tl, T=float(T), alpha=float(a))
                        s_d, s_t = scores(m.proba(Xs["dev"]), ydev, nb), scores(m.proba(Xs["test"]), yte, nb)
                        rows.append({"condition": cond, "seed": seed, "family": fam, "model": "distilled", "T": T, "alpha": a,
                                     "dev_f1": s_d["macro_f1"], "test_f1": s_t["macro_f1"], "test_ece": s_t["ece"]})
        print(f"distill_ext seed {seed} done", flush=True)
    df = pd.DataFrame(rows)
    out = ensure_dir(ROOT / "results" / "research")
    df.to_csv(out / "distill_ext_runs.csv", index=False)

    summ = []
    for cond in ("full", "lowlabel"):
        for fam in fams:
            d = df[(df.condition == cond) & (df.family == fam)]
            grid = d[d.model == "distilled"].groupby(["T", "alpha"])["dev_f1"].mean()
            (T, a) = grid.idxmax()
            dist = d[(d.model == "distilled") & (d["T"] == T) & (d.alpha == a)].set_index("seed")
            hard = d[d.model == "hard"].set_index("seed")
            diff_f1 = (dist.test_f1 - hard.test_f1).to_numpy()
            diff_ece = (dist.test_ece - hard.test_ece).to_numpy()
            r = paired(diff_f1)
            summ.append({"condition": cond, "family": fam, "T": T, "alpha": a, "hard_f1": hard.test_f1.mean(), "distilled_f1": dist.test_f1.mean(),
                         "diff_f1": r["mean_diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "wilcoxon_p": r["wilcoxon_p"], "tost_p": r["tost_p"],
                         "hard_ece": hard.test_ece.mean(), "distilled_ece": dist.test_ece.mean(), "diff_ece": float(diff_ece.mean()),
                         "teacher_f1": d[d.model == "teacher"].test_f1.mean()})
    pd.DataFrame(summ).to_csv(out / "distill_ext_summary.csv", index=False)
    print(pd.DataFrame(summ).round(4).to_string())


if __name__ == "__main__":
    main()
