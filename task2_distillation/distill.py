"""Task 2: knowledge distillation (teacher -> small student), compared with a hard-label student.

Teacher : MLP on the 'teacher' feature view, trained on the FULL labelled train set (independent of Task 1).
Student : linear model on the smaller 'student' feature view.
Loss    : alpha * CE(y, s) + (1 - alpha) * T^2 * KL(softmax(t/T) || softmax(s/T))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval import full_report
from src.models import CharLR, train_model
from src.eval import scores


def run_distillation(base, cfg: dict, seeds, log=print) -> pd.DataFrame:
    t2 = cfg["task2"]
    Xt, Xs, Xv = base.X["teacher"], base.X["student"], base.X["verifier"]
    y_tr, y_dev, y_te = base.y["train"], base.y["dev"], base.y["test"]
    rows = []
    for seed in seeds:
        teacher = train_model("teacher", Xt["train"], y_tr, cfg, seed)
        rep = full_report(teacher, Xt["dev"], y_dev, Xt["test"], y_te, cfg)
        rows.append({"seed": seed, "model": "teacher", "T": np.nan, "alpha": np.nan, **rep})
        t_logits = teacher.logits(Xt["train"])
        log(f"  [seed {seed}] teacher test F1={rep['test_macro_f1']:.4f} params={rep['params']}")

        hard = train_model("student", Xs["train"], y_tr, cfg, seed)
        rep = full_report(hard, Xs["dev"], y_dev, Xs["test"], y_te, cfg)
        rows.append({"seed": seed, "model": "student_hard", "T": np.nan, "alpha": np.nan, **rep})
        log(f"  [seed {seed}] hard-label student test F1={rep['test_macro_f1']:.4f} params={rep['params']}")

        for T in t2["temperatures"]:
            for a in t2["alphas"]:
                st = train_model("student", Xs["train"], y_tr, cfg, seed, teacher_logits=t_logits, T=float(T), alpha=float(a))
                rep = full_report(st, Xs["dev"], y_dev, Xs["test"], y_te, cfg)
                rows.append({"seed": seed, "model": "student_distilled", "T": T, "alpha": a, **rep})
        lr = CharLR(cfg["models"]["verifier_lr_C"], base.n_classes).fit(Xv["train"], y_tr)
        nb = cfg["eval"]["ece_bins"]
        d, t = scores(lr.proba(Xv["dev"]), y_dev, nb), scores(lr.proba(Xv["test"]), y_te, nb)
        rows.append({"seed": seed, "model": "baseline_charLR", "T": np.nan, "alpha": np.nan,
                     "dev_macro_f1": d["macro_f1"], "dev_ece": d["ece"], "test_macro_f1": t["macro_f1"],
                     "test_accuracy": t["accuracy"], "test_ece": t["ece"], "params": lr.n_params,
                     "size_kb": lr.n_params * 4 / 1024.0, "latency_ms": np.nan})
    return pd.DataFrame(rows)


def select_best(df: pd.DataFrame) -> dict:
    """Pick (T, alpha) by mean DEV macro-F1 across seeds (never by test)."""
    d = df[df["model"] == "student_distilled"].groupby(["T", "alpha"])["dev_macro_f1"].mean().sort_values(ascending=False)
    (T, a), dev_f1 = d.index[0], float(d.iloc[0])
    return {"T": float(T), "alpha": float(a), "mean_dev_macro_f1": dev_f1}
