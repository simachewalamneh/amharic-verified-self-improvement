"""Paired statistics for the research extension.

python -m research.stats
Reads results/research/*_runs.jsonl and preds/*.npz; writes results/research/analysis_*.csv and analysis_facts.json.
Tests (all paired by seed, i.e. same seed-set draw): bootstrap 95% CI of the mean difference over seeds,
Wilcoxon signed-rank (two-sided) and a TOST equivalence test (paired t, margin 0.01 macro-F1).
Also: per-example paired bootstrap on dev/test within each seed (SE of the dev gain, CI of the test gain).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats as st

from src.data import load_base
from src.eval import macro_f1
from src.utils import ROOT, load_config

R = ROOT / "results" / "research"
RNG = np.random.default_rng(0)


def load(name: str) -> pd.DataFrame:
    p = R / f"{name}_runs.jsonl"
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")]) if p.exists() else pd.DataFrame()


def paired(d: np.ndarray, margin: float = 0.01) -> dict:
    d = np.asarray(d, dtype=float)
    n = len(d)
    boots = RNG.choice(d, size=(10000, n), replace=True).mean(1)
    out = {"n": n, "mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)) if n > 1 else float("nan"),
           "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5)),
           "n_improved": int((d > 0).sum()), "n_worsened": int((d < 0).sum())}
    try:
        out["wilcoxon_p"] = float(st.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    if n > 1 and d.std(ddof=1) > 0:
        p1 = st.ttest_1samp(d, -margin, alternative="greater").pvalue
        p2 = st.ttest_1samp(d, margin, alternative="less").pvalue
        out["tost_p"] = float(max(p1, p2))
    else:
        out["tost_p"] = float("nan")
    return out


def diffs(df: pd.DataFrame, a: str, b: str, col: str, budget=20, ref_df=None, ref_col=None) -> np.ndarray:
    """Per-seed difference (variant a minus variant b) of `col`, paired by seed."""
    A = df[(df.variant == a) & (df.budget == budget)].set_index("seed")[col]
    B = (ref_df if ref_df is not None else df)
    B = B[(B.variant == b) & (B.budget == budget)].set_index("seed")[ref_col or col]
    j = A.index.intersection(B.index)
    return (A.loc[j] - B.loc[j]).to_numpy()


def per_example_bootstrap(suite, variant, budget, seed, y_dev, y_test, B=1000):
    z = np.load(R / "preds" / f"{suite}__{variant}__{budget}__s{seed}.npz")
    out = {}
    for split, y in (("dev", y_dev), ("test", y_test)):
        a, b = z[f"{split}_r0"], z[f"{split}_final"]
        n = len(y)
        diffs_ = np.empty(B)
        for i in range(B):
            idx = RNG.integers(0, n, n)
            diffs_[i] = macro_f1(y[idx], b[idx]) - macro_f1(y[idx], a[idx])
        out[split] = (macro_f1(y, b) - macro_f1(y, a), float(diffs_.std()), np.percentile(diffs_, [2.5, 97.5]))
    return out


def main():
    cfg = load_config()
    base = load_base(cfg)
    y_dev, y_test = base.y["dev"], base.y["test"]
    facts = {}
    main_df, ch_df, bud_df, thr_df = load("main"), load("charlr"), load("budget"), load("threshold")

    # ---- paired comparisons --------------------------------------------------------------
    comps = []

    def add(label, suite, d):
        r = paired(d)
        r.update({"suite": suite, "comparison": label})
        comps.append(r)

    if len(main_df):
        for a, b, lab in (("verified", "seed_only", "verified loop - seed-only teacher"),
                          ("naive", "seed_only", "naive loop - seed-only teacher"),
                          ("verified_margin", "seed_only", "verified (margin 0.01) - seed-only teacher"),
                          ("cap", "naive", "per-class cap - naive"),
                          ("balanced", "naive", "balanced selection - naive"),
                          ("balanced", "seed_only", "balanced (no rollback) - seed-only teacher"),
                          ("balanced_rb", "seed_only", "balanced + rollback - seed-only teacher"),
                          ("balanced_rb", "verified", "balanced + rollback - verified")):
            add(lab + " [test F1]", "main", diffs(main_df, a, b, "test_f1_final"))
        add("verified loop - seed-only teacher [dev F1]", "main", diffs(main_df, "verified", "seed_only", "dev_f1_final"))
    if len(ch_df):
        for a, b, lab in (("verified", "seed_only", "verified - seed-only"), ("conf_rb", "seed_only", "confidence+rollback - seed-only"),
                          ("naive", "seed_only", "naive - seed-only"), ("balanced_rb", "seed_only", "balanced+rollback - seed-only"),
                          ("verified", "conf_rb", "verified - confidence+rollback"), ("verified_uncal", "seed_only", "verified, uncalibrated - seed-only"),
                          ("verified_thr0.7", "seed_only", "verified, threshold 0.7 - seed-only")):
            add("char-LR teacher: " + lab + " [test F1]", "charlr", diffs(ch_df, a, b, "test_f1_final"))
    if len(main_df) and len(ch_df):
        add("char-LR seed-only - MLP seed-only [test F1]", "main+charlr",
            diffs(ch_df, "seed_only", "seed_only", "test_f1_final", ref_df=main_df))
    pd.DataFrame(comps).to_csv(R / "analysis_paired.csv", index=False)

    # ---- variant summary tables ---------------------------------------------------------------
    names = base.label_names
    for suite, df in (("main", main_df), ("charlr", ch_df)):
        if not len(df):
            continue
        agg = {"test_f1_r0": "mean", "test_f1_final": ["mean", "std"], "dev_f1_r0": "mean", "dev_f1_final": "mean",
               "test_ece_final": "mean", "n_accepted": "mean", "final_pl_precision": "mean", "class_entropy": "mean"}
        for c in names:
            agg[f"acc_{c}"] = "mean"
            agg[f"rate_{c}"] = "mean"
        df[df.budget == 20].groupby("variant").agg(agg).to_csv(R / f"analysis_{suite}_variants.csv")

    # ---- verifier activity ---------------------------------------------------------------------
    rows = []
    for suite, df in (("main", main_df), ("charlr", ch_df)):
        if not len(df):
            continue
        d = df[(df.budget == 20) & (df.sum_selected > 0)]
        for v, g in d.groupby("variant"):
            sel, dis = g.sum_selected.sum(), g.sum_disagree.sum()
            rows.append({"suite": suite, "variant": v, "selected": int(sel), "disagreements": int(dis),
                         "disagree_rate": float(dis / sel) if sel else float("nan"),
                         "teacher_correct_on_disagree": int(g.sum_teacher_correct_on_disagree.sum()),
                         "verifier_correct_on_disagree": int(g.sum_verifier_correct_on_disagree.sum())})
    pd.DataFrame(rows).to_csv(R / "analysis_verifier.csv", index=False)

    # ---- label budgets and thresholds -----------------------------------------------------------
    if len(bud_df) or len(main_df):
        allb = pd.concat([bud_df, main_df[main_df.seed < 5]]) if len(bud_df) else main_df
        allb = allb[allb.variant.isin(["seed_only", "naive", "verified", "balanced_rb"])]
        rows = []
        for (b, v), g in allb.groupby(["budget", "variant"]):
            rows.append({"budget_per_class": b, "variant": v, "n_seeds": len(g), "test_f1_r0": g.test_f1_r0.mean(),
                         "test_f1_final": g.test_f1_final.mean(), "test_f1_final_sd": g.test_f1_final.std(),
                         "n_accepted": g.n_accepted.mean()})
        pd.DataFrame(rows).sort_values(["budget_per_class", "variant"]).to_csv(R / "analysis_budget.csv", index=False)
    if len(thr_df):
        thr_df.groupby("variant").agg(test_f1_r0=("test_f1_r0", "mean"), test_f1_final=("test_f1_final", "mean"),
                                       n_accepted=("n_accepted", "mean"), pl_precision=("final_pl_precision", "mean"),
                                       class_entropy=("class_entropy", "mean")).to_csv(R / "analysis_threshold.csv")

    # ---- dev optimism: per-example bootstrap ---------------------------------------------------------
    if len(main_df):
        rows = []
        for variant in ("verified", "naive"):
            for seed in sorted(main_df[main_df.variant == variant].seed.unique()):
                r = per_example_bootstrap("main", variant, 20, seed, y_dev, y_test)
                rows.append({"variant": variant, "seed": seed, "dev_gain": r["dev"][0], "dev_se": r["dev"][1],
                             "test_gain": r["test"][0], "test_se": r["test"][1],
                             "test_ci_lo": r["test"][2][0], "test_ci_hi": r["test"][2][1]})
        pe = pd.DataFrame(rows)
        pe.to_csv(R / "analysis_per_example.csv", index=False)
        v = pe[pe.variant == "verified"]
        facts["dev_gain_mean"] = float(v.dev_gain.mean())
        facts["dev_se_mean"] = float(v.dev_se.mean())
        facts["dev_gain_gt_2se"] = int((v.dev_gain > 2 * v.dev_se).sum())
        facts["test_ci_excludes_zero_verified"] = int(((v.test_ci_lo > 0) | (v.test_ci_hi < 0)).sum())
        nv = pe[pe.variant == "naive"]
        facts["test_ci_excludes_zero_naive"] = int(((nv.test_ci_lo > 0) | (nv.test_ci_hi < 0)).sum())
        facts["n_seeds_main"] = int(len(v))
    json.dump(facts, open(R / "analysis_facts.json", "w"), indent=2)
    print("analysis written:", sorted(p.name for p in R.glob("analysis_*")))


if __name__ == "__main__":
    main()
