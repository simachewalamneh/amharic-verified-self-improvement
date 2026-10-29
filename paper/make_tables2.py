"""Builds paper/tables2/*.tex and paper/tables2/macros.tex from results/ (no number in the paper is typed by hand).
Run after: python -m paper.research.suite --suite all; python -m paper.research.diagnostics; python -m paper.research.distill_ext;
python -m paper.research.latency_e2e; python -m paper.research.stats; python -m task3_merged.run (10 seeds).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from paper.research.stats import load, paired  # noqa: E402

R = ROOT / "results" / "research"
OUT = Path(__file__).resolve().parent / "tables2"
OUT.mkdir(exist_ok=True)
PM = r"\,$\pm$\,"
MAC = {}


def neg(s):
    return s.replace("-", "$-$")


def f3(x, p=3):
    return neg(f"{x:.{p}f}")


def sg(x, p=3):
    return neg(f"{x:+.{p}f}")


def ci(m, lo, hi, p=3):
    return f"{sg(m, p)} [{sg(lo, p)}, {sg(hi, p)}]"


def pv(p):
    if p != p:
        return "--"
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def ms(m, s, p=3):
    return f"{f3(m, p)}{PM}{f3(s, p)}"


def write(name, lines):
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


_DIG = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]


def macro(name, value):
    """LaTeX macro names may contain letters only: digits are spelled out."""
    name = "".join(_DIG[int(ch)] if ch.isdigit() else ch for ch in name if ch not in "._")
    MAC[name] = value


def paired_vs(df, a, b, col="test_f1_final", budget=20, ref=None):
    A = df[(df.variant == a) & (df.budget == budget)].set_index("seed")[col]
    Bd = (ref if ref is not None else df)
    B = Bd[(Bd.variant == b) & (Bd.budget == budget)].set_index("seed")[col]
    j = A.index.intersection(B.index)
    return paired((A.loc[j] - B.loc[j]).to_numpy())


main, ch = load("main"), load("charlr")
bud, thr = load("budget"), load("threshold")
classes = ["business", "health", "politics", "sports"]
VL = {"seed_only": "Seed-only teacher (no loop)", "verified": "Verified loop (confidence + agreement + rollback)",
      "verified_margin": "Verified loop, acceptance margin 0.01", "naive": "Naive loop (confidence, no rollback)",
      "cap": "Per-class cap (10 per class per round), no rollback", "balanced": "Balanced top-$k$ per class, no rollback",
      "balanced_rb": "Balanced top-$k$ per class + rollback"}

# ---------------------------------------------------------------- main table
L = [r"\begin{tabular}{lccccc}", r"\toprule",
     r"Variant & Test macro-F1 & $\Delta$ vs seed-only [95\% CI] & Wilcoxon $p$ & Accepted & PL precision \\", r"\midrule"]
for v in ["seed_only", "verified", "verified_margin", "naive", "cap", "balanced", "balanced_rb"]:
    d = main[(main.variant == v) & (main.budget == 20)]
    row = f"{VL[v]} & {ms(d.test_f1_final.mean(), d.test_f1_final.std())} & "
    if v == "seed_only":
        row += "-- & -- & 0 & --"
    else:
        r = paired_vs(main, v, "seed_only")
        macro("cmpMain" + "".join(w.capitalize() for w in v.split("_")), ci(r["mean_diff"], r["ci_lo"], r["ci_hi"]))
        macro("pMain" + "".join(w.capitalize() for w in v.split("_")), pv(r["wilcoxon_p"]))
        macro("nimpMain" + "".join(w.capitalize() for w in v.split("_")), f"{r['n_improved']}/{r['n']}")
        if v == "verified":
            macro("tostMainVerified", pv(r["tost_p"]))
        pr = d.final_pl_precision.mean()
        row += f"{ci(r['mean_diff'], r['ci_lo'], r['ci_hi'])} & {pv(r['wilcoxon_p'])} & {d.n_accepted.mean():.0f} & {f3(pr)}"
    L.append(row + r" \\")
    if v == "seed_only":
        L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}"]
write("main.tex", L)
for v in ["seed_only", "verified", "naive", "cap", "balanced", "balanced_rb", "verified_margin"]:
    d = main[(main.variant == v) & (main.budget == 20)]
    key = "".join(w.capitalize() for w in v.split("_"))
    macro("fMain" + key, f3(d.test_f1_final.mean()))
    macro("sdMain" + key, f3(d.test_f1_final.std()))
    macro("accMain" + key, f"{d.n_accepted.mean():.0f}")
    if v != "seed_only":
        macro("precMain" + key, f3(d.final_pl_precision.mean()))
        macro("entMain" + key, f3(d.class_entropy.mean()))
for a, b, nm in (("cap", "naive", "CapVsNaive"), ("balanced", "naive", "BalancedVsNaive"), ("balanced_rb", "verified", "BalancedRbVsVerified")):
    r = paired_vs(main, a, b)
    macro("cmp" + nm, ci(r["mean_diff"], r["ci_lo"], r["ci_hi"]))
    macro("p" + nm, pv(r["wilcoxon_p"]))
macro("devGainVerified", sg(float(json.load(open(R / "analysis_facts.json"))["dev_gain_mean"])))
facts = json.load(open(R / "analysis_facts.json"))
macro("devSeVerified", f3(facts["dev_se_mean"]))
macro("devGainGtTwoSe", str(facts["dev_gain_gt_2se"]))
macro("testCiExclNaive", str(facts["test_ci_excludes_zero_naive"]))
macro("testCiExclVerified", str(facts["test_ci_excludes_zero_verified"]))
macro("nSeeds", str(facts["n_seeds_main"]))
r = paired_vs(main, "verified", "seed_only", col="dev_f1_final")
macro("cmpDevVerified", ci(r["mean_diff"], r["ci_lo"], r["ci_hi"]))

# ---------------------------------------------------------------- class composition
L = [r"\begin{tabular}{lcccc}", r"\toprule", r"Variant & Business & Health & Politics & Sports \\", r"\midrule"]
pool = {"business": 262, "health": 330, "politics": 330, "sports": 309}
for v in ["naive", "cap", "balanced", "verified"]:
    d = main[(main.variant == v) & (main.budget == 20)]
    L.append(f"{VL[v]} & " + " & ".join(f"{d[f'acc_{c}'].mean():.1f} ({100 * d[f'rate_{c}'].mean():.1f}\\%)" for c in classes) + r" \\")
for c in classes:
    d = main[(main.variant == "naive") & (main.budget == 20)]
    macro("accNaive" + c.capitalize(), f"{d[f'acc_{c}'].mean():.1f}")
    macro("zeroNaive" + c.capitalize(), str(int((d[f"acc_{c}"] == 0).sum())))
L += [r"\bottomrule", r"\end{tabular}"]
write("classes.tex", L)

# ---------------------------------------------------------------- diagnostics
if (R / "diag_per_class.csv").exists():
    dc = pd.read_csv(R / "diag_per_class.csv")
    L = [r"\begin{tabular}{llcccccc}", r"\toprule",
         r"Teacher & Class & Pool & Mean conf. & Median conf. & Conf.\ $\ge$ 0.9 (\%) & Confidently wrong & Recall \\", r"\midrule"]
    for t, tn in (("mlp", "MLP (raw)"), ("charlr", "Char-LR (calibrated)")):
        for c in classes:
            d = dc[(dc.teacher == t) & (dc["class"] == c)]
            L.append(f"{tn if c == 'business' else ''} & {c.capitalize()} & {d.n_pool.mean():.0f} & {f3(d.mean_conf.mean())} & {f3(d.median_conf.mean())} & "
                     f"{100 * d.share_conf_ge_thr.mean():.1f} & {d.confidently_wrong.mean():.1f} & {f3(d.recall.mean())} \\\\")
            if t == "mlp":
                macro("share09Mlp" + c.capitalize(), f"{100 * d.share_conf_ge_thr.mean():.1f}")
                macro("meanConfMlp" + c.capitalize(), f3(d.mean_conf.mean()))
                macro("recallMlp" + c.capitalize(), f3(d.recall.mean()))
                macro("cwrongMlp" + c.capitalize(), f"{d.confidently_wrong.mean():.1f}")
        if t == "mlp":
            L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("diag_class.tex", L)

    rel = pd.read_csv(R / "diag_reliability.csv")
    L = [r"\begin{tabular}{lrcc}", r"\toprule", r"Confidence bin & $n$ & Mean confidence & Accuracy \\", r"\midrule"]
    g = rel[rel.teacher == "mlp"].groupby(["bin_lo", "bin_hi"])[["n", "sum_conf", "n_correct"]].sum().reset_index()
    for _, r in g.iterrows():
        if r.n > 0:
            L.append(f"({r.bin_lo:.2f}, {r.bin_hi:.2f}] & {int(r.n)} & {f3(r.sum_conf / r.n)} & {f3(r.n_correct / r.n)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("reliability.tex", L)
    cf = pd.read_csv(R / "diag_confidence.csv")
    for t in ("mlp", "charlr"):
        d = cf[cf.teacher == t]
        macro("meanConf" + t.capitalize(), f3(d.mean_conf.mean()))
        macro("accRound0" + t.capitalize(), f3(d.accuracy.mean()))
    cm = pd.read_csv(R / "diag_confusion_mlp.csv", index_col=0)
    L = [r"\begin{tabular}{lrrrr}", r"\toprule", r"True $\backslash$ predicted & Business & Health & Politics & Sports \\", r"\midrule"]
    for k, row in cm.iterrows():
        L.append(f"{k.capitalize()} & " + " & ".join(str(int(x)) for x in row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("confusion.tex", L)

# ---------------------------------------------------------------- verifier activity
if (R / "analysis_verifier.csv").exists():
    va = pd.read_csv(R / "analysis_verifier.csv")
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule", r"Teacher & Variant & Selected & Disagreements & Rate (\%) & Teacher right & Verifier right \\", r"\midrule"]
    names = {"naive": "naive", "conf_rb": "confidence + rollback", "verified": "verified", "verified_thr0.7": "verified, thr.\\ 0.7"}
    for suite, tn in (("main", "MLP"), ("charlr", "Char-LR")):
        for v in ("naive", "conf_rb", "verified", "verified_thr0.7"):
            d = va[(va.suite == suite) & (va.variant == v)]
            if len(d):
                r = d.iloc[0]
                L.append(f"{tn} & {names[v]} & {int(r.selected):,} & {int(r.disagreements)} & {100 * r.disagree_rate:.1f} & {int(r.teacher_correct_on_disagree)} & {int(r.verifier_correct_on_disagree)} \\\\".replace(",", "{,}") if False else
                         f"{tn} & {names[v]} & {int(r.selected)} & {int(r.disagreements)} & {100 * r.disagree_rate:.1f} & {int(r.teacher_correct_on_disagree)} & {int(r.verifier_correct_on_disagree)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("verifier.tex", L)
    d = va[(va.suite == "charlr") & (va.variant == "verified_thr0.7")].iloc[0]
    macro("verThrSel", str(int(d.selected))); macro("verThrDis", str(int(d.disagreements)))
    macro("verThrRate", f"{100 * d.disagree_rate:.1f}")
    macro("verThrTeacherRight", str(int(d.teacher_correct_on_disagree))); macro("verThrVerRight", str(int(d.verifier_correct_on_disagree)))
    d = va[(va.suite == "charlr") & (va.variant == "verified")].iloc[0]
    macro("verCharSel", str(int(d.selected))); macro("verCharDis", str(int(d.disagreements)))
    macro("verCharTeacherRight", str(int(d.teacher_correct_on_disagree))); macro("verCharVerRight", str(int(d.verifier_correct_on_disagree)))
    d = va[(va.suite == "main") & (va.variant == "verified")].iloc[0]
    macro("verMainSel", str(int(d.selected))); macro("verMainDis", str(int(d.disagreements)))
    d = va[(va.suite == "main") & (va.variant == "naive")].iloc[0]
    macro("verNaiveSel", str(int(d.selected))); macro("verNaiveDis", str(int(d.disagreements)))
if (R / "diag_verifier.csv").exists():
    dv = pd.read_csv(R / "diag_verifier.csv")
    L = [r"\begin{tabular}{lcccccc}", r"\toprule", r"Teacher & Agreement & Teacher acc. & Verifier acc. & Error $\phi$ & P(both wrong) & P(ver.\ right $|$ teacher wrong) \\", r"\midrule"]
    for t, tn in (("mlp", "MLP vs char-LR"), ("charlr", "Char-LR vs word-LR")):
        d = dv[dv.teacher == t]
        L.append(f"{tn} & {f3(d.agree_rate.mean())} & {f3(d.teacher_acc.mean())} & {f3(d.verifier_acc.mean())} & {f3(d.error_phi.mean())} & "
                 f"{f3(d.p_both_wrong.mean())} & {f3(d.p_verifier_right_given_teacher_wrong.mean())} \\\\")
        macro("phi" + t.capitalize(), f3(d.error_phi.mean()))
        macro("pvr" + t.capitalize(), f3(d.p_verifier_right_given_teacher_wrong.mean()))
        macro("agree" + t.capitalize(), f3(d.agree_rate.mean()))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("verifier_corr.tex", L)

# ---------------------------------------------------------------- char-LR teacher
CL = {"seed_only": "Seed-only char-LR (calibrated)", "naive": "Naive (no rollback)", "conf_rb": "Confidence + rollback",
      "verified": "Verified (agreement + rollback)", "verified_thr0.7": "Verified, threshold 0.7",
      "verified_uncal": "Verified, uncalibrated confidence", "balanced_rb": "Balanced top-$k$ + rollback"}
L = [r"\begin{tabular}{lccccc}", r"\toprule", r"Variant & Test macro-F1 & $\Delta$ vs seed-only [95\% CI] & Wilcoxon $p$ & Accepted & PL precision \\", r"\midrule"]
for v in ["seed_only", "naive", "conf_rb", "verified", "verified_thr0.7", "verified_uncal", "balanced_rb"]:
    d = ch[(ch.variant == v) & (ch.budget == 20)]
    key = "".join(w.capitalize() for w in v.replace(".", "").split("_"))
    macro("fChar" + key, f3(d.test_f1_final.mean()))
    if v == "seed_only":
        L.append(f"{CL[v]} & {ms(d.test_f1_final.mean(), d.test_f1_final.std())} & -- & -- & 0 & -- \\\\")
        L.append(r"\addlinespace")
    else:
        r = paired_vs(ch, v, "seed_only")
        macro("cmpChar" + key, ci(r["mean_diff"], r["ci_lo"], r["ci_hi"]))
        macro("pChar" + key, pv(r["wilcoxon_p"]))
        macro("nimpChar" + key, f"{r['n_improved']}/{r['n']}")
        L.append(f"{CL[v]} & {ms(d.test_f1_final.mean(), d.test_f1_final.std())} & {ci(r['mean_diff'], r['ci_lo'], r['ci_hi'])} & {pv(r['wilcoxon_p'])} & "
                 f"{d.n_accepted.mean():.0f} & {f3(d.final_pl_precision.mean()) if d.final_pl_precision.notna().any() else '--'} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
write("charlr.tex", L)
r = paired_vs(ch, "verified", "conf_rb")
macro("cmpCharAgreeEffect", ci(r["mean_diff"], r["ci_lo"], r["ci_hi"], 4)); macro("pCharAgreeEffect", pv(r["wilcoxon_p"]))
r = paired(( ch[(ch.variant == "seed_only") & (ch.budget == 20)].set_index("seed").test_f1_final
            - main[(main.variant == "seed_only") & (main.budget == 20)].set_index("seed").test_f1_final).to_numpy())
macro("cmpCharVsMlp", ci(r["mean_diff"], r["ci_lo"], r["ci_hi"])); macro("pCharVsMlp", pv(r["wilcoxon_p"]))
macro("fCharNaiveMinusMlpNaive", sg(ch[(ch.variant == "naive")].test_f1_final.mean() - main[(main.variant == "naive")].test_f1_final.mean()))

# ---------------------------------------------------------------- label budget and thresholds
if len(bud):
    allb = pd.concat([bud, main[main.seed < 5]])
    allb = allb[allb.variant.isin(["seed_only", "naive", "verified", "balanced_rb"])]
    L = [r"\begin{tabular}{lcccc}", r"\toprule", r"Labels per class (total) & Seed-only & Verified & Naive & Balanced + rollback \\", r"\midrule"]
    for b in (5, 10, 20, 40):
        d = allb[allb.budget == b]
        cells = []
        for v in ["seed_only", "verified", "naive", "balanced_rb"]:
            x = d[d.variant == v].test_f1_final
            cells.append(ms(x.mean(), x.std()))
            macro(f"bud{b}{''.join(w.capitalize() for w in v.split('_'))}", f3(x.mean()))
        L.append(f"{b} ({4 * b}) & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("budget.tex", L)
if len(thr):
    L = [r"\begin{tabular}{lccccc}", r"\toprule", r"Threshold & Variant & Test macro-F1 & $\Delta$ vs seed-only & Accepted & PL precision \\", r"\midrule"]
    sub = main[main.seed < 5]
    for t in (0.5, 0.7, 0.9):
        for v, nm in (("naive", "naive"), ("verified", "verified")):
            d = (sub[(sub.variant == v) & (sub.budget == 20)] if t == 0.9 else thr[(thr.variant == f"{v}_thr{t}") & (thr.budget == 20)])
            base0 = d.test_f1_r0.mean()
            L.append(f"{t} & {nm} & {ms(d.test_f1_final.mean(), d.test_f1_final.std())} & {sg(d.test_f1_final.mean() - base0)} & {d.n_accepted.mean():.0f} & "
                     f"{f3(d.final_pl_precision.mean()) if d.final_pl_precision.notna().any() else '--'} \\\\")
            macro(f"thr{str(t).replace('.', '')}{nm.capitalize()}", f3(d.test_f1_final.mean()))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("threshold.tex", L)

# ---------------------------------------------------------------- distillation extension
if (R / "distill_ext_summary.csv").exists():
    ds = pd.read_csv(R / "distill_ext_summary.csv")
    runs_d = pd.read_csv(R / "distill_ext_runs.csv")
    L = [r"\begin{tabular}{llcccccc}", r"\toprule", r"Condition & Student & $(T,\alpha)$ & Hard F1 & Distilled F1 & $\Delta$ [95\% CI] & Wilcoxon $p$ & $\Delta$ECE \\", r"\midrule"]
    for _, r in ds.iterrows():
        cond = "All gold labels" if r.condition == "full" else "80 labels"
        fam = "Linear" if r.family == "linear" else "MLP (64)"
        L.append(f"{cond} & {fam} & ({r['T']:g}, {r.alpha:g}) & {f3(r.hard_f1)} & {f3(r.distilled_f1)} & {ci(r.diff_f1, r.ci_lo, r.ci_hi)} & {pv(r.wilcoxon_p)} & {sg(r.diff_ece)} \\\\")
        k = ("Full" if r.condition == "full" else "Low") + ("Linear" if r.family == "linear" else "Mlp")
        macro("dsDiff" + k, ci(r.diff_f1, r.ci_lo, r.ci_hi)); macro("dsP" + k, pv(r.wilcoxon_p)); macro("dsTost" + k, pv(r.tost_p))
        macro("dsHard" + k, f3(r.hard_f1)); macro("dsDist" + k, f3(r.distilled_f1)); macro("dsTeacher" + k, f3(runs_d[(runs_d.condition == r.condition) & (runs_d.family == "teacher")].test_f1.mean()))
        macro("dsDece" + k, sg(r.diff_ece)); macro("dsT" + k, f"{r['T']:g}"); macro("dsA" + k, f"{r.alpha:g}")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("distill_ext.tex", L)
if (R / "latency_e2e.json").exists():
    lj = json.load(open(R / "latency_e2e.json"))
    L = [r"\begin{tabular}{lccc}", r"\toprule", r"Model & Vectorise (ms) & Forward (ms) & End-to-end (ms) \\", r"\midrule"]
    for k in ("teacher", "student"):
        L.append(f"{k.capitalize()} & {lj[k]['vectorise_ms']:.3f} & {lj[k]['forward_ms']:.3f} & {lj[k]['end_to_end_ms']:.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("latency.tex", L)
    macro("latSpeedup", f"{lj['end_to_end_speedup']:.1f}")
    macro("latTeacher", f"{lj['teacher']['end_to_end_ms']:.2f}"); macro("latStudent", f"{lj['student']['end_to_end_ms']:.2f}")
    macro("fwdSpeedup", f"{lj['teacher']['forward_ms'] / lj['student']['forward_ms']:.1f}")

# ---------------------------------------------------------------- merged pipeline, 10 seeds
t3_parts = [pd.read_csv(p) for p in (ROOT / "results/task3/all_runs.csv", ROOT / "results/task3_seeds3to9/all_runs.csv") if p.exists()]
if t3_parts:
    d3 = pd.concat(t3_parts).drop_duplicates(["seed", "model"])
    order = [("teacher_round0", "Teacher, round 0"), ("teacher_final_loop", "Teacher, after loop"), ("student_seed_hard", "Student, hard labels (seed set)"),
             ("student_hard_pseudo", "Student, hard labels (seed + accepted)"), ("student_from_round0", "Student, distilled from round-0 teacher"),
             ("student_from_final", "Student, distilled from loop teacher"), ("student_task2_full_labels(ref)", "Student, distilled, full-label teacher (ref.)"),
             ("teacher_full_labels(ref)", "Teacher, all gold labels (ref.)")]
    L = [r"\begin{tabular}{lcc}", r"\toprule", r"Model & Test macro-F1 & Test ECE \\", r"\midrule"]
    for k, nm in order:
        x = d3[d3.model == k]
        L.append(f"{nm} & {ms(x.test_macro_f1.mean(), x.test_macro_f1.std())} & {ms(x.test_ece.mean(), x.test_ece.std())} \\\\")
        if k in ("teacher_final_loop", "student_from_final"):
            L.append(r"\addlinespace")
        macro("merged" + "".join(w.capitalize() for w in k.replace("(ref)", "Ref").split("_")), f3(x.test_macro_f1.mean()))
    L += [r"\bottomrule", r"\end{tabular}"]
    write("task3.tex", L)
    piv = d3.pivot(index="seed", columns="model", values="test_macro_f1")
    pdev = d3.pivot(index="seed", columns="model", values="dev_macro_f1")
    for nm, a, b, P in (("TeacherTest", "teacher_final_loop", "teacher_round0", piv), ("StudentTest", "student_from_final", "student_from_round0", piv),
                        ("TeacherDev", "teacher_final_loop", "teacher_round0", pdev), ("StudentDev", "student_from_final", "student_from_round0", pdev)):
        r = paired((P[a] - P[b]).to_numpy())
        macro("mergedGain" + nm, ci(r["mean_diff"], r["ci_lo"], r["ci_hi"])); macro("mergedP" + nm, pv(r["wilcoxon_p"]))
    macro("mergedN", str(len(piv)))
    r = paired((piv["student_from_final"] - piv["student_seed_hard"]).to_numpy())
    macro("mergedDistVsSeedHard", ci(r["mean_diff"], r["ci_lo"], r["ci_hi"])); macro("mergedPDistVsSeedHard", pv(r["wilcoxon_p"]))

lines = [r"% generated by paper/make_tables2.py"]
for k, v in sorted(MAC.items()):
    lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
(OUT / "macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("tables2:", sorted(p.name for p in OUT.iterdir()))
