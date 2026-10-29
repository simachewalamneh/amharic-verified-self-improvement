"""Builds report/report.tex from results/*.csv, then compile: cd report && pdflatex report.tex (twice not needed)."""
import pandas as pd
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "results"


def ms(m, s, p=3):
    return f"{m:.{p}f}" + (f" $\\pm$ {s:.{p}f}" if pd.notna(s) else "")


def esc(s):
    return s.replace("_", "\\_")


def tab(df, cols, header, names):
    lines = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule", header + " \\\\", "\\midrule"]
    for k in df.index:
        cells = []
        for c, p in cols:
            cells.append(ms(df.loc[k, (c, "mean")], df.loc[k, (c, "std")], p) if (c, "mean") in df.columns else "")
        lines.append(f"{names.get(k, esc(k))} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


t1 = pd.read_csv(R / "task1/final_test.csv").groupby("variant")[["test_macro_f1", "test_ece"]].agg(["mean", "std"])
t1 = t1.loc[["baseline_charLR_seed_only", "verified:round0_teacher", "verified:final_teacher", "confidence_only:final_teacher", "naive:final_teacher", "reference_full_labels"]]
n1 = {"baseline_charLR_seed_only": "Char-LR, seed only", "verified:round0_teacher": "Round-0 teacher (seed only)",
      "verified:final_teacher": "Verified loop, final teacher", "confidence_only:final_teacher": "Confidence-only loop, final",
      "naive:final_teacher": "Naive loop (no rollback), final", "reference_full_labels": "Teacher, all gold labels (ref.)"}
T1 = tab(t1, [("test_macro_f1", 3), ("test_ece", 3)], "Model & Test macro-F1 & Test ECE", n1)

a = pd.read_csv(R / "task2/all_runs.csv")
import json
best = json.load(open(R / "task2/best.json"))
ch = a[(a.model == "student_distilled") & (a["T"] == best["T"]) & (a.alpha == best["alpha"])].assign(model="student_distilled")
c2 = pd.concat([a[a.model.isin(["teacher", "student_hard", "baseline_charLR"])], ch])
t2 = c2.groupby("model")[["test_macro_f1", "test_ece", "params", "latency_ms"]].agg(["mean", "std"]).loc[["teacher", "student_hard", "student_distilled", "baseline_charLR"]]
n2 = {"teacher": "Teacher (MLP)", "student_hard": "Student, hard labels", "student_distilled": f"Student, distilled (T={best['T']:g}, $\\alpha$={best['alpha']:g})", "baseline_charLR": "Char-LR baseline"}
lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Model & Test macro-F1 & Test ECE & Params & CPU ms/example \\\\", "\\midrule"]
for k in t2.index:
    r = t2.loc[k]
    lat = ms(r[("latency_ms", "mean")], r[("latency_ms", "std")], 4) if pd.notna(r[("latency_ms", "mean")]) else "--"
    lines.append(f"{n2[k]} & {ms(r[('test_macro_f1','mean')], r[('test_macro_f1','std')])} & {ms(r[('test_ece','mean')], r[('test_ece','std')])} & {int(r[('params','mean')]):,} & {lat} \\\\".replace(",", "{,}"))
lines += ["\\bottomrule", "\\end{tabular}"]
T2 = "\n".join(lines)

d = pd.read_csv(R / "task3/all_runs.csv")
order = ["teacher_round0", "teacher_final_loop", "student_seed_hard", "student_hard_pseudo", "student_from_round0", "student_from_final", "student_task2_full_labels(ref)", "teacher_full_labels(ref)"]
t3 = d.groupby("model")[["test_macro_f1", "test_ece"]].agg(["mean", "std"]).loc[order]
n3 = {k: esc(k).replace("(ref)", "(ref.)") for k in order}
T3 = tab(t3, [("test_macro_f1", 3), ("test_ece", 3)], "Model & Test macro-F1 & Test ECE", n3)
g = pd.read_csv(R / "task3/gain_survival.csv")
gm = g.mean(numeric_only=True)

tex = r"""\documentclass[10pt]{article}
\usepackage[margin=2cm]{geometry}\usepackage{booktabs}\usepackage{amsmath}\usepackage{microtype}
\title{Verified Recursive Self-Training and Knowledge Distillation for Amharic News Topic Classification}
\author{iCog Labs AI Internship: Emerging AI Paradigms assignment}
\date{}
\begin{document}\maketitle
\section{Setup}
\textbf{Data.} Amharic MasakhaNEWS (official splits): 1{,}311 train, 188 dev, 376 test; four classes (business, health, politics, sports). Text is headline plus body, truncated to 3{,}000 characters. The dataset license was not verified.
\textbf{Protocol.} Tasks 1 and 3 use a stratified labelled seed set of 20 articles per class (80 labels); the other 1{,}231 training articles are an unlabeled pool whose gold labels are used only to measure pseudo-label precision. Dev decides accept/rollback and $(T,\alpha)$ selection; test is used only for final reporting. All numbers are mean $\pm$ std over 3 seeds (0,1,2).
\textbf{Models.} Teacher: 2-layer MLP (256 hidden) on word 1--2-gram TF-IDF (10k features). Student: linear model on a 2k-feature unigram TF-IDF view (8{,}004 parameters). Independent verifier and baseline: logistic regression on character 2--4-gram TF-IDF. AfroXLMR features are implemented but were not run.
\section{Task 1: verified recursive self-training}
Each round pseudo-labels the pool, accepts labels with confidence $\ge 0.9$ (verified variant: and agreement with the independent character model), retrains a copy on seed plus accepted examples, and accepts the copy only if dev macro-F1 does not drop; two consecutive rejections stop the loop (at most 5 rounds).
\begin{center}""" + T1 + r"""\end{center}
The verified loop gave no reliable test gain (per-seed change in test macro-F1: $+0.005$, $-0.003$, $-0.017$), although dev improved after round 1. The agreement check changed nothing relative to confidence-only. Without rollback, test macro-F1 fell by 0.11--0.17 even though pseudo-label precision stayed at 0.97--1.00; the entropy of accepted classes decreased across rounds, consistent with class drift toward easy classes (class counts were not inspected directly). The character-LR baseline trained on the seed set alone scored above the MLP teacher.
\section{Task 2: knowledge distillation}
Loss: $\alpha\,\mathrm{CE}(y,s)+(1-\alpha)\,T^2\,\mathrm{KL}\big(\mathrm{softmax}(t/T)\,\|\,\mathrm{softmax}(s/T)\big)$; sweep $T\in\{1,2,4\}$, $\alpha\in\{0.3,0.5,0.7\}$, selected on dev. Teacher trained on all gold train labels.
\begin{center}""" + T2 + r"""\end{center}
The student has about 320$\times$ fewer parameters and runs about 3.4$\times$ faster than the teacher at a cost of roughly 0.7 to 0.9 macro-F1 points. Distillation did not improve student accuracy over hard labels; it lowered test ECE. The character-LR baseline was the strongest model here but is much larger (200{,}004 parameters).
\section{Task 3: merged pipeline}
The Task 1 loop produces a teacher that is distilled into the student using $(T,\alpha)=(""" + f"{best['T']:g},{best['alpha']:g}" + r""")$ from Task 2. Hard labels exist only for the seed set during distillation.
\begin{center}""" + T3 + r"""\end{center}
Mean change (final minus round-0): teacher test """ + f"{gm['teacher_gain_test']:+.4f}" + r""", student test """ + f"{gm['student_gain_test']:+.4f}" + r""", teacher dev """ + f"{gm['teacher_gain_dev']:+.4f}" + r""", student dev """ + f"{gm['student_gain_dev']:+.4f}" + r""". Because the teacher gain on test is itself within noise, these results do not show that the loop's improvement survives distillation; they are a null result with 3 seeds.
\section{Limitations}
Three seeds and a 188-example dev set make accept/rollback decisions noisy. The teacher is underconfident at small label budgets (high ECE), which limits how many pseudo-labels pass the 0.9 threshold. The verifier is a proxy rather than ground truth. TF-IDF features were used throughout, so conclusions may differ with an AfroXLMR encoder. Dataset license unverified.
\end{document}
"""
(Path(__file__).resolve().parent / "report.tex").write_text(tex, encoding="utf-8")
print("wrote report.tex")
