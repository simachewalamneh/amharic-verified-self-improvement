"""Task 3 runner:  python -m task3_merged.run [--seeds 0 1 2]

Merged pipeline: run the Task 1 loop to obtain an improved teacher, then distill it (Task 2 loss) into a small student.

Distillation transfer set = seed + pool INPUTS. Hard labels exist ONLY for the seed set (label -1 elsewhere), so
the CE term never sees pseudo-labels or hidden gold labels; the soft-target term sees every input.

Compared (all students use the same student architecture, T and alpha):
  student_seed_hard        : hard labels, seed set only
  student_from_round0      : distilled from the round-0 teacher (seed-only teacher)
  student_from_final       : distilled from the final teacher produced by the Task 1 loop
  student_hard_pseudo      : hard labels on seed + accepted pseudo-labels (no soft targets)
  student_task2_full_labels: Task 2 style student, distilled from a teacher trained on ALL gold train labels (reference)
Key question: does the loop's teacher gain survive distillation?
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data import make_split
from src.eval import full_report
from src.models import train_model
from src.pipeline import base_parser, setup
from src.utils import ROOT, ensure_dir, save_json
from task1_recursive.selftrain import run_selftrain


def _resolve_T_alpha(cfg: dict):
    T, a = cfg["task3"]["temperature"], cfg["task3"]["alpha"]
    if T is None or a is None:
        p = ROOT / "results" / "task2" / "best.json"
        if p.exists():
            best = json.load(open(p, encoding="utf-8"))
            T = best["T"] if T is None else T
            a = best["alpha"] if a is None else a
            print(f"using (T, alpha) = ({T}, {a}) selected on dev in Task 2")
        else:
            T = 2.0 if T is None else T
            a = 0.5 if a is None else a
            print(f"results/task2/best.json not found; using defaults (T, alpha) = ({T}, {a})")
    return float(T), float(a)


def main(argv=None):
    args = base_parser("Task 3: recursive learning + distillation").parse_args(argv)
    cfg, base = setup(args)
    seeds = args.seeds if args.seeds is not None else cfg["task3"]["seeds"]
    out = ensure_dir(ROOT / (args.out or "results/task3"))
    T, alpha = _resolve_T_alpha(cfg)
    Xt, Xs = base.X["teacher"], base.X["student"]
    y_dev, y_te = base.y["dev"], base.y["test"]
    n_train = Xt["train"].shape[0]

    rows = []
    for seed in seeds:
        split = make_split(base, seed, cfg["data"]["seed_per_class"])
        res = run_selftrain(base, split, cfg, seed)
        transfer_idx = np.concatenate([split.seed_idx, split.pool_idx])
        y_transfer = np.full(len(transfer_idx), -1, dtype=np.int64)
        y_transfer[: len(split.seed_idx)] = base.y["train"][split.seed_idx]

        def report(name, model, role):
            X = Xt if role == "teacher" else Xs
            rep = full_report(model, X["dev"], y_dev, X["test"], y_te, cfg)
            rows.append({"seed": seed, "model": name, "T": T if "distilled" in name or "from" in name else np.nan,
                         "alpha": alpha if "distilled" in name or "from" in name else np.nan, **rep})

        def distill(teacher, s):
            logits = teacher.logits(Xt["train"][transfer_idx])
            return train_model("student", Xs["train"][transfer_idx], y_transfer, cfg, s,
                               teacher_logits=logits, T=T, alpha=alpha)

        # teachers
        full_teacher = train_model("teacher", Xt["train"], base.y["train"], cfg, seed)
        report("teacher_round0", res.teacher_round0, "teacher")
        report("teacher_final_loop", res.teacher_final, "teacher")
        report("teacher_full_labels(ref)", full_teacher, "teacher")

        # students
        report("student_seed_hard", train_model("student", Xs["train"][split.seed_idx], base.y["train"][split.seed_idx], cfg, seed), "student")
        report("student_from_round0", distill(res.teacher_round0, seed), "student")
        report("student_from_final", distill(res.teacher_final, seed), "student")
        hp_idx = np.concatenate([split.seed_idx, res.accepted_idx]).astype(np.int64)
        hp_y = np.concatenate([base.y["train"][split.seed_idx], res.accepted_y]).astype(np.int64)
        report("student_hard_pseudo", train_model("student", Xs["train"][hp_idx], hp_y, cfg, seed), "student")
        ft_logits = full_teacher.logits(Xt["train"])
        report("student_task2_full_labels(ref)",
               train_model("student", Xs["train"], base.y["train"], cfg, seed, teacher_logits=ft_logits, T=T, alpha=alpha), "student")
        rows[-1]["T"], rows[-1]["alpha"] = T, alpha
        print(f"  [seed {seed}] done: teacher F1 r0->final "
              f"{[r for r in rows if r['seed']==seed and r['model']=='teacher_round0'][0]['test_macro_f1']:.4f}->"
              f"{[r for r in rows if r['seed']==seed and r['model']=='teacher_final_loop'][0]['test_macro_f1']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "all_runs.csv", index=False)
    summary = df.groupby("model")[["dev_macro_f1", "test_macro_f1", "test_accuracy", "test_ece", "params", "size_kb", "latency_ms"]].agg(["mean", "std"]).round(4)
    summary.to_csv(out / "summary.csv")

    # does the loop's gain survive distillation?  paired differences per seed (final - round0)
    piv = df.pivot(index="seed", columns="model", values="test_macro_f1")
    piv_dev = df.pivot(index="seed", columns="model", values="dev_macro_f1")
    gain = pd.DataFrame({
        "teacher_gain_test": piv["teacher_final_loop"] - piv["teacher_round0"],
        "student_gain_test": piv["student_from_final"] - piv["student_from_round0"],
        "teacher_gain_dev": piv_dev["teacher_final_loop"] - piv_dev["teacher_round0"],
        "student_gain_dev": piv_dev["student_from_final"] - piv_dev["student_from_round0"]})
    gain.to_csv(out / "gain_survival.csv")
    save_json({"T": T, "alpha": alpha, "seeds": seeds}, out / "run_config.json")
    print("\n=== Task 3: TEST results (mean/std over seeds) ===")
    print(summary[["test_macro_f1", "test_ece", "params", "latency_ms"]].to_string())
    print("\n=== Does the loop's gain survive distillation? (final - round0, macro-F1) ===")
    print(gain.round(4).to_string())
    print("mean:", gain.mean().round(4).to_dict())
    return df


if __name__ == "__main__":
    main()
