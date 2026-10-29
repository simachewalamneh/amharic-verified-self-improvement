"""Research-extension experiment suites (resumable; one JSON line per run).

python -m paper.research.suite --suite main      # MLP teacher, 20 labels/class, 10 seeds, selection-rule ablation
python -m paper.research.suite --suite charlr    # stronger char-LR teacher, calibrated, verifier that can disagree
python -m paper.research.suite --suite budget    # label budgets 5/10/40 per class
python -m paper.research.suite --suite threshold # confidence thresholds 0.5/0.7 (0.9 is in 'main')
python -m paper.research.suite --suite all
Add --smoke --set data.source=synthetic for a seconds-long wiring test (never report synthetic numbers).
Outputs: results/research/<suite>_runs.jsonl, <suite>_rounds.jsonl, preds/<suite>__<variant>__<budget>__s<seed>.npz
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np

from src.data import make_split
from src.eval import ece, macro_f1
from src.pipeline import base_parser, setup
from src.utils import ROOT, ensure_dir
from .loop import LoopSpec, run_loop

M = LoopSpec  # shorthand

SUITES = {
    "main": dict(budgets=[20], seeds=list(range(10)), variants={
        "seed_only": M(rounds=0),
        "naive": M(rollback=False),
        "verified": M(agreement=True, rollback=True),
        "verified_margin": M(agreement=True, rollback=True, margin=0.01),
        "cap": M(selection="cap", rollback=False),
        "balanced": M(selection="balanced", rollback=False),
        "balanced_rb": M(selection="balanced", rollback=True),
    }),
    "charlr": dict(budgets=[20], seeds=list(range(10)), variants={
        "seed_only": M(teacher="charlr", calibrate=True, rounds=0),
        "naive": M(teacher="charlr", calibrate=True, rollback=False),
        "conf_rb": M(teacher="charlr", calibrate=True, rollback=True),
        "verified": M(teacher="charlr", calibrate=True, agreement=True, rollback=True),
        "verified_thr0.7": M(teacher="charlr", calibrate=True, agreement=True, rollback=True, threshold=0.7),
        "verified_uncal": M(teacher="charlr", calibrate=False, agreement=True, rollback=True),
        "balanced_rb": M(teacher="charlr", calibrate=True, selection="balanced", rollback=True),
    }),
    "budget": dict(budgets=[5, 10, 40], seeds=list(range(5)), variants={
        "seed_only": M(rounds=0),
        "naive": M(rollback=False),
        "verified": M(agreement=True, rollback=True),
        "balanced_rb": M(selection="balanced", rollback=True),
    }),
    "threshold": dict(budgets=[20], seeds=list(range(5)), variants={
        "naive_thr0.5": M(rollback=False, threshold=0.5),
        "naive_thr0.7": M(rollback=False, threshold=0.7),
        "verified_thr0.5": M(agreement=True, rollback=True, threshold=0.5),
        "verified_thr0.7": M(agreement=True, rollback=True, threshold=0.7),
    }),
}


def _done(path):
    seen = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            seen.add((r["variant"], r["budget"], r["seed"]))
    return seen


def _scores(f, split, y, nb):
    p = f.proba(split)
    pred = p.argmax(1)
    return macro_f1(y, pred), float((pred == y).mean()), ece(p, y, nb), pred


def run_suite(name: str, cfg: dict, base, seeds=None, smoke=False, log=print):
    spec = SUITES[name]
    out = ensure_dir(ROOT / "results" / "research")
    pdir = ensure_dir(out / "preds")
    runs_p, rounds_p = out / f"{name}_runs.jsonl", out / f"{name}_rounds.jsonl"
    seen = _done(runs_p)
    nb = cfg["eval"]["ece_bins"]
    for budget in spec["budgets"]:
        for seed in (seeds if seeds is not None else spec["seeds"]):
            split = make_split(base, seed, budget)
            for vname, vs in spec["variants"].items():
                if smoke:
                    vs = replace(vs, rounds=min(vs.rounds, 2))
                key = (vname, budget, seed)
                if key in seen:
                    continue
                t0 = time.time()
                res = run_loop(base, split, cfg, seed, vs)
                d0, a0, e0, p0d = _scores(res.round0, "dev", base.y["dev"], nb)
                t0f, ta0, te0, p0t = _scores(res.round0, "test", base.y["test"], nb)
                d1, a1, e1, p1d = _scores(res.final, "dev", base.y["dev"], nb)
                t1f, ta1, te1, p1t = _scores(res.final, "test", base.y["test"], nb)
                h = res.history
                rounds = [r for r in h if r["round"] > 0]
                last = rounds[-1] if rounds else {}
                names = base.label_names
                row = {"suite": name, "variant": vname, "budget": budget, "seed": seed, "spec": vs.tag(),
                       "dev_f1_r0": d0, "test_f1_r0": t0f, "test_acc_r0": ta0, "test_ece_r0": te0,
                       "dev_f1_final": d1, "test_f1_final": t1f, "test_acc_final": ta1, "test_ece_final": te1,
                       "n_rounds": len(rounds), "n_accepted": int(len(res.accepted_idx)),
                       "final_pl_precision": float((res.accepted_y == base.y["train"][res.accepted_idx]).mean()) if len(res.accepted_idx) else None,
                       "class_entropy": last.get("class_entropy"),
                       "sum_selected": int(sum(r["n_selected"] for r in rounds)),
                       "sum_disagree": int(sum(r["n_disagree"] for r in rounds)),
                       "sum_teacher_correct_on_disagree": int(round(sum((r["teacher_prec_on_disagree"] or 0) * r["n_disagree"] for r in rounds if r["n_disagree"] and r["teacher_prec_on_disagree"] == r["teacher_prec_on_disagree"]))),
                       "sum_verifier_correct_on_disagree": int(round(sum((r["verifier_prec_on_disagree"] or 0) * r["n_disagree"] for r in rounds if r["n_disagree"] and r["verifier_prec_on_disagree"] == r["verifier_prec_on_disagree"]))),
                       "seconds": round(time.time() - t0, 1)}
                for c in range(base.n_classes):
                    row[f"acc_{names[c]}"] = last.get(f"acc_{names[c]}", 0)
                    row[f"rate_{names[c]}"] = last.get(f"rate_{names[c]}", 0.0)
                with open(runs_p, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                with open(rounds_p, "a", encoding="utf-8") as f:
                    for r in h:
                        f.write(json.dumps({"suite": name, "variant": vname, "budget": budget, "seed": seed, **r}) + "\n")
                np.savez_compressed(pdir / f"{name}__{vname}__{budget}__s{seed}.npz", dev_r0=p0d, test_r0=p0t, dev_final=p1d, test_final=p1t)
                log(f"[{name}] budget={budget} seed={seed} {vname}: test F1 {t0f:.4f}->{t1f:.4f} accepted={row['n_accepted']} ({row['seconds']}s)")
    return runs_p


def main(argv=None):
    p = base_parser("Research-extension suites")
    p.add_argument("--suite", default="main", choices=list(SUITES) + ["all"])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    cfg, base = setup(args)
    for name in (list(SUITES) if args.suite == "all" else [args.suite]):
        run_suite(name, cfg, base, seeds=args.seeds, smoke=args.smoke)


if __name__ == "__main__":
    main()
