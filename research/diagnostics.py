"""Round-0 diagnostics: why are some classes never selected, is the verifier informative, how calibrated is the teacher?

python -m research.diagnostics [--seeds 0 1 ... 9]
Writes results/research/diag_*.csv. Gold labels of the pool are used for MEASUREMENT only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import make_split
from src.models import CharLR
from src.pipeline import base_parser, setup
from src.utils import ROOT, ensure_dir
from .loop import LoopSpec, _fit, _views


def phi(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(float), b.astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main(argv=None):
    p = base_parser("Round-0 diagnostics")
    args = p.parse_args(argv)
    cfg, base = setup(args)
    seeds = args.seeds if args.seeds is not None else list(range(10))
    out = ensure_dir(ROOT / "results" / "research")
    nc, names = base.n_classes, base.label_names
    thr = 0.9
    cls_rows, rel_rows, ver_rows, conf_rows, conf_mats = [], [], [], [], {}
    for tkind, calibrate in (("mlp", False), ("charlr", True)):
        spec = LoopSpec(teacher=tkind, calibrate=calibrate, rounds=0)
        conf_mats[tkind] = np.zeros((nc, nc), dtype=int)
        for seed in seeds:
            split = make_split(base, seed, cfg["data"]["seed_per_class"])
            y_seed = base.y["train"][split.seed_idx]
            t = _fit(base, cfg, spec, split.seed_idx, y_seed, seed)
            _, Xv = _views(base, tkind)
            ver = CharLR(cfg["models"]["verifier_lr_C"], nc).fit(Xv["train"][split.seed_idx], y_seed)
            # ---- pool (hidden gold), per true class ----
            P = t.proba("train", split.pool_idx)
            conf, pred, gold = P.max(1), P.argmax(1), base.y["train"][split.pool_idx]
            for c in range(nc):
                m = gold == c
                above = m & (conf >= thr)
                cls_rows.append({"teacher": tkind, "seed": seed, "class": names[c], "n_pool": int(m.sum()),
                                 "mean_conf": float(conf[m].mean()), "median_conf": float(np.median(conf[m])),
                                 "share_conf_ge_thr": float(above.sum() / m.sum()),
                                 "correct_and_conf": int((above & (pred == c)).sum()),
                                 "confidently_wrong": int((above & (pred != c)).sum()),
                                 "recall": float((pred[m] == c).mean())})
            # ---- test: reliability, confusion, verifier correlation ----
            Pt = t.proba("test")
            ct, pt, yt = Pt.max(1), Pt.argmax(1), base.y["test"]
            for a, b in zip(yt, pt):
                conf_mats[tkind][a, b] += 1
            conf_rows.append({"teacher": tkind, "seed": seed, "mean_conf": float(ct.mean()), "accuracy": float((pt == yt).mean())})
            edges = np.linspace(0, 1, 16)
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (ct > lo) & (ct <= hi)
                rel_rows.append({"teacher": tkind, "seed": seed, "bin_lo": lo, "bin_hi": hi, "n": int(m.sum()),
                                 "sum_conf": float(ct[m].sum()), "n_correct": int((pt[m] == yt[m]).sum())})
            vp = ver.proba(Xv["test"]).argmax(1)
            te, ve = pt != yt, vp != yt
            ver_rows.append({"teacher": tkind, "seed": seed, "agree_rate": float((pt == vp).mean()),
                             "teacher_acc": float((~te).mean()), "verifier_acc": float((~ve).mean()),
                             "error_phi": phi(te, ve), "p_both_wrong": float((te & ve).mean()),
                             "p_verifier_right_given_teacher_wrong": float((~ve[te]).mean()) if te.any() else float("nan"),
                             "p_verifier_wrong_given_teacher_right": float(ve[~te].mean()) if (~te).any() else float("nan")})
    pd.DataFrame(cls_rows).to_csv(out / "diag_per_class.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(out / "diag_reliability.csv", index=False)
    pd.DataFrame(ver_rows).to_csv(out / "diag_verifier.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(out / "diag_confidence.csv", index=False)
    for k, m in conf_mats.items():
        pd.DataFrame(m, index=names, columns=names).to_csv(out / f"diag_confusion_{k}.csv")
    print("diagnostics written to", out)


if __name__ == "__main__":
    main()
