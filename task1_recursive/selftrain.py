"""Task 1: bounded recursive self-training with external verification, held-out dev gating and rollback.

generate candidates -> verify -> curate -> retrain a COPY -> evaluate on dev -> accept or roll back -> repeat.
Gold labels of the pool are used ONLY to measure pseudo-label precision (logged, never trained on).
The test split is never touched here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.eval import ece, macro_f1
from src.models import CharLR, train_model
from .verifier import Verifier


@dataclass
class SelfTrainResult:
    teacher_round0: object
    teacher_final: object
    history: list = field(default_factory=list)
    accepted_idx: np.ndarray = None   # train indices of accepted pseudo-labelled examples
    accepted_y: np.ndarray = None     # their pseudo-labels


def _entropy(counts: np.ndarray) -> float:
    p = counts / counts.sum() if counts.sum() > 0 else counts
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def run_selftrain(base, split, cfg: dict, seed: int, *, require_agreement=None, use_rollback=None, log=print) -> SelfTrainResult:
    t1 = cfg["task1"]
    require_agreement = t1["require_agreement"] if require_agreement is None else require_agreement
    use_rollback = t1["use_rollback"] if use_rollback is None else use_rollback
    nc = base.n_classes

    Xt, Xv = base.X["teacher"], base.X["verifier"]
    y_gold = base.y["train"]                     # used only for measurement on the hidden pool
    seed_idx, pool_idx = split.seed_idx, split.pool_idx
    y_seed = y_gold[seed_idx]
    Xdev, ydev = Xt["dev"], base.y["dev"]

    def dev_eval(model):
        p = model.proba(Xdev)
        return macro_f1(ydev, p.argmax(1)), ece(p, ydev, cfg["eval"]["ece_bins"])

    # round 0: teacher on the seed set only
    teacher = train_model("teacher", Xt["train"][seed_idx], y_seed, cfg, seed)
    teacher0 = teacher
    cur_f1, cur_ece = dev_eval(teacher)
    indep = CharLR(cfg["models"]["verifier_lr_C"], nc).fit(Xv["train"][seed_idx], y_seed)
    verifier = Verifier(t1["conf_threshold"], require_agreement, t1["max_per_class_per_round"])

    history = [{"round": 0, "decision": "baseline", "pool_remaining": len(pool_idx), "n_candidates": 0,
                "n_accepted_total": 0, "pl_precision": np.nan, "dev_macro_f1": cur_f1, "dev_ece": cur_ece,
                "class_entropy": np.nan}]
    log(f"  [seed {seed}] round 0 baseline dev macro-F1={cur_f1:.4f}")

    accepted_idx = np.array([], dtype=np.int64)
    accepted_y = np.array([], dtype=np.int64)
    remaining = pool_idx.copy()
    rejects_in_a_row = 0

    for r in range(1, t1["rounds"] + 1):
        if len(remaining) == 0:
            break
        tp = teacher.proba(Xt["train"][remaining])
        ip = indep.proba(Xv["train"][remaining]) if require_agreement else None
        mask = verifier.accept(tp, ip)
        cand_idx = remaining[mask]
        cand_y = tp.argmax(1)[mask]
        precision = float((cand_y == y_gold[cand_idx]).mean()) if len(cand_idx) else float("nan")

        new_idx = np.concatenate([accepted_idx, cand_idx])
        new_y = np.concatenate([accepted_y, cand_y])
        train_idx = np.concatenate([seed_idx, new_idx])
        train_y = np.concatenate([y_seed, new_y])
        proposal = train_model("teacher", Xt["train"][train_idx], train_y, cfg, seed * 1000 + r)
        p_f1, p_ece = dev_eval(proposal)

        accept_round = (p_f1 >= cur_f1 + t1["min_delta"]) if use_rollback else True
        counts = np.bincount(new_y, minlength=nc) if len(new_y) else np.zeros(nc)
        row = {"round": r, "pool_remaining": len(remaining), "n_candidates": int(len(cand_idx)),
               "pl_precision": precision, "dev_macro_f1": p_f1, "dev_ece": p_ece,
               "class_entropy": _entropy(counts.astype(float)),
               "decision": "accepted" if accept_round else "rejected"}
        if accept_round:
            teacher, cur_f1, cur_ece = proposal, p_f1, p_ece
            accepted_idx, accepted_y = new_idx, new_y
            remaining = remaining[~mask]
            rejects_in_a_row = 0
        else:
            rejects_in_a_row += 1
        row["n_accepted_total"] = int(len(accepted_idx))
        history.append(row)
        log(f"  [seed {seed}] round {r}: cand={len(cand_idx)} precision={precision:.3f} "
            f"dev F1 {p_f1:.4f} -> {row['decision']} (current {cur_f1:.4f})")
        if use_rollback and rejects_in_a_row >= t1["stop_after_rejects"]:
            log(f"  [seed {seed}] stopping: {rejects_in_a_row} consecutive rejections")
            break

    return SelfTrainResult(teacher0, teacher, history, accepted_idx, accepted_y)
