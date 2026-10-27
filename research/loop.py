"""Generalised self-training loop for the research extension (Tasks 1-3 are NOT modified).

Adds, relative to task1_recursive.selftrain.run_selftrain:
  * selection rules: 'threshold' (global confidence), 'cap' (threshold + per-class cap), 'balanced' (per-class top-k)
  * teacher kinds: 'mlp' (word TF-IDF MLP, as in Task 1) or 'charlr' (char n-gram logistic regression)
  * optional temperature scaling of teacher confidence on the dev set
  * rollback with an acceptance margin
  * per-round verifier diagnostics (how often the cross-representation check disagrees, and who is right)
  * per-class accepted counts and per-class acceptance rates

With teacher='mlp', selection='threshold', agreement=True, rollback=True, margin=0 it reproduces Task 1's
'verified' variant exactly (same seeds, same fits).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from src.eval import ece, macro_f1
from src.models import CharLR, train_model


@dataclass(frozen=True)
class LoopSpec:
    teacher: str = "mlp"             # mlp | charlr
    selection: str = "threshold"     # threshold | cap | balanced
    threshold: float = 0.9
    agreement: bool = False          # require cross-representation agreement
    rollback: bool = True
    margin: float = 0.0              # required dev macro-F1 gain to accept a round
    calibrate: bool = False          # temperature-scale teacher confidence on dev
    rounds: int = 5
    per_class_k: int = 10            # 'cap' and 'balanced'
    floor: float = 0.5               # minimum confidence for 'balanced'
    stop_after_rejects: int = 2

    def tag(self) -> str:
        return (f"{self.teacher}|{self.selection}|thr{self.threshold}|agree{int(self.agreement)}|rb{int(self.rollback)}"
                f"|m{self.margin}|cal{int(self.calibrate)}")


@dataclass
class Fitted:
    """A trained teacher plus its views and (optional) temperature."""
    model: object
    kind: str
    views: dict                      # split -> matrix for the teacher's own view
    temperature: float = 1.0

    def logits(self, split: str, idx=None) -> np.ndarray:
        X = self.views[split]
        X = X if idx is None else X[idx]
        if self.kind == "mlp":
            return self.model.logits(X)
        return np.log(np.clip(self.model.proba(X), 1e-9, 1.0))

    def proba(self, split: str, idx=None) -> np.ndarray:
        z = self.logits(split, idx) / self.temperature
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Scalar temperature minimising dev NLL (grid search; argmax is unchanged by temperature)."""
    best_t, best = 1.0, np.inf
    for t in np.concatenate([np.linspace(0.05, 1.0, 39), np.linspace(1.05, 5.0, 40)]):
        z = logits / t
        z = z - z.max(1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(1, keepdims=True))
        nll = -lp[np.arange(len(y)), y].mean()
        if nll < best:
            best, best_t = nll, float(t)
    return best_t


def _views(base, kind: str):
    """(teacher view, verifier view) matrices; the roles swap for the char-LR teacher."""
    if kind == "mlp":
        return base.X["teacher"], base.X["verifier"]
    return base.X["verifier"], base.X["teacher"]


def _fit(base, cfg, spec: LoopSpec, idx, y, seed) -> Fitted:
    tv, _ = _views(base, spec.teacher)
    if spec.teacher == "mlp":
        m = train_model("teacher", tv["train"][idx], y, cfg, seed)
    else:
        m = CharLR(cfg["models"]["verifier_lr_C"] * 10, base.n_classes).fit(tv["train"][idx], y)
    f = Fitted(m, spec.teacher, tv)
    if spec.calibrate:
        f.temperature = fit_temperature(f.logits("dev"), base.y["dev"])
    return f


def _entropy(counts: np.ndarray) -> float:
    s = counts.sum()
    if s <= 0:
        return float("nan")
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _select(spec: LoopSpec, conf, pred, nc) -> np.ndarray:
    if spec.selection in ("threshold", "cap"):
        mask = conf >= spec.threshold
        if spec.selection == "cap":
            for c in range(nc):
                cand = np.where(mask & (pred == c))[0]
                if len(cand) > spec.per_class_k:
                    keep = cand[np.argsort(-conf[cand])[: spec.per_class_k]]
                    mask[np.setdiff1d(cand, keep)] = False
        return mask
    if spec.selection == "balanced":
        mask = np.zeros(len(conf), dtype=bool)
        for c in range(nc):
            cand = np.where((pred == c) & (conf >= spec.floor))[0]
            mask[cand[np.argsort(-conf[cand])[: spec.per_class_k]]] = True
        return mask
    raise ValueError(spec.selection)


@dataclass
class LoopResult:
    spec: LoopSpec
    round0: Fitted
    final: Fitted
    history: list = field(default_factory=list)
    accepted_idx: np.ndarray = None
    accepted_y: np.ndarray = None


def run_loop(base, split, cfg: dict, seed: int, spec: LoopSpec, log=None) -> LoopResult:
    nc = base.n_classes
    y_gold = base.y["train"]                  # used ONLY to measure pseudo-label quality on the hidden pool
    names = base.label_names
    seed_idx, pool_idx = split.seed_idx, split.pool_idx
    y_seed = y_gold[seed_idx]
    ydev = base.y["dev"]
    pool_counts = np.bincount(y_gold[pool_idx], minlength=nc)
    _, Xv = _views(base, spec.teacher)
    verifier = CharLR(cfg["models"]["verifier_lr_C"], nc).fit(Xv["train"][seed_idx], y_seed)

    def dev_eval(f: Fitted):
        p = f.proba("dev")
        return macro_f1(ydev, p.argmax(1)), ece(p, ydev, cfg["eval"]["ece_bins"])

    cur = _fit(base, cfg, spec, seed_idx, y_seed, seed)
    r0 = cur
    cur_f1, cur_ece = dev_eval(cur)
    history = [{"round": 0, "decision": "baseline", "dev_macro_f1": cur_f1, "dev_ece": cur_ece, "n_selected": 0,
                "n_accepted_total": 0}]
    acc_idx = np.array([], dtype=np.int64)
    acc_y = np.array([], dtype=np.int64)
    remaining = pool_idx.copy()
    rejects = 0

    for r in range(1, spec.rounds + 1):
        if len(remaining) == 0:
            break
        P = cur.proba("train", remaining)
        conf, pred = P.max(1), P.argmax(1)
        sel = _select(spec, conf, pred, nc)
        vpred = verifier.proba(Xv["train"][remaining]).argmax(1)
        agree = vpred == pred
        n_sel = int(sel.sum())
        dis = sel & ~agree
        gold = y_gold[remaining]
        row = {"round": r, "n_selected": n_sel, "n_disagree": int(dis.sum()),
               "teacher_prec_on_disagree": float((pred[dis] == gold[dis]).mean()) if dis.any() else float("nan"),
               "verifier_prec_on_disagree": float((vpred[dis] == gold[dis]).mean()) if dis.any() else float("nan"),
               "teacher_prec_on_agree": float((pred[sel & agree] == gold[sel & agree]).mean()) if (sel & agree).any() else float("nan")}
        mask = sel & agree if spec.agreement else sel
        cand_idx, cand_y = remaining[mask], pred[mask]
        row["n_candidates"] = int(len(cand_idx))
        row["pl_precision"] = float((cand_y == y_gold[cand_idx]).mean()) if len(cand_idx) else float("nan")

        new_idx = np.concatenate([acc_idx, cand_idx])
        new_y = np.concatenate([acc_y, cand_y])
        prop = _fit(base, cfg, spec, np.concatenate([seed_idx, new_idx]), np.concatenate([y_seed, new_y]), seed * 1000 + r)
        p_f1, p_ece = dev_eval(prop)
        ok = (p_f1 >= cur_f1 + spec.margin) if spec.rollback else True
        row.update({"dev_macro_f1": p_f1, "dev_ece": p_ece, "decision": "accepted" if ok else "rejected"})
        if ok:
            cur, cur_f1, cur_ece = prop, p_f1, p_ece
            acc_idx, acc_y = new_idx, new_y
            remaining = remaining[~mask]
            rejects = 0
        else:
            rejects += 1
        counts = np.bincount(acc_y, minlength=nc) if len(acc_y) else np.zeros(nc, dtype=int)
        row["n_accepted_total"] = int(len(acc_idx))
        row["class_entropy"] = _entropy(counts.astype(float))
        for c in range(nc):
            row[f"acc_{names[c]}"] = int(counts[c])
            row[f"rate_{names[c]}"] = float(counts[c] / pool_counts[c])
        history.append(row)
        if log:
            log(f"    r{r}: sel={n_sel} cand={len(cand_idx)} prec={row['pl_precision']:.3f} dev={p_f1:.4f} {row['decision']}")
        if spec.rollback and rejects >= spec.stop_after_rejects:
            break
    return LoopResult(spec, r0, cur, history, acc_idx, acc_y)
