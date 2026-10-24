"""Task 1 verifier: decides which pseudo-labels are allowed into the next training round.

A pseudo-label is accepted only if
  (1) the teacher's confidence >= conf_threshold, and
  (2) (optional) an INDEPENDENT model (character n-gram TF-IDF + logistic regression, fitted on the seed set only)
      predicts the same class.
The verifier is a proxy, not ground truth. Gold labels of the pool are never used here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Verifier:
    conf_threshold: float = 0.9
    require_agreement: bool = True
    max_per_class_per_round: int | None = None

    def accept(self, teacher_proba: np.ndarray, indep_proba: np.ndarray | None) -> np.ndarray:
        conf = teacher_proba.max(1)
        pred = teacher_proba.argmax(1)
        mask = conf >= self.conf_threshold
        if self.require_agreement and indep_proba is not None:
            mask &= indep_proba.argmax(1) == pred
        if self.max_per_class_per_round is not None:
            for c in np.unique(pred):
                cand = np.where(mask & (pred == c))[0]
                if len(cand) > self.max_per_class_per_round:
                    keep = cand[np.argsort(-conf[cand])[: self.max_per_class_per_round]]
                    drop = np.setdiff1d(cand, keep)
                    mask[drop] = False
        return mask
