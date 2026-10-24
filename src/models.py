"""Teacher / student classifiers and the independent verifier model.

Distillation loss (as in the training material, KL direction teacher -> student):
    L = alpha * CE(y, s) + (1 - alpha) * T^2 * KL( softmax(t/T) || softmax(s/T) )
torch's F.kl_div(log q, p) computes KL(p || q), so p = softmax(teacher/T) and q = softmax(student/T).
Examples with label -1 have no hard label: they contribute to the KL term only.
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", message="Sparse invariant checks")


def rows_to_tensor(X, idx) -> torch.Tensor:
    """Rows of X as a torch tensor. Sparse input stays sparse (fast on CPU for TF-IDF)."""
    if sp.issparse(X):
        c = X[idx].tocoo()
        indices = torch.from_numpy(np.vstack([c.row, c.col]).astype(np.int64))
        return torch.sparse_coo_tensor(indices, torch.from_numpy(c.data.astype(np.float32)), size=c.shape).coalesce()
    return torch.from_numpy(np.asarray(X[idx], dtype=np.float32))


class SparseAwareLinear(nn.Module):
    """Linear layer that accepts dense or sparse input (same parameter count as nn.Linear)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        bound = 1.0 / np.sqrt(in_dim)
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        out = torch.sparse.mm(x, self.weight) if x.is_sparse else x @ self.weight
        return out + self.bias


class Classifier:
    def __init__(self, kind: str, in_dim: int, n_classes: int, hidden: int = 256, dropout: float = 0.3, seed: int = 0):
        torch.manual_seed(seed)
        if kind == "mlp":
            self.net = nn.Sequential(SparseAwareLinear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))
        elif kind == "linear":
            self.net = SparseAwareLinear(in_dim, n_classes)
        else:
            raise ValueError(kind)
        self.kind = kind
        self.n_classes = n_classes

    # ---- training -------------------------------------------------------------------------
    @staticmethod
    def _loss(logits, yb, tl, T, alpha):
        valid = yb >= 0
        ce = F.cross_entropy(logits[valid], yb[valid]) if bool(valid.any()) else logits.new_zeros(())
        if tl is None:
            return ce
        kl = F.kl_div(F.log_softmax(logits / T, dim=-1), F.softmax(tl / T, dim=-1), reduction="batchmean")
        return alpha * ce + (1.0 - alpha) * (T * T) * kl

    def fit(self, X, y, *, epochs: int, lr: float, weight_decay: float, batch_size: int, seed: int,
            teacher_logits=None, T: float = 1.0, alpha: float = 1.0):
        n = X.shape[0]
        g = torch.Generator().manual_seed(int(seed))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
        tl_t = None if teacher_logits is None else torch.as_tensor(np.asarray(teacher_logits), dtype=torch.float32)
        for _ in range(epochs):
            self.net.train()
            perm = torch.randperm(n, generator=g).numpy()
            for i in range(0, n, batch_size):
                b = perm[i:i + batch_size]
                logits = self.net(rows_to_tensor(X, b))
                loss = self._loss(logits, y_t[b], None if tl_t is None else tl_t[b], T, alpha)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    # ---- inference ------------------------------------------------------------------------
    @torch.no_grad()
    def logits(self, X, batch_size: int = 512) -> np.ndarray:
        self.net.eval()
        out = [self.net(rows_to_tensor(X, np.arange(i, min(i + batch_size, X.shape[0])))).numpy()
               for i in range(0, X.shape[0], batch_size)]
        return np.concatenate(out) if out else np.zeros((0, self.n_classes), dtype=np.float32)

    def proba(self, X) -> np.ndarray:
        z = self.logits(X)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))


def make_model(role: str, X, n_classes: int, cfg: dict, seed: int) -> Classifier:
    m = cfg["models"][role]
    return Classifier(m["kind"], X.shape[1], n_classes, hidden=m["hidden"], dropout=m["dropout"], seed=seed)


def fit_kwargs(role: str, cfg: dict, seed: int) -> dict:
    m = cfg["models"][role]
    return dict(epochs=m["epochs"], lr=m["lr"], weight_decay=m["weight_decay"], batch_size=m["batch_size"], seed=seed)


def train_model(role: str, X, y, cfg: dict, seed: int, teacher_logits=None, T: float = 1.0, alpha: float = 1.0) -> Classifier:
    """Fresh model of the given role ('teacher' or 'student'), trained on rows of X with labels y (-1 = no label)."""
    model = make_model(role, X, cfg["_n_classes"], cfg, seed)
    return model.fit(X, y, teacher_logits=teacher_logits, T=T, alpha=alpha, **fit_kwargs(role, cfg, seed))


class CharLR:
    """Independent verifier / baseline: logistic regression on character n-gram TF-IDF."""

    def __init__(self, C: float = 10.0, n_classes: int | None = None):
        self.clf = LogisticRegression(C=C, max_iter=2000)
        self.n_classes = n_classes

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def proba(self, X) -> np.ndarray:
        p = self.clf.predict_proba(X)
        if self.n_classes is not None and p.shape[1] != self.n_classes:
            full = np.zeros((p.shape[0], self.n_classes), dtype=p.dtype)
            full[:, self.clf.classes_] = p
            return full
        return p

    @property
    def n_params(self) -> int:
        return int(self.clf.coef_.size + self.clf.intercept_.size)
