"""Evaluation: macro-F1, accuracy, ECE, parameter count, model size, CPU latency."""
from __future__ import annotations

import io
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .models import Classifier, rows_to_tensor


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro"))


def ece(proba: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = proba.max(1)
    pred = proba.argmax(1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def scores(proba: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    pred = proba.argmax(1)
    return {"macro_f1": macro_f1(y, pred), "accuracy": float(accuracy_score(y, pred)), "ece": ece(proba, y, n_bins)}


def model_size_kb(model: Classifier) -> float:
    buf = io.BytesIO()
    torch.save(model.net.state_dict(), buf)
    return buf.getbuffer().nbytes / 1024.0


def cpu_latency_ms(model: Classifier, X, n_samples: int = 200) -> float:
    """Median single-example forward latency on one CPU thread (features precomputed, forward pass only)."""
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    model.net.eval()
    n = min(n_samples, X.shape[0])
    xs = [rows_to_tensor(X, np.array([i])) for i in range(n)]
    times = []
    with torch.no_grad():
        for x in xs[:10]:
            model.net(x)
        for x in xs:
            t0 = time.perf_counter()
            model.net(x)
            times.append((time.perf_counter() - t0) * 1000.0)
    torch.set_num_threads(prev)
    return float(np.median(times))


def full_report(model: Classifier, X_dev, y_dev, X_test, y_test, cfg: dict, with_efficiency: bool = True) -> dict:
    nb = cfg["eval"]["ece_bins"]
    d = scores(model.proba(X_dev), y_dev, nb)
    t = scores(model.proba(X_test), y_test, nb)
    out = {"dev_macro_f1": d["macro_f1"], "dev_ece": d["ece"],
           "test_macro_f1": t["macro_f1"], "test_accuracy": t["accuracy"], "test_ece": t["ece"]}
    if with_efficiency:
        out.update({"params": model.n_params, "size_kb": model_size_kb(model),
                    "latency_ms": cpu_latency_ms(model, X_test, cfg["eval"]["latency_samples"])})
    return out
