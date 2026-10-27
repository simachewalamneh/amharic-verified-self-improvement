"""End-to-end single-article latency (vectorisation + forward pass), one CPU thread.

python -m research.latency_e2e   -> results/research/latency_e2e.json
Teacher = word 1-2gram TF-IDF (10k) + MLP; student = word unigram TF-IDF (2k) + linear.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data import load_base
from src.features import TOKEN
from src.models import rows_to_tensor, train_model
from src.utils import ROOT, ensure_dir, load_config


def main():
    cfg = load_config()
    base = load_base(cfg)
    f = cfg["features"]
    cfg["_n_classes"] = base.n_classes
    tr, te = base.texts["train"], base.texts["test"]
    vt = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=f["teacher_max_features"], token_pattern=TOKEN, dtype=np.float32).fit(tr)
    vs = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=2, sublinear_tf=True, max_features=f["student_max_features"], token_pattern=TOKEN, dtype=np.float32).fit(tr)
    teacher = train_model("teacher", vt.transform(tr), base.y["train"], cfg, 0)
    student = train_model("student", vs.transform(tr), base.y["train"], cfg, 0)
    torch.set_num_threads(1)
    out = {}
    for name, vec, model in (("teacher", vt, teacher), ("student", vs, student)):
        model.net.eval()
        vec_t, fwd_t, tot_t = [], [], []
        with torch.no_grad():
            for t in te[:20]:
                model.net(rows_to_tensor(vec.transform([t]), np.array([0])))
            for t in te[:200]:
                a = time.perf_counter()
                X = vec.transform([t])
                b = time.perf_counter()
                model.net(rows_to_tensor(X, np.array([0])))
                c = time.perf_counter()
                vec_t.append((b - a) * 1000); fwd_t.append((c - b) * 1000); tot_t.append((c - a) * 1000)
        out[name] = {"vectorise_ms": float(np.median(vec_t)), "forward_ms": float(np.median(fwd_t)), "end_to_end_ms": float(np.median(tot_t))}
    out["end_to_end_speedup"] = out["teacher"]["end_to_end_ms"] / out["student"]["end_to_end_ms"]
    json.dump(out, open(ensure_dir(ROOT / "results" / "research") / "latency_e2e.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
