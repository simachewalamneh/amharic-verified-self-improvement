"""Feature views.

Three views of the same text, so that the verifier is genuinely independent of the teacher:
  teacher  : word 1-2gram TF-IDF (or AfroXLMR mean-pooled embeddings)
  student  : smaller word unigram TF-IDF vocabulary (or the same embeddings)
  verifier : character n-gram TF-IDF, used only by the independent logistic-regression verifier

Vectorisers are fitted on the text of the full train split (no labels used), then applied to dev/test.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Base
from .utils import ROOT, ensure_dir

TOKEN = r"(?u)\w+"


def _tfidf_views(base: Base, f: dict) -> dict:
    tr = base.texts["train"]
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                           max_features=f["teacher_max_features"], token_pattern=TOKEN, dtype=np.float32)
    small = TfidfVectorizer(analyzer="word", ngram_range=(1, 1), min_df=2, sublinear_tf=True,
                            max_features=f["student_max_features"], token_pattern=TOKEN, dtype=np.float32)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True,
                           max_features=f["verifier_max_features"], dtype=np.float32)
    views = {}
    for name, vec in (("teacher", word), ("student", small), ("verifier", char)):
        vec.fit(tr)
        views[name] = {s: vec.transform(base.texts[s]).astype(np.float32).tocsr() for s in base.texts}
    return views


def _embed(texts, name: str, max_len: int, cache_key: str) -> np.ndarray:
    """Mean-pooled encoder embeddings, cached on disk. Needs `transformers` and HF access."""
    cache = ensure_dir(ROOT / "data" / "cache") / f"{cache_key}.npy"
    if cache.exists():
        return np.load(cache)
    import torch
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
    arr = np.concatenate(out).astype(np.float32)
    np.save(cache, arr)
    return arr


def _embedding_views(base: Base, f: dict) -> dict:
    tag = f["afroxlmr_name"].replace("/", "_")
    emb = {s: _embed(base.texts[s], f["afroxlmr_name"], f["afroxlmr_max_len"], f"{tag}_{s}") for s in base.texts}
    mu, sd = emb["train"].mean(0), emb["train"].std(0) + 1e-6
    emb = {s: ((v - mu) / sd).astype(np.float32) for s, v in emb.items()}
    views = _tfidf_views(base, f)          # reuse for the independent char-n-gram verifier
    views["teacher"] = emb
    views["student"] = emb
    return views


def attach_views(base: Base, cfg: dict) -> Base:
    f = cfg["features"]
    if f["backend"] == "tfidf":
        base.X = _tfidf_views(base, f)
    elif f["backend"] == "afroxlmr":
        base.X = _embedding_views(base, f)
    else:
        raise ValueError(f"unknown features.backend: {f['backend']}")
    return base
