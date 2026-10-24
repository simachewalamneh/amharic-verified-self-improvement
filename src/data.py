"""Data loading and splitting.

Splits used throughout the repo:
  train  = official train split (labels exist, but are treated as HIDDEN for the unlabeled pool)
  seed   = small stratified labelled subset of train (Tasks 1 and 3)
  pool   = the rest of train, treated as unlabeled; gold labels used ONLY for measurement
  dev    = official dev split, used for accept/rollback decisions and model selection
  test   = official test split, touched only for final reporting
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ROOT


@dataclass
class Base:
    label_names: list
    texts: dict          # split -> list[str]
    y: dict              # split -> np.ndarray (gold)
    X: dict = field(default_factory=dict)   # view -> split -> matrix

    @property
    def n_classes(self) -> int:
        return len(self.label_names)


@dataclass
class Split:
    seed_idx: np.ndarray
    pool_idx: np.ndarray


def _read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_MINIMAL, dtype=str).fillna("")
    return df


def _build_text(df: pd.DataFrame, max_chars: int) -> list:
    return [(f"{h} . {t}").strip()[:max_chars] for h, t in zip(df["headline"], df["text"])]


def _load_local(cfg: dict):
    raw = ROOT / cfg["data"]["raw_dir"]
    frames = {}
    for split in ("train", "dev", "test"):
        p = raw / f"{split}.tsv"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run: python data/download.py")
        frames[split] = _read_tsv(p)
    return frames


def _load_hf(cfg: dict):
    """Optional: Hugging Face loader (not exercised in the development sandbox)."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("masakhane/masakhanews", "amh")
    names = ds["train"].features["label"].names
    frames = {}
    for split in ("train", "validation", "test"):
        d = ds[split].to_pandas()
        d["category"] = [names[i] for i in d["label"]]
        frames["dev" if split == "validation" else split] = d[["category", "headline", "text"]]
    return frames


def _make_synthetic(seed: int = 0, n_classes: int = 4):
    """Synthetic Ge'ez-script-like data for smoke tests ONLY. Never report results on it."""
    rng = np.random.default_rng(seed)

    def token():
        return "".join(chr(0x1200 + int(rng.integers(0, 300))) for _ in range(int(rng.integers(2, 5))))

    shared = [token() for _ in range(600)]
    topical = [[token() for _ in range(120)] for _ in range(n_classes)]
    names = [f"class{c}" for c in range(n_classes)]

    def make(n):
        rows = []
        for i in range(n):
            c = i % n_classes
            words = [
                topical[c][int(rng.integers(0, 120))] if rng.random() < 0.18 else shared[int(rng.integers(0, 600))]
                for _ in range(80)
            ]
            rows.append({"category": names[c], "headline": " ".join(words[:8]), "text": " ".join(words[8:])})
        return pd.DataFrame(rows)

    return {"train": make(800), "dev": make(160), "test": make(240)}


def load_base(cfg: dict) -> Base:
    source = cfg["data"]["source"]
    if source == "local":
        frames = _load_local(cfg)
    elif source == "hf":
        frames = _load_hf(cfg)
    elif source == "synthetic":
        frames = _make_synthetic()
    else:
        raise ValueError(f"unknown data.source: {source}")

    label_names = sorted(frames["train"]["category"].unique())
    to_id = {n: i for i, n in enumerate(label_names)}
    texts, y = {}, {}
    for split, df in frames.items():
        texts[split] = _build_text(df, cfg["data"]["max_chars"])
        y[split] = np.array([to_id[c] for c in df["category"]], dtype=np.int64)
    return Base(label_names=label_names, texts=texts, y=y)


def make_split(base: Base, seed: int, seed_per_class: int) -> Split:
    """Stratified seed set; everything else in train is the unlabeled pool."""
    rng = np.random.default_rng(seed)
    y = base.y["train"]
    seed_idx = []
    for c in range(base.n_classes):
        idx = np.where(y == c)[0]
        seed_idx.extend(rng.choice(idx, size=min(seed_per_class, len(idx)), replace=False).tolist())
    seed_idx = np.array(sorted(seed_idx), dtype=np.int64)
    pool_idx = np.setdiff1d(np.arange(len(y)), seed_idx).astype(np.int64)
    assert len(np.intersect1d(seed_idx, pool_idx)) == 0
    return Split(seed_idx=seed_idx, pool_idx=pool_idx)
