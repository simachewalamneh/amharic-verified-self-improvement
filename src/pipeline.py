"""Shared CLI setup used by every run.py."""
from __future__ import annotations

import argparse

import yaml

from .data import Base, load_base
from .features import attach_views
from .utils import load_config


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="extra YAML merged over configs/default.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="dotted overrides, e.g. data.source=synthetic task1.rounds=2")
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--out", default=None, help="results directory")
    return p


def parse_overrides(pairs) -> dict:
    out = {}
    for kv in pairs:
        k, v = kv.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


def setup(args) -> tuple[dict, Base]:
    cfg = load_config(args.config, parse_overrides(args.set))
    base = attach_views(load_base(cfg), cfg)
    cfg["_n_classes"] = base.n_classes
    print(f"data source={cfg['data']['source']} backend={cfg['features']['backend']} "
          f"classes={base.label_names} sizes={ {k: len(v) for k, v in base.texts.items()} }")
    return cfg, base
