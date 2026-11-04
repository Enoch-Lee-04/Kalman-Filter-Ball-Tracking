"""Configuration loading and dotted-key overrides.

The default path is resolved relative to this file, so tests, tools and the
demo all load the same config.yaml no matter which directory they run from.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Return a copy of cfg with "section.key=value" overrides applied.

    Values are parsed as YAML ("6" -> int, "0.5" -> float, "[0, -3]" -> list).
    The key path must already exist, so a typo fails instead of silently
    running the default configuration.
    """
    out = copy.deepcopy(cfg)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"override must look like section.key=value, got {item!r}")
        parts = key.strip().split(".")
        node = out
        for depth, part in enumerate(parts):
            if not isinstance(node, dict) or part not in node:
                where = ".".join(parts[:depth]) or "<root>"
                raise KeyError(f"unknown config key {key!r}: no {part!r} under {where}")
            if depth == len(parts) - 1:
                node[part] = yaml.safe_load(raw)
            else:
                node = node[part]
    return out
