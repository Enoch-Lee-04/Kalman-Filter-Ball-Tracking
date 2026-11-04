"""Seed-7 regression check.

The canonical headless run (seed 7, 1500 frames, the repo's config.yaml) must
reproduce the numbers in tests/golden/seed7.json: strikes, hits and misses
exactly, residual statistics to within FLOAT_TOL.

    python -m tools.regression            # check; exit 1 with a diff on mismatch
    python -m tools.regression --record   # re-record after a deliberate change

Rule: a change that is *meant* to alter results re-records the golden file in
the same commit, so the diff shows exactly which numbers moved and by how much.
A change that is not meant to alter results must leave this check passing.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

from config import load_config
from simulation import run_headless

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "tests" / "golden" / "seed7.json"
DEFAULT_SEED = 7
DEFAULT_FRAMES = 1500
# Absolute tolerance (px) for residual statistics. Counts must match exactly.
# This absorbs last-bit libm/BLAS differences between platforms, nothing more:
# any real behaviour change moves these statistics by far more than 1e-6 px.
FLOAT_TOL = 1e-6
COUNT_KEYS = ("strikes", "hits", "misses")


def measure(seed: int = DEFAULT_SEED, frames: int = DEFAULT_FRAMES) -> dict:
    stats = run_headless(load_config(), seed=seed, frames=frames)
    return {
        "seed": seed,
        "frames": frames,
        "strikes": stats.strikes,
        "hits": stats.hits,
        "misses": stats.misses,
        "residual_px": {k: (round(v, 9) if isinstance(v, float) else v)
                        for k, v in stats.residual_summary().items()},
    }


def load_golden(path: Path = GOLDEN_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare(expected: dict, actual: dict, tol: float = FLOAT_TOL) -> list[str]:
    """Human-readable list of differences; empty when the run matches."""
    problems = []
    for key in COUNT_KEYS:
        if expected[key] != actual[key]:
            problems.append(f"{key}: expected {expected[key]}, got {actual[key]}")
    exp_res, act_res = expected["residual_px"], actual["residual_px"]
    for key in sorted(set(exp_res) | set(act_res)):
        e, a = exp_res.get(key), act_res.get(key)
        if e is None or a is None:
            problems.append(f"residual_px.{key}: expected {e}, got {a}")
        elif key == "count":
            if e != a:
                problems.append(f"residual_px.count: expected {e}, got {a}")
        elif not math.isclose(e, a, rel_tol=0.0, abs_tol=tol):
            problems.append(f"residual_px.{key}: expected {e:.9f}, got {a:.9f} "
                            f"(diff {a - e:+.3g})")
    return problems


def record(path: Path = GOLDEN_PATH, seed: int = DEFAULT_SEED,
           frames: int = DEFAULT_FRAMES) -> dict:
    golden = {
        "_comment": ("Expected results of the canonical headless run. Regenerate with "
                     "`python -m tools.regression --record` ONLY when a change is meant "
                     "to alter results, and commit it together with that change."),
        **measure(seed, frames),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden, indent=2) + "\n", encoding="utf-8")
    return golden


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed-7 regression check")
    ap.add_argument("--record", action="store_true",
                    help="overwrite the golden file with the current results")
    args = ap.parse_args(argv)

    if args.record:
        seed, frames = DEFAULT_SEED, DEFAULT_FRAMES
        if GOLDEN_PATH.exists():
            old = load_golden()
            seed, frames = old["seed"], old["frames"]
        golden = record(seed=seed, frames=frames)
        print(f"recorded {GOLDEN_PATH}")
        print(json.dumps({k: v for k, v in golden.items() if k != "_comment"}, indent=2))
        return 0

    expected = load_golden()
    actual = measure(expected["seed"], expected["frames"])
    problems = compare(expected, actual)
    if problems:
        print(f"seed-{expected['seed']} results changed:")
        for p in problems:
            print(f"  {p}")
        print("If this change is intended, run `python -m tools.regression --record` "
              "and commit the golden file with it.")
        return 1
    print(f"seed-{expected['seed']} regression OK "
          f"(strikes {actual['strikes']}, hits {actual['hits']}, misses {actual['misses']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
