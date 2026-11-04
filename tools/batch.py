"""No-render batch harness: run many seeds headlessly and aggregate the results.

Nothing is rendered, so a 1500-frame episode takes a fraction of a second and
no OpenGL context is needed (CI runs this with MUJOCO_GL=disable).

    python -m tools.batch                                  # seeds 0-49, 1500 frames
    python -m tools.batch --seeds 0-9 --frames 3000
    python -m tools.batch --set forward_model.residual_threshold_px=6
    python -m tools.batch --json results.json --md summary.md
    python -m tools.batch --standing-still                 # robot never strikes

"--set" takes any dotted config key and may be repeated; a key that does not
exist in config.yaml is an error rather than a silently ignored typo.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Sequence

from config import apply_overrides, load_config
from simulation import run_headless


def parse_seeds(spec: str) -> list[int]:
    """'0-49' -> 0..49, '1,4,9' -> [1, 4, 9], '3-5,10' -> [3, 4, 5, 10]."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if sep:
            a, b = int(lo), int(hi)
            if b < a:
                raise ValueError(f"empty seed range {part!r}")
            seeds.extend(range(a, b + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError(f"no seeds in {spec!r}")
    return seeds


def run_batch(cfg: dict, seeds: Sequence[int], frames: int,
              allow_strikes: bool = True) -> dict:
    per_run = [run_headless(cfg, s, frames, allow_strikes=allow_strikes).to_dict()
               for s in seeds]
    return {
        "frames": frames,
        "seeds": list(seeds),
        "allow_strikes": allow_strikes,
        "dt_s": cfg["sim"]["dt_ms"] / 1000.0,
        "per_run": per_run,
        "aggregate": aggregate(per_run, frames * cfg["sim"]["dt_ms"] / 1000.0),
    }


def aggregate(per_run: list[dict], run_seconds: float) -> dict:
    strikes = sum(r["strikes"] for r in per_run)
    hits = sum(r["hits"] for r in per_run)
    misses = sum(r["misses"] for r in per_run)
    frames = sum(r["frames"] for r in per_run)
    seen = sum(r["detection_frames"] for r in per_run)
    records = [rec for r in per_run for rec in r["strike_records"]]
    miss_recs = [rec for rec in records if not rec["hit"]]
    hit_recs = [rec for rec in records if rec["hit"]]
    medians = [r["residual_px"]["median"] for r in per_run if r["residual_px"]["count"]]

    def med(values):
        return statistics.median(values) if values else math.nan

    return {
        "runs": len(per_run),
        "strikes": strikes,
        "hits": hits,
        "misses": misses,
        "unfinished": strikes - hits - misses,  # still charging when the run ended
        "hit_rate": hits / (hits + misses) if hits + misses else math.nan,
        "strikes_per_min": strikes / (len(per_run) * run_seconds / 60.0) if per_run else math.nan,
        "misses_beyond_reach": sum(1 for rec in miss_recs if rec["beyond_reach"]),
        "hits_beyond_reach": sum(1 for rec in hit_recs if rec["beyond_reach"]),
        "median_start_dist_hit_m": med([rec["start_dist_m"] for rec in hit_recs]),
        "median_start_dist_miss_m": med([rec["start_dist_m"] for rec in miss_recs]),
        "median_time_to_contact_s": med([rec["duration_s"] for rec in hit_recs]),
        "out_of_view_frac": 1.0 - seen / frames if frames else math.nan,
        "median_of_run_median_residual_px": med(medians),
    }


def format_table(result: dict) -> str:
    lines = [f"{'seed':>5} {'strikes':>7} {'hits':>5} {'misses':>6} "
             f"{'res_med_px':>10} {'out_of_view':>11}"]
    for r in result["per_run"]:
        res = r["residual_px"]
        med = f"{res['median']:.2f}" if res["count"] else "-"
        lines.append(f"{r['seed']:>5} {r['strikes']:>7} {r['hits']:>5} {r['misses']:>6} "
                     f"{med:>10} {r['out_of_view_frac']:>10.1%}")
    return "\n".join(lines)


def format_summary(result: dict) -> str:
    a = result["aggregate"]
    misses = a["misses"]
    beyond = (f"{a['misses_beyond_reach']}/{misses} "
              f"({a['misses_beyond_reach'] / misses:.0%})" if misses else "0/0")
    return "\n".join([
        f"runs: {a['runs']} x {result['frames']} frames"
        + ("" if result["allow_strikes"] else " (robot standing still)"),
        f"strikes: {a['strikes']}  hits: {a['hits']}  misses: {misses}  "
        f"unfinished: {a['unfinished']}  hit rate: {a['hit_rate']:.1%}  "
        f"strikes/min: {a['strikes_per_min']:.2f}",
        f"misses that started beyond reach: {beyond}  "
        f"(hits that did: {a['hits_beyond_reach']})",
        f"median start distance: hit {a['median_start_dist_hit_m']:.2f} m, "
        f"miss {a['median_start_dist_miss_m']:.2f} m  "
        f"(hits make contact after {a['median_time_to_contact_s']:.2f} s median)",
        f"ball out of camera view: {a['out_of_view_frac']:.1%} of frames",
        f"median of per-run median residual: {a['median_of_run_median_residual_px']:.2f} px",
    ])


def format_markdown(result: dict, overrides: Sequence[str]) -> str:
    a = result["aggregate"]
    rows = [
        ("runs x frames", f"{a['runs']} x {result['frames']}"),
        ("strikes / hits / misses / unfinished",
         f"{a['strikes']} / {a['hits']} / {a['misses']} / {a['unfinished']}"),
        ("hit rate", f"{a['hit_rate']:.1%}"),
        ("strikes per sim-minute", f"{a['strikes_per_min']:.2f}"),
        ("misses that started beyond reach", f"{a['misses_beyond_reach']}"),
        ("ball out of camera view", f"{a['out_of_view_frac']:.1%}"),
        ("median of per-run median residual", f"{a['median_of_run_median_residual_px']:.2f} px"),
    ]
    head = "### No-render batch harness"
    if overrides:
        head += " (" + ", ".join(f"`{o}`" for o in overrides) + ")"
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return f"{head}\n\n| metric | value |\n|---|---|\n{body}\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seeds", default="0-49", help="e.g. 0-49, 1,4,9 (default 0-49)")
    ap.add_argument("--frames", type=int, default=1500, help="frames per run (default 1500)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="override a config value, e.g. "
                    "forward_model.residual_threshold_px=6 (repeatable)")
    ap.add_argument("--config", default=None, help="config file (default: repo config.yaml)")
    ap.add_argument("--standing-still", action="store_true",
                    help="never strike: measure the tracker with a stationary robot")
    ap.add_argument("--json", type=Path, default=None, help="write full results here")
    ap.add_argument("--md", type=Path, default=None,
                    help="append a markdown summary here (CI: $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--quiet", action="store_true", help="skip the per-seed table")
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    cfg = apply_overrides(cfg, args.overrides)
    result = run_batch(cfg, parse_seeds(args.seeds), args.frames,
                       allow_strikes=not args.standing_still)
    result["overrides"] = list(args.overrides)

    if not args.quiet:
        print(format_table(result))
        print()
    print(format_summary(result))

    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    if args.md:
        with open(args.md, "a", encoding="utf-8") as f:
            f.write(format_markdown(result, args.overrides))
    return 0


if __name__ == "__main__":
    sys.exit(main())
