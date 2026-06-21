"""Entry point for the forward-model ball tracking demo.

Usage:
  python main.py                     # live window (press q/ESC to quit)
  python main.py --headless 600      # render 600 frames, save preview PNGs
  python main.py --top               # also show the overhead debug view
"""
from __future__ import annotations

import argparse
import os
import random
import time

import cv2
import numpy as np
import yaml

from controller import TrackingController
from sim.world import SimWorld
from viz import overlay

PREVIEW_DIR = os.path.join("assets", "preview")


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(headless_frames: int | None, show_top: bool, seed: int) -> None:
    cfg = load_config()
    rng = random.Random(seed)

    world = SimWorld(cfg, rng=rng)
    controller = TrackingController(cfg, rng=rng)

    dt = cfg["sim"]["dt_ms"] / 1000.0
    realtime = cfg["sim"]["realtime"] and headless_frames is None
    latency = cfg["latency_ms"]

    # strike animation state
    strike_timer = 0.0
    status = "TRACKING"

    if headless_frames is not None:
        os.makedirs(PREVIEW_DIR, exist_ok=True)

    t = 0.0
    frame_idx = 0
    n = headless_frames if headless_frames is not None else 10 ** 12
    residual_samples: list[float] = []
    strikes = 0

    while frame_idx < n:
        loop_start = time.perf_counter()
        world.step(dt)
        t += dt

        frame = world.render_robot_view()
        result = controller.step(t, dt, frame, world)

        if not np.isnan(result.residual_px):
            residual_samples.append(result.residual_px)

        striking = strike_timer > 0
        if result.confident and not striking:
            status = "STRIKE"
            strike_timer = 0.4  # seconds of strike animation
            strikes += 1

        if striking:
            strike_timer -= dt
            if strike_timer <= 0:
                world.relocate_robot_to_ball()
                world.ball.reset_far_from(*world.robot.position, facing=world.robot.facing)
                controller.reset()
                status = "TRACKING"

        overlay.draw(
            frame,
            match_bbox=result.detection_bbox,
            predicted_center=result.predicted_center,
            predicted_bbox=result.predicted_bbox,
            match_center=result.detection_center,
            residual_px=result.residual_px,
            residual_threshold=cfg["forward_model"]["residual_threshold_px"],
            horizon_ms=controller.horizon_ms,
            latency=latency,
            confidence=result.confidence,
            status=status,
            striking=striking,
        )

        if show_top:
            top = world.render_top_view()
            top = cv2.resize(top, (frame.shape[1] // 3, frame.shape[0] // 3))
            frame[5:5 + top.shape[0], frame.shape[1] - top.shape[1] - 5:frame.shape[1] - 5] = top

        if headless_frames is not None:
            if frame_idx % 30 == 0 or striking:
                tag = "strike_" if striking else "frame_"
                path = os.path.join(PREVIEW_DIR, f"{tag}{frame_idx:05d}.png")
                cv2.imwrite(path, frame)
        else:
            cv2.imshow("Forward Model - Robot View", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if realtime:
                elapsed = time.perf_counter() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

        frame_idx += 1

    if headless_frames is None:
        cv2.destroyAllWindows()
    else:
        print(f"Saved preview frames to {PREVIEW_DIR}")
        if residual_samples:
            arr = np.array(residual_samples)
            print(f"residual px: min={arr.min():.1f} median={np.median(arr):.1f} "
                  f"p10={np.percentile(arr, 10):.1f} max={arr.max():.1f}")
        print(f"strikes: {strikes}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward-model ball tracking demo")
    ap.add_argument("--headless", type=int, default=None, metavar="N",
                    help="render N frames without a window and save preview PNGs")
    ap.add_argument("--top", action="store_true", help="show overhead debug view inset")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    run(args.headless, args.top, args.seed)


if __name__ == "__main__":
    main()
