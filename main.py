"""Entry point for the forward-model ball tracking demo.

Usage:
  python main.py                     # live window (press q/ESC to quit)
  python main.py --headless 600      # render 600 frames, save preview PNGs
  python main.py --top               # also show the overhead debug view inset

Live controls:
  Tab          switch orbit / robot camera view
  drag         orbit (orbit view only)
  +/- / wheel  zoom (orbit view only)
  R            reset orbit camera
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np

from config import load_config
from simulation import Simulation
from sim.world import SimWorld
from viz import overlay
from viz.camera_controls import CameraControls
from viz.view_controls import ViewState, handle_view_key

PREVIEW_DIR = os.path.join("assets", "preview")
WINDOW_NAME = "Forward Model"
INSET_MARGIN = 5


def _draw_frame(frame: np.ndarray, result, controller, cfg, latency, status,
                striking, ui_scale, view_state: ViewState, world: SimWorld,
                hits: int, misses: int) -> np.ndarray:
    view_label = "Tab: robot view" if view_state.is_orbit else "Tab: orbit view"
    tracking = world.display_tracking(result) if view_state.is_robot else {
        "match_bbox": None,
        "match_center": None,
        "predicted_center": None,
        "predicted_bbox": None,
    }
    overlay.draw(
        frame,
        match_bbox=tracking["match_bbox"],
        predicted_center=tracking["predicted_center"],
        predicted_bbox=tracking["predicted_bbox"],
        match_center=tracking["match_center"],
        residual_px=result.residual_px,
        residual_threshold=cfg["forward_model"]["residual_threshold_px"],
        horizon_ms=controller.horizon_ms,
        latency=latency,
        confidence=result.confidence,
        status=status,
        striking=striking,
        ui_scale=ui_scale,
        tracking_scale=1.0,
        show_tracking=view_state.is_robot,
        view_label=view_label,
        hits=hits,
        misses=misses,
    )
    return frame


def run(headless_frames: int | None, show_top: bool, seed: int) -> None:
    cfg = load_config()
    sim = Simulation(cfg, seed=seed)
    world, controller = sim.world, sim.controller

    dt = sim.dt
    realtime = cfg["sim"]["realtime"] and headless_frames is None
    latency = cfg["latency_ms"]
    ui_scale = world.display_ui_scale()

    view_state = ViewState(mode="orbit")
    camera_controls: CameraControls | None = None
    if headless_frames is None:
        camera_controls = CameraControls(
            world.orbit_camera,
            WINDOW_NAME,
            is_orbit_active=lambda: view_state.is_orbit,
        )

    if headless_frames is not None:
        os.makedirs(PREVIEW_DIR, exist_ok=True)

    n = headless_frames if headless_frames is not None else 10 ** 12
    stats = sim.stats

    while sim.frame < n:
        loop_start = time.perf_counter()
        info = sim.step()
        frame_idx, result, striking = info.frame, info.result, info.striking

        if view_state.is_orbit:
            frame = world.render_orbit_view()
        else:
            frame = world.render_robot_display_view()

        _draw_frame(frame, result, controller, cfg, latency, info.status, striking,
                    ui_scale, view_state, world, stats.hits, stats.misses)

        if show_top:
            top = world.render_top_view()
            top_w = frame.shape[1] // 3
            top_h = max(1, int(top.shape[0] * top_w / top.shape[1]))
            top = cv2.resize(top, (top_w, top_h), interpolation=cv2.INTER_AREA)
            frame[INSET_MARGIN:INSET_MARGIN + top_h,
                  INSET_MARGIN:INSET_MARGIN + top_w] = top

        if headless_frames is not None:
            if frame_idx % 30 == 0 or striking:
                tag = "strike_" if striking else "frame_"
                path = os.path.join(PREVIEW_DIR, f"{tag}{frame_idx:05d}.png")
                cv2.imwrite(path, frame)
        else:
            if frame_idx == 0:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW_NAME, frame.shape[1], frame.shape[0])
                camera_controls.attach()
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            handle_view_key(key, view_state)
            if camera_controls is not None:
                camera_controls.handle_key(key)
            if realtime:
                elapsed = time.perf_counter() - loop_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    if headless_frames is None:
        cv2.destroyAllWindows()
    else:
        print(f"Saved preview frames to {PREVIEW_DIR}")
        res = stats.residual_summary()
        if res["count"]:
            print(f"residual px: min={res['min']:.1f} median={res['median']:.1f} "
                  f"p10={res['p10']:.1f} max={res['max']:.1f}")
        print(f"strikes: {stats.strikes}  hits: {stats.hits}  misses: {stats.misses}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward-model ball tracking demo")
    ap.add_argument("--headless", type=int, default=None, metavar="N",
                    help="render N frames without a window and save preview PNGs")
    ap.add_argument("--top", action="store_true",
                    help="show overhead debug view inset (top-left)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    run(args.headless, args.top, args.seed)


if __name__ == "__main__":
    main()
