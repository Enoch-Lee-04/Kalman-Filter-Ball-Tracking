"""Tracking controller: the predict -> act -> correct loop.

Owns the detector, Kalman forward model, latency pipeline, and residual
tracker. Each frame it:
  1. detects the ball (current frame),
  2. feeds the *delayed* detection to the Kalman filter,
  3. predicts where the ball will be after the full sense+think+move latency,
  4. scores past predictions against reality (residual),
  5. decides whether the robot is confident enough to strike.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

from forward_model.kalman import KalmanCV
from forward_model.latency import LatencyPipeline
from forward_model.residual import ResidualTracker
from perception.sim_detector import SimDetector


@dataclass
class FrameResult:
    detection_bbox: Optional[tuple]
    detection_center: Optional[np.ndarray]
    predicted_center: Optional[np.ndarray]
    predicted_bbox: Optional[tuple]
    predicted_world: Optional[np.ndarray]
    residual_px: float
    confidence: float
    confident: bool


class TrackingController:
    def __init__(self, cfg: dict, noise_rng: random.Random | None = None):
        """noise_rng is the detector-noise stream; nothing else draws from it."""
        self.cfg = cfg

        fm = cfg["forward_model"]
        lat = cfg["latency_ms"]

        self.detector = SimDetector(cfg, noise_rng)
        self.kalman = KalmanCV(
            process_noise=fm["process_noise"],
            measurement_noise=fm["measurement_noise"],
            initial_velocity_var=fm["initial_velocity_var"],
        )
        self.latency = LatencyPipeline(lat["sense"], lat["think"], lat["move"])
        self.residual = ResidualTracker(
            threshold_px=fm["residual_threshold_px"],
            confidence_frames=fm["confidence_frames"],
        )
        self.horizon_ms = int(round(self.latency.horizon * 1000))
        self._last_meas_t: Optional[float] = None

    def reset(self) -> None:
        self.kalman.reset()
        self.latency.clear()
        self.residual.reset()
        self._last_meas_t = None

    def estimate_at(self, t: float) -> Optional[np.ndarray]:
        """The filter's belief of the ball's world (x, y) at sim time t,
        extrapolated at constant velocity from the last measurement it folded
        in. Uses no ground truth; None until the filter has a measurement."""
        pv = self.kalman.position_velocity()
        if pv is None or self._last_meas_t is None:
            return None
        pos, vel = pv
        return pos + vel * (t - self._last_meas_t)

    def step(self, t: float, dt: float, frame: np.ndarray, world) -> FrameResult:
        plane_z = world.ball.radius

        # 1. detect in the current frame, then back-project to world (x, y)
        det = self.detector.detect(frame, world)
        if det is not None:
            world_xy = world.camera.unproject_to_plane(world.data, det.cx, det.cy, plane_z)
            if world_xy is not None:
                self.latency.push_measurement(t, world_xy)

        # 2. feed the delayed (stale) world measurement to the filter
        delayed = self.latency.delayed_measurement(t)
        new_measurement = False
        if delayed is not None:
            ts, z = delayed
            if self._last_meas_t is None or ts > self._last_meas_t:
                step_dt = dt if self._last_meas_t is None else max(1e-3, ts - self._last_meas_t)
                self.kalman.update(z, step_dt)
                self._last_meas_t = ts
                new_measurement = True

        # 3. forward prediction over the full latency horizon (world meters).
        #    Logged against the measurement time so the residual is scored at the
        #    correct future instant (measurement_time + horizon).
        predicted_world = self.kalman.predict_ahead(self.latency.horizon)
        predicted_center = None
        predicted_bbox = None
        if predicted_world is not None:
            if new_measurement:
                self.latency.log_prediction(self._last_meas_t, predicted_world)
            world_xyz = (predicted_world[0], predicted_world[1], plane_z)
            p = world.camera.project(world.data, world_xyz)
            if p is not None:
                predicted_center = np.array([p[0], p[1]])
                predicted_bbox = world.camera.bbox_for_sphere(world.data, world_xyz, world.ball.radius)

        # 4. score a due prediction against reality, both projected to the image
        due = self.latency.due_prediction(t)
        if due is not None:
            actual_img = world.ball_image_center()
            pred_p = world.camera.project(world.data, (due[0], due[1], plane_z))
            if actual_img is not None and pred_p is not None:
                self.residual.add(np.array([pred_p[0], pred_p[1]]), actual_img)

        return FrameResult(
            detection_bbox=det.bbox if det else None,
            detection_center=det.center if det else None,
            predicted_center=predicted_center,
            predicted_bbox=predicted_bbox,
            predicted_world=predicted_world,
            residual_px=self.residual.smoothed,
            confidence=self.residual.confidence_ratio,
            confident=self.residual.confident,
        )
