"""Free orbit camera for third-person scene views.

Uses MuJoCo's MjvCamera (mjCAMERA_FREE) via the shared fixed-size RenderPool.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .render_pool import RenderPool


class OrbitCamera:
    def __init__(self, model: mujoco.MjModel, cfg: dict, pool: RenderPool):
        cam_cfg = cfg["camera"]
        disp_cfg = cfg.get("display", {})
        arena = cfg["arena"]
        width = disp_cfg.get("width", cam_cfg["width"])
        height = disp_cfg.get("height", cam_cfg["height"])

        self.model = model
        self.width = width
        self.height = height
        self._pool = pool

        extent = max(arena["width"], arena["depth"])
        self._default = {
            "lookat": np.array([0.0, 0.0, 0.3]),
            "distance": extent * 1.8,
            "azimuth": 135.0,
            "elevation": -30.0,
        }
        self._min_distance = extent * 0.6
        self._max_distance = extent * 4.0

        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.reset()

    def reset(self) -> None:
        self._cam.lookat[:] = self._default["lookat"]
        self._cam.distance = self._default["distance"]
        self._cam.azimuth = self._default["azimuth"]
        self._cam.elevation = self._default["elevation"]

    def orbit(self, delta_azimuth_deg: float, delta_elevation_deg: float) -> None:
        self._cam.azimuth += delta_azimuth_deg
        self._cam.elevation = float(np.clip(
            self._cam.elevation + delta_elevation_deg, -85.0, -5.0,
        ))

    def zoom(self, delta: float) -> None:
        """Positive delta zooms in (reduces distance)."""
        self._cam.distance = float(np.clip(
            self._cam.distance - delta, self._min_distance, self._max_distance,
        ))

    @property
    def distance(self) -> float:
        return float(self._cam.distance)

    def render_bgr(self, data: mujoco.MjData) -> np.ndarray:
        return self._pool.render_bgr(data, self._cam)
