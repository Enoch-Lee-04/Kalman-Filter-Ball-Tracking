"""Single shared MuJoCo offscreen renderer for orbit + inset views.

MuJoCo only supports one active Renderer per model for display views. Orbit and
inset share this pool at a fixed resolution; the inset is downscaled at composite
time so the pool never thrashes between sizes.
"""
from __future__ import annotations

import mujoco
import numpy as np


class RenderPool:
    def __init__(self, model: mujoco.MjModel, width: int, height: int):
        self.model = model
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        # Created on first render, so a headless run never opens a GL context.
        self._renderer: mujoco.Renderer | None = None

    def render_bgr(self, data: mujoco.MjData, camera) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)
        self._renderer.update_scene(data, camera=camera)
        rgb = self._renderer.render()
        return rgb[:, :, ::-1].copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
