"""Phase 1 detector: projects the ball's true 3D position to an image bbox.

A small amount of Gaussian pixel noise is added so the Kalman filter has
something realistic to smooth. This is the perfect "ground truth" stand-in
before swapping in a real detector (YOLO) in Phase 2.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from .base import Detection


class SimDetector:
    def __init__(self, cfg: dict, rng: random.Random | None = None):
        self.noise_px = cfg["perception"]["noise_px"]
        self.rng = rng or random.Random()

    def detect(self, frame: np.ndarray, sim) -> Optional[Detection]:
        bbox = sim.camera.bbox_for_sphere(sim.data, sim.ball.position, sim.ball.radius)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox

        # reject if fully outside the frame
        h, w = frame.shape[:2]
        if x2 < 0 or y2 < 0 or x1 > w or y1 > h:
            return None

        nx = self.rng.gauss(0, self.noise_px)
        ny = self.rng.gauss(0, self.noise_px)
        cx = (x1 + x2) / 2.0 + nx
        cy = (y1 + y2) / 2.0 + ny
        bbox_noisy = (x1 + nx, y1 + ny, x2 + nx, y2 + ny)
        return Detection(cx=cx, cy=cy, bbox=bbox_noisy)
