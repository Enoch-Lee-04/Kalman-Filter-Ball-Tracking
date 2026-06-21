"""Detector interface shared by the simulated and (Phase 2) YOLO detectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass
class Detection:
    """Ball detection in image space."""
    cx: float
    cy: float
    bbox: tuple  # (x1, y1, x2, y2)

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy], dtype=float)


class Detector(Protocol):
    def detect(self, frame: np.ndarray, sim) -> Optional[Detection]:
        """Return a Detection for the ball, or None if not visible.

        `sim` is the SimWorld, giving simulated detectors access to ground-truth
        geometry. Real detectors (YOLO) ignore it and use only `frame`.
        """
        ...
