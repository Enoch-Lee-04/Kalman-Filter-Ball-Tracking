"""Residual tracking: how far each forward prediction lands from reality.

The residual is the pixel distance between where the model predicted the ball
would be and where it actually ended up. A smoothed residual drives the
robot's confidence: once predictions are consistently accurate, it strikes.
"""
from __future__ import annotations

import numpy as np


class ResidualTracker:
    def __init__(self, threshold_px: float, confidence_frames: int, ema_alpha: float = 0.2):
        self.threshold = threshold_px
        self.confidence_frames = confidence_frames
        self.ema_alpha = ema_alpha

        self.smoothed: float = float("nan")
        self.last: float = float("nan")
        self.streak: int = 0

    def reset(self) -> None:
        self.smoothed = float("nan")
        self.last = float("nan")
        self.streak = 0

    def add(self, predicted_xy: np.ndarray, actual_xy: np.ndarray) -> float:
        err = float(np.linalg.norm(predicted_xy - actual_xy))
        self.last = err
        if np.isnan(self.smoothed):
            self.smoothed = err
        else:
            self.smoothed = (1 - self.ema_alpha) * self.smoothed + self.ema_alpha * err

        if self.smoothed <= self.threshold:
            self.streak += 1
        else:
            self.streak = 0
        return err

    @property
    def confident(self) -> bool:
        return self.streak >= self.confidence_frames

    @property
    def confidence_ratio(self) -> float:
        return min(1.0, self.streak / max(1, self.confidence_frames))
