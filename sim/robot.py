"""Stationary robot state: holds its ground position and handles relocation.

In Phase 1 the robot does not physically move while tracking. When it gains
enough confidence it "hits" the ball and relocates to a new resting spot, from
which it begins tracking afresh.
"""
from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np


class Robot:
    def __init__(self, cfg: dict, rng: random.Random | None = None):
        arena = cfg["arena"]
        rcfg = cfg["robot"]
        self.rng = rng or random.Random()

        self.x, self.y = rcfg["start_pos"]
        self.base_radius = rcfg["base_radius"]

        margin = rcfg["base_radius"] + 0.2
        self.hw = arena["width"] / 2.0 - margin
        self.hd = arena["depth"] / 2.0 - margin

        # the robot always faces the arena centre so its camera sees the field
        self.target = (0.0, 0.0)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def facing(self) -> Tuple[float, float]:
        """Unit vector the camera looks along (toward the target), in world XY."""
        dx, dy = self.target[0] - self.x, self.target[1] - self.y
        n = math.hypot(dx, dy) or 1.0
        return (dx / n, dy / n)

    def facing_quat(self) -> np.ndarray:
        """Quaternion (w, x, y, z) yawing the body so its +Y axis points at the
        target. The camera is mounted looking along the body +Y axis."""
        phi = math.atan2(self.target[1] - self.y, self.target[0] - self.x)
        alpha = phi - math.pi / 2.0  # body +Y starts along world +Y
        return np.array([math.cos(alpha / 2.0), 0.0, 0.0, math.sin(alpha / 2.0)])

    def relocate_near(self, x: float, y: float, standoff: float = 0.9) -> None:
        """Move to a new stationary spot a short distance from (x, y)."""
        angle = self.rng.uniform(-math.pi, math.pi)
        nx = x + standoff * math.cos(angle)
        ny = y + standoff * math.sin(angle)
        self.x = max(-self.hw, min(self.hw, nx))
        self.y = max(-self.hd, min(self.hd, ny))
