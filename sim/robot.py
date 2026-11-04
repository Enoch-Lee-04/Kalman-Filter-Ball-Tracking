"""Robot state: ground position plus the yaw its camera looks along.

While tracking, the robot's base stays put but it turns in place (at a capped
rate) to keep the ball in view. When it gains enough confidence it charges the
ball, and tracks afresh from wherever the charge ended.
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

        # Max turn-in-place rate while tracking (rad/s); 0 = never turn.
        self.turn_rate = math.radians(rcfg.get("turn_rate_deg_s", 0.0))

        # World-frame heading of the camera (rad, 0 = +X). Starts at the arena
        # centre so the camera sees the field.
        self.yaw = math.pi / 2.0
        self.face_arena_center()

    def face_toward(self, x: float, y: float) -> None:
        """Snap the camera to look at (x, y) on the ground plane. Standing on
        (x, y) itself keeps the current heading."""
        dx, dy = x - self.x, y - self.y
        if math.hypot(dx, dy) > 1e-9:
            self.yaw = math.atan2(dy, dx)

    def turn_toward(self, x: float, y: float, dt: float) -> None:
        """Turn toward (x, y) by at most turn_rate * dt this step."""
        dx, dy = x - self.x, y - self.y
        if self.turn_rate <= 0 or math.hypot(dx, dy) < 1e-9:
            return
        diff = _wrap_angle(math.atan2(dy, dx) - self.yaw)
        max_step = self.turn_rate * dt
        self.yaw = _wrap_angle(self.yaw + max(-max_step, min(max_step, diff)))

    def face_arena_center(self) -> None:
        self.face_toward(0.0, 0.0)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def facing(self) -> Tuple[float, float]:
        """Unit vector the camera looks along, in world XY."""
        return (math.cos(self.yaw), math.sin(self.yaw))

    def facing_quat(self) -> np.ndarray:
        """Quaternion (w, x, y, z) yawing the body so its +Y axis points along
        the heading. The camera is mounted looking along the body +Y axis."""
        alpha = self.yaw - math.pi / 2.0  # body +Y starts along world +Y
        return np.array([math.cos(alpha / 2.0), 0.0, 0.0, math.sin(alpha / 2.0)])

    def move_toward(self, x: float, y: float, dt: float, speed: float = 2.5) -> None:
        """Physically advance toward (x, y) at constant speed."""
        dx, dy = x - self.x, y - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return
        step = min(speed * dt, dist)
        self.x += dx / dist * step
        self.y += dy / dist * step
        self.x = max(-self.hw, min(self.hw, self.x))
        self.y = max(-self.hd, min(self.hd, self.y))

    def relocate_near(self, x: float, y: float, standoff: float = 0.9) -> None:
        """Move to a new stationary spot a short distance from (x, y)."""
        angle = self.rng.uniform(-math.pi, math.pi)
        nx = x + standoff * math.cos(angle)
        ny = y + standoff * math.sin(angle)
        self.x = max(-self.hw, min(self.hw, nx))
        self.y = max(-self.hd, min(self.hd, ny))


def _wrap_angle(a: float) -> float:
    """Map an angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi
