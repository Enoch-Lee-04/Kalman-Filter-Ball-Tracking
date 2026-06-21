"""Smooth, random ball motion with elastic wall bounces.

The ball cruises at constant speed and steers toward random waypoints with a
capped turn rate. The capped turn rate produces gentle curves -- which is
exactly what makes a constant-velocity prediction overshoot, giving the forward
model something meaningful to correct.
"""
from __future__ import annotations

import math
import random
from typing import Tuple

import numpy as np


class Ball:
    def __init__(self, cfg: dict, rng: random.Random | None = None):
        arena = cfg["arena"]
        bcfg = cfg["ball"]
        self.rng = rng or random.Random()

        self.radius = bcfg["radius"]
        self.speed = bcfg["speed"]
        self.max_turn = math.radians(bcfg["max_turn_deg"])
        self.waypoint_interval = bcfg["waypoint_interval"]
        self.margin = bcfg["margin"]
        self.keep_out = bcfg.get("keep_out", 0.0)

        self.hw = arena["width"] / 2.0 - self.radius
        self.hd = arena["depth"] / 2.0 - self.radius

        # circular keep-out zone (the robot's post); updated each frame
        self.avoid_x = 0.0
        self.avoid_y = 0.0

        self.x = 0.0
        self.y = 0.0
        self.heading = self.rng.uniform(-math.pi, math.pi)
        self._waypoint_timer = 0.0
        self._pick_waypoint()

    # -- public API -------------------------------------------------------
    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.radius)

    @property
    def velocity_xy(self) -> Tuple[float, float]:
        return (self.speed * math.cos(self.heading), self.speed * math.sin(self.heading))

    def set_keep_out_center(self, x: float, y: float) -> None:
        self.avoid_x, self.avoid_y = x, y

    def reset_far_from(self, x: float, y: float, facing=None) -> None:
        """Place the ball away from (x, y) and, if a facing vector is given, in
        front of the robot so it is immediately visible. Then pick a waypoint."""
        for _ in range(80):
            nx = self.rng.uniform(-self.hw, self.hw)
            ny = self.rng.uniform(-self.hd, self.hd)
            if math.hypot(nx - x, ny - y) <= 1.2:
                continue
            if facing is not None:
                dx, dy = nx - x, ny - y
                n = math.hypot(dx, dy) or 1.0
                if (dx / n) * facing[0] + (dy / n) * facing[1] < 0.5:
                    continue  # not in front of the camera
            self.x, self.y = nx, ny
            break
        self.heading = self.rng.uniform(-math.pi, math.pi)
        self._pick_waypoint()

    def step(self, dt: float) -> Tuple[float, float, float]:
        self._waypoint_timer += dt
        if self._waypoint_timer >= self.waypoint_interval or self._reached_waypoint():
            self._pick_waypoint()

        desired = math.atan2(self.wy - self.y, self.wx - self.x)
        self.heading = _turn_toward(self.heading, desired, self.max_turn * dt)

        vx = self.speed * math.cos(self.heading)
        vy = self.speed * math.sin(self.heading)
        self.x += vx * dt
        self.y += vy * dt

        self._bounce()
        self._avoid_keep_out()
        return self.position

    # -- internals --------------------------------------------------------
    def _pick_waypoint(self) -> None:
        m = self.margin
        for _ in range(20):
            self.wx = self.rng.uniform(-self.hw + m, self.hw - m)
            self.wy = self.rng.uniform(-self.hd + m, self.hd - m)
            if self.keep_out <= 0 or math.hypot(
                    self.wx - self.avoid_x, self.wy - self.avoid_y) > self.keep_out + 0.3:
                break
        self._waypoint_timer = 0.0

    def _reached_waypoint(self) -> bool:
        return math.hypot(self.wx - self.x, self.wy - self.y) < 0.25

    def _avoid_keep_out(self) -> None:
        if self.keep_out <= 0:
            return
        dx = self.x - self.avoid_x
        dy = self.y - self.avoid_y
        dist = math.hypot(dx, dy)
        if dist < self.keep_out:
            if dist < 1e-6:
                dx, dy, dist = 1.0, 0.0, 1.0
            nx, ny = dx / dist, dy / dist
            # push to the boundary and reflect heading about the radial normal
            self.x = self.avoid_x + nx * self.keep_out
            self.y = self.avoid_y + ny * self.keep_out
            vx, vy = math.cos(self.heading), math.sin(self.heading)
            dot = vx * nx + vy * ny
            if dot < 0:  # heading into the zone -> reflect outward
                vx -= 2 * dot * nx
                vy -= 2 * dot * ny
                self.heading = math.atan2(vy, vx)

    def _bounce(self) -> None:
        if self.x > self.hw:
            self.x = self.hw
            self.heading = math.pi - self.heading
        elif self.x < -self.hw:
            self.x = -self.hw
            self.heading = math.pi - self.heading
        if self.y > self.hd:
            self.y = self.hd
            self.heading = -self.heading
        elif self.y < -self.hd:
            self.y = -self.hd
            self.heading = -self.heading


def _turn_toward(current: float, target: float, max_step: float) -> float:
    diff = (target - current + math.pi) % (2 * math.pi) - math.pi
    diff = max(-max_step, min(max_step, diff))
    return current + diff
