"""SimWorld: owns the MuJoCo model/data and syncs mocap bodies each frame."""
from __future__ import annotations

import random
from typing import Optional, Sequence

import mujoco
import numpy as np

from . import arena
from .ball import Ball
from .camera import RobotCamera
from .robot import Robot


class SimWorld:
    def __init__(self, cfg: dict, pillars: Optional[Sequence] = None,
                 rng: random.Random | None = None):
        self.cfg = cfg
        self.rng = rng or random.Random()

        self.model = arena.build_model(cfg, pillars)
        self.data = mujoco.MjData(self.model)

        self.ball = Ball(cfg, self.rng)
        self.robot = Robot(cfg, self.rng)
        self.ball.set_keep_out_center(*self.robot.position)
        self.camera = RobotCamera(
            self.model, arena.ROBOT_CAM,
            cfg["camera"]["width"], cfg["camera"]["height"],
        )
        self.top_camera = RobotCamera(
            self.model, arena.TOP_CAM,
            cfg["camera"]["width"], cfg["camera"]["height"],
        )

        self._ball_mocap = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, arena.BALL_BODY)]
        self._robot_mocap = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, arena.ROBOT_BODY)]

        self._sync()

    def _sync(self) -> None:
        bx, by, bz = self.ball.position
        self.data.mocap_pos[self._ball_mocap] = (bx, by, bz)
        rx, ry = self.robot.position
        self.data.mocap_pos[self._robot_mocap] = (rx, ry, 0.0)
        self.data.mocap_quat[self._robot_mocap] = self.robot.facing_quat()
        mujoco.mj_forward(self.model, self.data)

    def step(self, dt: float) -> None:
        self.ball.set_keep_out_center(*self.robot.position)
        self.ball.step(dt)
        self._sync()

    def relocate_robot_to_ball(self) -> None:
        bx, by, _ = self.ball.position
        self.robot.relocate_near(bx, by)
        self.ball.set_keep_out_center(*self.robot.position)
        self._sync()

    def render_robot_view(self) -> np.ndarray:
        return self.camera.render_bgr(self.data)

    def render_top_view(self) -> np.ndarray:
        return self.top_camera.render_bgr(self.data)

    def ball_image_center(self) -> Optional[np.ndarray]:
        """Ground-truth ball center in the robot image (no detector noise)."""
        p = self.camera.project(self.data, self.ball.position)
        if p is None:
            return None
        return np.array([p[0], p[1]], dtype=float)
