"""SimWorld: owns the MuJoCo model/data and syncs mocap bodies each frame."""
from __future__ import annotations

import math
import random
from typing import Optional, Sequence

import mujoco
import numpy as np

from . import arena
from .ball import Ball
from .camera import RobotCamera
from .orbit_camera import OrbitCamera
from .render_pool import RenderPool
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
        disp = cfg.get("display", {})
        pool_w = disp.get("width", cfg["camera"]["width"])
        pool_h = disp.get("height", cfg["camera"]["height"])
        self._display_pool = RenderPool(self.model, pool_w, pool_h)
        self.orbit_camera = OrbitCamera(self.model, cfg, self._display_pool)
        self._robot_cam_id = self.camera.cam_id
        ch, cw = cfg["camera"]["height"], cfg["camera"]["width"]
        self._tracking_stub = np.zeros((ch, cw, 3), dtype=np.uint8)

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

    def step(self, dt: float, striking: bool = False) -> None:
        self.ball.set_keep_out_center(*self.robot.position)
        self.ball.step(dt, apply_keep_out=not striking)
        self._sync()

    def strike_toward(self, target_xy: tuple[float, float], dt: float, speed: float) -> None:
        self.robot.face_toward(target_xy[0], target_xy[1])
        self.robot.move_toward(target_xy[0], target_xy[1], dt, speed)
        self._sync()

    def strike_contact_distance(self) -> float:
        return self.robot.base_radius + self.ball.radius

    def strike_connected(self, slop: float = 0.12) -> bool:
        bx, by, _ = self.ball.position
        rx, ry = self.robot.position
        return math.hypot(bx - rx, by - ry) <= self.strike_contact_distance() + slop

    def relocate_robot_to_ball(self) -> None:
        bx, by, _ = self.ball.position
        self.robot.relocate_near(bx, by)
        self.ball.set_keep_out_center(*self.robot.position)
        self._sync()

    def render_robot_view(self) -> np.ndarray:
        return self.camera.render_bgr(self.data)

    @property
    def tracking_frame(self) -> np.ndarray:
        """Placeholder frame for the sim detector (uses 3D projection, not pixels)."""
        return self._tracking_stub

    def render_robot_display_view(self) -> np.ndarray:
        """Render robot POV at pool resolution; composite downscales for inset size."""
        return self._display_pool.render_bgr(self.data, self._robot_cam_id)

    def display_ui_scale(self) -> float:
        """Scale HUD layout relative to the tracking camera baseline."""
        return self._display_pool.width / self.camera.width

    def display_tracking(self, result) -> dict:
        """Re-project tracking overlays to the display render resolution."""
        dw, dh = self._display_pool.width, self._display_pool.height
        cam = self.camera
        plane_z = self.ball.radius
        out = {
            "match_bbox": None,
            "match_center": None,
            "predicted_center": None,
            "predicted_bbox": None,
        }

        if result.detection_center is not None:
            xy = cam.unproject_to_plane(
                self.data, result.detection_center[0], result.detection_center[1], plane_z,
            )
            if xy is not None:
                xyz = (xy[0], xy[1], plane_z)
                p = cam.project_at(self.data, xyz, dw, dh)
                if p is not None:
                    out["match_center"] = np.array([p[0], p[1]])
                    out["match_bbox"] = cam.bbox_for_sphere_at(
                        self.data, xyz, self.ball.radius, dw, dh,
                    )

        if result.predicted_center is not None:
            xy = cam.unproject_to_plane(
                self.data, result.predicted_center[0], result.predicted_center[1], plane_z,
            )
            if xy is not None:
                xyz = (xy[0], xy[1], plane_z)
                p = cam.project_at(self.data, xyz, dw, dh)
                if p is not None:
                    out["predicted_center"] = np.array([p[0], p[1]])
                    out["predicted_bbox"] = cam.bbox_for_sphere_at(
                        self.data, xyz, self.ball.radius, dw, dh,
                    )

        return out

    def render_top_view(self) -> np.ndarray:
        return self.top_camera.render_bgr(self.data)

    def render_orbit_view(self) -> np.ndarray:
        return self.orbit_camera.render_bgr(self.data)

    def ball_image_center(self) -> Optional[np.ndarray]:
        """Ground-truth ball center in the robot image (no detector noise)."""
        p = self.camera.project(self.data, self.ball.position)
        if p is None:
            return None
        return np.array([p[0], p[1]], dtype=float)
