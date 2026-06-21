"""Robot camera: renders frames and projects 3D world points to image pixels.

The projection uses MuJoCo's actual camera pose (read from MjData after
mj_forward), so it stays correct even when the robot relocates after a hit.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import mujoco
import numpy as np


class RobotCamera:
    def __init__(self, model: mujoco.MjModel, cam_name: str, width: int, height: int):
        self.model = model
        self.width = width
        self.height = height
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self.cam_id < 0:
            raise ValueError(f"camera '{cam_name}' not found in model")
        self._renderer = mujoco.Renderer(model, height, width)
        self.fovy = float(model.cam_fovy[self.cam_id])
        # focal length in pixels (square pixels, derived from vertical FOV)
        self.focal = 0.5 * height / math.tan(math.radians(self.fovy) / 2.0)

    def render_bgr(self, data: mujoco.MjData) -> np.ndarray:
        """Render an RGB frame and return it as BGR for OpenCV."""
        self._renderer.update_scene(data, camera=self.cam_id)
        rgb = self._renderer.render()
        return rgb[:, :, ::-1].copy()

    def project(self, data: mujoco.MjData, world_xyz) -> Optional[Tuple[float, float, float]]:
        """Project a world point to (u, v, depth). Returns None if behind camera.

        depth is the distance along the camera's viewing axis (meters).
        """
        cam_pos = data.cam_xpos[self.cam_id]
        cam_mat = data.cam_xmat[self.cam_id].reshape(3, 3)  # cols = cam axes in world

        rel = np.asarray(world_xyz, dtype=float) - cam_pos
        local = cam_mat.T @ rel  # express point in camera frame
        xc, yc, zc = local
        depth = -zc  # camera looks down its -Z axis
        if depth <= 1e-4:
            return None

        u = self.width / 2.0 + self.focal * (xc / depth)
        v = self.height / 2.0 - self.focal * (yc / depth)
        return (u, v, depth)

    def unproject_to_plane(self, data, u: float, v: float, plane_z: float):
        """Back-project image pixel (u, v) onto the world plane z == plane_z.

        This is the monocular ground-plane assumption: because the ball rolls at
        a known height, a single camera ray is enough to recover its world (x, y).
        Returns np.array([x, y]) or None if the ray does not hit the plane ahead.
        """
        cam_pos = data.cam_xpos[self.cam_id]
        cam_mat = data.cam_xmat[self.cam_id].reshape(3, 3)

        # ray direction in camera frame, then rotate to world
        local = np.array([(u - self.width / 2.0) / self.focal,
                          (self.height / 2.0 - v) / self.focal,
                          -1.0])
        dir_world = cam_mat @ local
        if abs(dir_world[2]) < 1e-9:
            return None
        s = (plane_z - cam_pos[2]) / dir_world[2]
        if s <= 0:
            return None
        hit = cam_pos + s * dir_world
        return np.array([hit[0], hit[1]], dtype=float)

    def bbox_for_sphere(self, data, world_xyz, radius) -> Optional[Tuple[int, int, int, int]]:
        """Return an axis-aligned bbox (x1, y1, x2, y2) for a sphere, or None."""
        p = self.project(data, world_xyz)
        if p is None:
            return None
        u, v, depth = p
        r_px = self.focal * radius / depth
        return (int(u - r_px), int(v - r_px), int(u + r_px), int(v + r_px))
