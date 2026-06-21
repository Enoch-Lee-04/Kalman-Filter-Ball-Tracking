"""Builds the MuJoCo model (MjModel) for the arena, ball, and stationary robot.

The arena is a rectangular box. The ball and robot are mocap bodies, meaning we
drive their positions directly from Python each frame (kinematic control) rather
than relying on the physics solver. This keeps motion fully under our control,
which is what we want for a clean forward-model demo.

Phase 1 builds an empty box. `pillars` is wired up now so Phase 2 only needs to
pass a list of obstacles.
"""
from __future__ import annotations

from typing import Optional, Sequence

import mujoco

BALL_BODY = "ball"
ROBOT_BODY = "robot"
ROBOT_CAM = "robotcam"
TOP_CAM = "topcam"

Pillar = tuple  # (x, y, radius, height)


def build_xml(cfg: dict, pillars: Optional[Sequence[Pillar]] = None) -> str:
    arena = cfg["arena"]
    ball = cfg["ball"]
    robot = cfg["robot"]
    cam = cfg["camera"]

    hw = arena["width"] / 2.0
    hd = arena["depth"] / 2.0
    h = arena["height"]
    t = 0.05  # wall thickness

    rx, ry = robot["start_pos"]
    br = robot["base_radius"]
    bh = robot["base_height"]

    ball_r = ball["radius"]

    # Camera up-vector tilted downward so the floor is visible.
    tilt = cam["tilt"]
    # xyaxes: x-axis then y-axis (up) of the camera, expressed in the body frame.
    # x=(1,0,0); y=(0, tilt, 1) -> view direction points into +Y and downward.
    xyaxes = f"1 0 0 0 {tilt} 1"

    pillar_xml = ""
    for i, p in enumerate(pillars or []):
        px, py, pr, ph = p
        pillar_xml += (
            f'    <geom name="pillar{i}" type="cylinder" '
            f'pos="{px} {py} {ph/2:.3f}" size="{pr} {ph/2:.3f}" '
            f'rgba="0.30 0.31 0.34 1"/>\n'
        )

    return f"""
<mujoco model="forward_model_arena">
  <option timestep="0.01" gravity="0 0 0"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"/>
    <map znear="0.05"/>
  </visual>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.12 0.12 0.14"
             rgb2="0.16 0.16 0.19" width="300" height="300"/>
    <material name="floor_mat" texture="grid" texrepeat="6 6" reflectance="0.05"/>
    <material name="wall_mat" rgba="0.20 0.21 0.24 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 {h + 1.5:.2f}" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <geom name="floor" type="plane" size="{hw} {hd} 0.1" material="floor_mat"/>

    <!-- perimeter walls -->
    <geom name="wall_xp" type="box" pos="{hw} 0 {h/2:.3f}" size="{t} {hd} {h/2:.3f}" material="wall_mat"/>
    <geom name="wall_xn" type="box" pos="{-hw} 0 {h/2:.3f}" size="{t} {hd} {h/2:.3f}" material="wall_mat"/>
    <geom name="wall_yp" type="box" pos="0 {hd} {h/2:.3f}" size="{hw} {t} {h/2:.3f}" material="wall_mat"/>
    <geom name="wall_yn" type="box" pos="0 {-hd} {h/2:.3f}" size="{hw} {t} {h/2:.3f}" material="wall_mat"/>
{pillar_xml}
    <!-- overhead debug camera -->
    <camera name="{TOP_CAM}" pos="0 0 {max(hw, hd) * 2.4:.2f}" xyaxes="1 0 0 0 1 0"/>

    <!-- ball: kinematically driven mocap body -->
    <body name="{BALL_BODY}" mocap="true" pos="0 0 {ball_r}">
      <geom name="ball_geom" type="sphere" size="{ball_r}" rgba="0.95 0.95 0.97 1"/>
      <!-- small marker so spin/orientation is visible -->
      <geom type="capsule" fromto="0 0 0 {ball_r*0.9:.3f} 0 0" size="{ball_r*0.18:.3f}"
            rgba="0.85 0.25 0.30 1" contype="0" conaffinity="0"/>
    </body>

    <!-- robot: stationary mocap body carrying the camera -->
    <body name="{ROBOT_BODY}" mocap="true" pos="{rx} {ry} 0">
      <geom name="robot_base" type="cylinder" pos="0 0 {bh/2:.3f}" size="{br} {bh/2:.3f}"
            rgba="0.16 0.34 0.82 1"/>
      <geom name="robot_head" type="box" pos="0 0 {bh + 0.12:.3f}" size="0.10 0.10 0.10"
            rgba="0.20 0.42 0.95 1"/>
      <camera name="{ROBOT_CAM}" pos="0 0 {bh + cam['mount_height']:.3f}"
              xyaxes="{xyaxes}" fovy="{cam['fovy']}"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def build_model(cfg: dict, pillars: Optional[Sequence[Pillar]] = None) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_xml(cfg, pillars))
