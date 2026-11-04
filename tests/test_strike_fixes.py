"""Respawn, contact, reach and turning: the four fixes that keep the numbers honest."""
import math

import numpy as np

from config import apply_overrides, load_config
from sim.rng import RandomStreams
from sim.robot import Robot
from sim.world import SimWorld
from simulation import Simulation

SEEDS = range(10)
FRAMES = 1500


def _strike_runs(cfg=None):
    """Yield (sim, finished StrikeRecord) for every strike in the seed set."""
    cfg = cfg or load_config()
    for seed in SEEDS:
        sim = Simulation(cfg, seed=seed)
        for _ in range(FRAMES):
            info = sim.step()
            if info.strike_finished is not None:
                yield sim, info.strike_finished


# -- respawn order ----------------------------------------------------------
def test_respawn_when_charge_ends_on_its_target():
    """A charge that ends exactly on its target point used to leave the robot
    with a (0, 0) facing vector, every respawn candidate failed the in-front
    test, and the ball stayed where it was hit, beside the robot."""
    world = SimWorld(load_config(), streams=RandomStreams.from_seed(1))
    world.robot.x, world.robot.y = 1.5, 0.5
    world.robot.face_toward(1.5, 0.5)            # standing on its own target
    world.ball.x, world.ball.y = 1.8, 0.6        # just hit, right beside the robot

    world.respawn_after_strike()

    assert world.robot_ball_distance() > 1.2
    assert world.ground_point_in_view(world.ball.x, world.ball.y)
    to_centre = np.array([-1.5, -0.5]) / math.hypot(1.5, 0.5)
    assert np.allclose(world.robot.facing, to_centre)


def test_every_respawn_lands_in_view():
    respawns = 0
    for sim, _ in _strike_runs():
        world = sim.world
        respawns += 1
        assert world.robot_ball_distance() > 1.2
        assert world.ground_point_in_view(world.ball.x, world.ball.y)
    assert respawns > 5


# -- contact every step -----------------------------------------------------
def test_contact_is_checked_on_every_step_of_the_charge():
    """A hit ends at first contact; a miss never touched the ball at any step."""
    records = [rec for _, rec in _strike_runs()]
    assert any(rec.hit for rec in records)
    strike_duration = load_config()["robot"]["strike_duration_s"]
    for rec in records:
        assert rec.hit == (rec.closest_m <= rec.contact_limit_m)
        if rec.hit:
            assert rec.end_dist_m <= rec.contact_limit_m
            assert rec.duration_s < strike_duration
        else:
            assert rec.duration_s >= strike_duration - 0.02  # ran the whole window


# -- reach gate -------------------------------------------------------------
def test_robot_holds_when_the_ball_cannot_be_reached():
    """Reach here (1.0 m/s * 0.1 s + contact ~0.58 m) is shorter than the ball's
    keep-out radius (1.0 m), so no charge could ever land. The robot still
    becomes confident, but must hold instead of striking."""
    cfg = apply_overrides(load_config(), ["robot.strike_speed=1.0",
                                          "robot.strike_duration_s=0.1"])
    held = 0
    for seed in SEEDS:
        stats = Simulation(cfg, seed=seed).run(FRAMES)
        assert stats.strikes == 0
        held += stats.held_frames
    assert held > 0


def test_strike_decisions_use_the_robots_estimate():
    """The gate uses the filter's belief, which stays close to the truth."""
    for _, rec in _strike_runs():
        assert abs(rec.start_est_dist_m - rec.start_dist_m) < 0.5


# -- turning ----------------------------------------------------------------
def _standing_still_view(cfg, seeds=range(5)):
    seen = frames = 0
    offsets = []
    for seed in seeds:
        sim = Simulation(cfg, seed=seed, allow_strikes=False)
        width = sim.world.camera.width
        for _ in range(FRAMES):
            res = sim.step().result
            frames += 1
            if res.detection_center is not None:
                seen += 1
                offsets.append(abs(res.detection_center[0] - width / 2) / width)
    return 1.0 - seen / frames, float(np.median(offsets))


def test_turning_keeps_the_ball_in_view_and_near_centre():
    cfg = load_config()
    out_of_view, median_offset = _standing_still_view(cfg)
    assert out_of_view == 0.0
    assert median_offset < 0.05  # within 5% of the width from the centre line

    fixed = apply_overrides(cfg, ["robot.turn_rate_deg_s=0"])
    out_fixed, offset_fixed = _standing_still_view(fixed)
    assert out_fixed > 0.0          # the test can tell the difference
    assert offset_fixed > median_offset


def test_turn_is_rate_limited_and_takes_the_short_way_round():
    cfg = apply_overrides(load_config(), ["robot.turn_rate_deg_s=180",
                                          "robot.start_pos=[0.0, 0.0]"])
    robot = Robot(cfg)
    robot.yaw = math.radians(170)
    target = (math.cos(math.radians(-170)), math.sin(math.radians(-170)))

    robot.turn_toward(*target, dt=0.01)  # 20 deg away across the +-180 seam
    assert math.isclose(math.degrees(robot.yaw), 171.8, abs_tol=1e-9)

    for _ in range(20):
        robot.turn_toward(*target, dt=0.01)
    assert math.isclose(math.degrees(robot.yaw), -170, abs_tol=1e-9)  # arrives, no overshoot


def test_zero_turn_rate_never_turns():
    cfg = apply_overrides(load_config(), ["robot.turn_rate_deg_s=0"])
    robot = Robot(cfg)
    before = robot.yaw
    robot.turn_toward(3.0, 0.0, dt=1.0)
    assert robot.yaw == before


def test_facing_own_position_keeps_heading():
    robot = Robot(load_config())
    before = robot.facing
    robot.face_toward(*robot.position)
    assert robot.facing == before
