"""Ball motion, respawns and detector noise draw from independent streams."""
import math

from config import apply_overrides, load_config
from sim.ball import Ball
from sim.rng import RandomStreams
from simulation import Simulation


def _ball_path(cfg, seed=3, frames=1500):
    sim = Simulation(cfg, seed=seed, allow_strikes=False)
    path = []
    for _ in range(frames):
        sim.step()
        path.append(sim.world.ball.position)
    return path, sim.stats


def test_ball_path_ignores_filter_detector_and_camera_settings():
    """With the robot standing still, the ball's path depends only on the seed
    and the ball/arena config. Under the old shared generator the narrower
    camera (fewer visible frames -> fewer noise draws) moved the ball from
    frame 264 on. Turning is switched off in the changed run so the narrow
    camera really does lose the ball; turning in place must not move it either."""
    cfg = load_config()
    base, base_stats = _ball_path(cfg)
    changed_cfg = apply_overrides(cfg, [
        "forward_model.process_noise=0.5",
        "forward_model.measurement_noise=0.2",
        "forward_model.residual_threshold_px=20",
        "perception.noise_px=6.0",
        "camera.fovy=60",
        "robot.turn_rate_deg_s=0",
    ])
    changed, changed_stats = _ball_path(changed_cfg)
    # the camera change really does alter how often the detector draws noise
    assert changed_stats.detection_frames != base_stats.detection_frames
    assert changed == base


def test_ball_path_up_to_first_strike_ignores_filter_settings():
    """With strikes on, a filter change may move the strike, and the world then
    legitimately diverges. Until the robot first acts, it must not."""
    cfg = load_config()
    loose = apply_overrides(cfg, ["forward_model.residual_threshold_px=8"])
    a, b = Simulation(cfg, seed=7), Simulation(loose, seed=7)
    while not (a.striking or b.striking):
        a.step()
        b.step()
        assert a.world.ball.position == b.world.ball.position
    assert a.frame > 100  # we compared a meaningful stretch


def _state(rng):
    return rng.getstate()


def test_respawn_does_not_touch_motion_stream_and_vice_versa():
    cfg = load_config()
    streams = RandomStreams.from_seed(11)
    ball = Ball(cfg, motion_rng=streams.ball_motion, respawn_rng=streams.respawn)

    motion_before = _state(streams.ball_motion)
    ball.reset_far_from(0.0, -3.8, visible=lambda x, y: y > -2.0)
    assert _state(streams.ball_motion) == motion_before

    respawn_before = _state(streams.respawn)
    for _ in range(500):  # long enough to pick several waypoints
        ball.step(0.01)
    assert _state(streams.respawn) == respawn_before
    assert _state(streams.ball_motion) != motion_before


def test_detector_draws_only_detector_noise():
    sim = Simulation(load_config(), seed=5, allow_strikes=False)
    world = sim.world
    motion, respawn = _state(world.streams.ball_motion), _state(world.streams.respawn)
    noise = _state(world.streams.detector_noise)
    det = sim.controller.detector.detect(world.tracking_frame, world)
    assert det is not None
    assert _state(world.streams.ball_motion) == motion
    assert _state(world.streams.respawn) == respawn
    assert _state(world.streams.detector_noise) != noise


def test_streams_are_reproducible_and_distinct():
    a, b = RandomStreams.from_seed(7), RandomStreams.from_seed(7)
    draws_a = [a.ball_motion.random(), a.respawn.random(), a.detector_noise.random()]
    draws_b = [b.ball_motion.random(), b.respawn.random(), b.detector_noise.random()]
    assert draws_a == draws_b
    assert len(set(draws_a)) == 3
    assert RandomStreams.from_seed(8).ball_motion.random() != draws_a[0]
    assert not math.isnan(draws_a[0])
