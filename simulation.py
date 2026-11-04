"""One simulated episode: world + tracking controller + strike state machine.

This is the whole experiment with no rendering, no window, and no OpenCV, so
the same code drives the live demo (main.py), the batch harness
(tools/batch.py), and the seed-7 regression test. Rendering is something
main.py does *to* a Simulation between steps; it never changes the numbers.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from controller import FrameResult, TrackingController
from sim.rng import RandomStreams
from sim.world import SimWorld

TRACKING = "TRACKING"
STRIKE = "STRIKE"


@dataclass
class StrikeRecord:
    """One strike, from the decision to the hit/miss verdict."""
    index: int
    start_frame: int
    start_t: float
    start_dist_m: float        # robot -> ball distance when the robot committed (truth)
    start_est_dist_m: float    # the same distance as the robot's filter believed it
    start_residual_px: float   # smoothed residual that triggered the strike
    reach_m: float             # farthest a stationary ball could be and still be hit
    contact_limit_m: float
    end_frame: int = -1        # contact on a hit, the end of the charge on a miss
    end_t: float = math.nan
    end_dist_m: float = math.nan
    closest_m: float = math.inf  # nearest the robot got to the ball during the charge
    hit: bool = False

    @property
    def beyond_reach(self) -> bool:
        return self.start_dist_m > self.reach_m

    @property
    def duration_s(self) -> float:
        return self.end_t - self.start_t


@dataclass
class StepInfo:
    frame: int                 # 0-based index of the step just taken
    t: float                   # sim time after the step (s)
    result: FrameResult
    status: str
    striking: bool             # a strike was in progress during this step
    strike_finished: Optional[StrikeRecord] = None


@dataclass
class RunStats:
    seed: int
    frames: int = 0
    strikes: int = 0
    hits: int = 0
    misses: int = 0
    detection_frames: int = 0  # frames where the detector saw the ball
    held_frames: int = 0       # confident, but the ball was out of reach: no strike
    # The smoothed (EMA) residual, sampled once per frame whenever defined.
    # These are the numbers main.py has always printed as "residual px".
    residual_samples: list[float] = field(default_factory=list)
    strike_records: list[StrikeRecord] = field(default_factory=list)

    @property
    def out_of_view_frac(self) -> float:
        return 1.0 - self.detection_frames / self.frames if self.frames else math.nan

    def residual_summary(self) -> dict:
        if not self.residual_samples:
            return {"count": 0}
        arr = np.asarray(self.residual_samples, dtype=float)
        return {
            "count": int(arr.size),
            "min": float(arr.min()),
            "p10": float(np.percentile(arr, 10)),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "max": float(arr.max()),
        }

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "frames": self.frames,
            "strikes": self.strikes,
            "hits": self.hits,
            "misses": self.misses,
            "detection_frames": self.detection_frames,
            "held_frames": self.held_frames,
            "out_of_view_frac": self.out_of_view_frac,
            "residual_px": self.residual_summary(),
            "strike_records": [
                {**asdict(r), "beyond_reach": r.beyond_reach, "duration_s": r.duration_s}
                for r in self.strike_records
            ],
        }


class Simulation:
    def __init__(self, cfg: dict, seed: int = 7, allow_strikes: bool = True):
        """allow_strikes=False keeps the robot standing still for the whole
        run, so results measure the tracker alone."""
        self.cfg = cfg
        self.seed = seed
        self.allow_strikes = allow_strikes

        # One independent stream per source of randomness (see sim/rng.py).
        self.streams = RandomStreams.from_seed(seed)
        self.world = SimWorld(cfg, streams=self.streams)
        self.controller = TrackingController(cfg, noise_rng=self.streams.detector_noise)

        self.dt = cfg["sim"]["dt_ms"] / 1000.0
        rcfg = cfg["robot"]
        self.strike_speed = rcfg.get("strike_speed", 5.0)
        self.strike_duration = rcfg.get("strike_duration_s", 1.0)
        # Farthest a ball that stood still could be and still be hit by one
        # charge. See _ball_within_reach for how the robot uses it.
        self.reach_m = (self.strike_speed * self.strike_duration
                        + self.world.strike_contact_limit())

        self.t = 0.0
        self.frame = 0
        self.status = TRACKING
        self.stats = RunStats(seed=seed)
        self._strike_timer = 0.0
        self._strike: Optional[StrikeRecord] = None

    @property
    def striking(self) -> bool:
        return self._strike_timer > 0

    def step(self) -> StepInfo:
        world, dt, stats = self.world, self.dt, self.stats

        world.step(dt, striking=self.striking)
        self.t += dt
        result = self.controller.step(self.t, dt, world.tracking_frame, world)

        stats.frames += 1
        if result.detection_center is not None:
            stats.detection_frames += 1
        if not np.isnan(result.residual_px):
            stats.residual_samples.append(result.residual_px)

        if self.allow_strikes and result.confident and self._strike_timer <= 0:
            reachable, est_dist = self._ball_within_reach()
            if reachable:
                self._begin_strike(result, est_dist)
            else:
                stats.held_frames += 1  # confident, but a charge could not get there

        striking = self._strike_timer > 0
        finished = None
        if striking:
            if result.predicted_world is not None:
                world.strike_toward((result.predicted_world[0], result.predicted_world[1]),
                                    dt, self.strike_speed)
            self._strike_timer -= dt
            self._strike.closest_m = min(self._strike.closest_m, world.robot_ball_distance())
            # Contact is checked on every step of the charge: the robot can
            # reach the ball mid-charge and then lose it again before the
            # charge ends. Per-step movement (<= (strike_speed + ball speed) * dt,
            # ~6.5 cm) is far below the contact limit, so nothing tunnels.
            if world.strike_connected():
                finished = self._finish_strike(hit=True)
            elif self._strike_timer <= 0:
                finished = self._finish_strike(hit=False)
        elif result.predicted_world is not None:
            # Tracking: turn in place toward where the forward model says the
            # ball will be, keeping it in the middle of the next frames.
            world.turn_robot_toward((result.predicted_world[0], result.predicted_world[1]), dt)

        info = StepInfo(frame=self.frame, t=self.t, result=result, status=self.status,
                        striking=striking, strike_finished=finished)
        self.frame += 1
        return info

    def run(self, frames: int) -> RunStats:
        for _ in range(frames):
            self.step()
        return self.stats

    # -- strike state machine ---------------------------------------------
    def _estimated_ball_distance(self, t: float) -> float:
        """Robot -> ball distance at time t, as the robot's filter believes it
        (no ground truth: the decision to strike must be the robot's own)."""
        est = self.controller.estimate_at(t)
        if est is None:
            return math.inf
        rx, ry = self.world.robot.position
        return math.hypot(est[0] - rx, est[1] - ry)

    def _ball_within_reach(self) -> tuple[bool, float]:
        """Can a charge that starts now reach the ball? Returns (ok, the
        estimated distance now).

        The test: if the ball keeps its estimated velocity, is where it will be
        when the charge window ends within reach? Because the ball is slower
        than the charge, that is exactly "a straight-line intercept exists": a
        ball that is reachable at any moment in the window is still reachable
        at its end. Unlike a plain distance check, this skips a ball inside
        reach that is running away, and allows one just outside it that is
        coming closer."""
        d_end = self._estimated_ball_distance(self.t + self.strike_duration)
        return d_end <= self.reach_m, self._estimated_ball_distance(self.t)

    def _begin_strike(self, result: FrameResult, est_dist: float) -> None:
        self.status = STRIKE
        self._strike_timer = self.strike_duration
        self.stats.strikes += 1
        self._strike = StrikeRecord(
            index=self.stats.strikes,
            start_frame=self.frame,
            start_t=self.t,
            start_dist_m=self.world.robot_ball_distance(),
            start_est_dist_m=est_dist,
            start_residual_px=result.residual_px,
            reach_m=self.reach_m,
            contact_limit_m=self.world.strike_contact_limit(),
        )

    def _finish_strike(self, hit: bool) -> StrikeRecord:
        world, stats = self.world, self.stats
        rec = self._strike
        rec.end_frame = self.frame
        rec.end_t = self.t
        rec.end_dist_m = world.robot_ball_distance()
        rec.hit = hit
        self._strike_timer = 0.0  # a hit ends the charge early
        if rec.hit:
            stats.hits += 1
        else:
            stats.misses += 1
        stats.strike_records.append(rec)
        self._strike = None

        world.respawn_after_strike()
        self.controller.reset()
        self.status = TRACKING
        return rec


def run_headless(cfg: dict, seed: int, frames: int, allow_strikes: bool = True) -> RunStats:
    """Run one episode with no rendering and return its statistics."""
    return Simulation(cfg, seed=seed, allow_strikes=allow_strikes).run(frames)
