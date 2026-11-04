"""Independent random streams, one per source of randomness.

Every stream is derived from the run seed *and its own name*, so drawing from
one stream never shifts another. Concretely: a camera that sees the ball in
fewer frames (fewer detector-noise draws), or a robot that strikes at a
different moment (a respawn at a different time), cannot change the random
numbers the ball uses to choose its waypoints.

Adding a new stream later does not change existing ones, because each is
seeded by name rather than by position. Seeding with a str goes through
SHA-512 inside `random.Random`, so it is identical on every platform and
unaffected by PYTHONHASHSEED.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class RandomStreams:
    ball_motion: random.Random     # initial heading + waypoints while the ball cruises
    respawn: random.Random         # where and how the ball reappears after a strike
    detector_noise: random.Random  # pixel noise on simulated detections

    @classmethod
    def from_seed(cls, seed: int | None = None) -> "RandomStreams":
        """Derive all streams from one run seed (None -> a fresh random seed)."""
        if seed is None:
            seed = random.SystemRandom().randrange(2 ** 63)
        return cls(**{f.name: random.Random(f"{seed}/{f.name}") for f in fields(cls)})
