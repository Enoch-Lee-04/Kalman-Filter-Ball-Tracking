"""Constant-velocity Kalman filter in image space.

State: [x, y, vx, vy] in pixels and pixels/second.

Two distinct operations:
  * update(z, dt): standard predict-then-correct using a delayed measurement.
  * predict_ahead(horizon): extrapolate the current estimate `horizon` seconds
    into the future WITHOUT mutating state. This is the forward model output --
    where the robot should aim so that by the time its action lands (after the
    sense+think+move latency) it meets the ball.

Written from scratch (no filterpy) so the math is fully visible for learning
and interviews.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class KalmanCV:
    def __init__(
        self,
        process_noise: float,
        measurement_noise: float,
        initial_velocity_var: float,
    ):
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.initial_velocity_var = float(initial_velocity_var)

        self.x: Optional[np.ndarray] = None  # state (4,)
        self.P = np.eye(4)                    # covariance
        # measurement matrix: we observe position only
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.R = np.eye(2) * self.r

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def reset(self) -> None:
        self.x = None
        self.P = np.eye(4)

    def _F(self, dt: float) -> np.ndarray:
        return np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1,  0],
                         [0, 0, 0,  1]], dtype=float)

    def _Q(self, dt: float) -> np.ndarray:
        # piecewise white-noise acceleration model
        q = self.q
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        return q * np.array([
            [dt4 / 4, 0,       dt3 / 2, 0],
            [0,       dt4 / 4, 0,       dt3 / 2],
            [dt3 / 2, 0,       dt2,     0],
            [0,       dt3 / 2, 0,       dt2],
        ], dtype=float)

    def update(self, z: np.ndarray, dt: float) -> None:
        """Advance by dt and fold in measurement z = [x, y]."""
        if self.x is None:
            # bootstrap: position from measurement, velocity unknown (large var)
            self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=float)
            self.P = np.diag([self.r, self.r,
                              self.initial_velocity_var, self.initial_velocity_var])
            return

        F = self._F(dt)
        # predict
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

        # correct
        y = z - self.H @ self.x                  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def predict_ahead(self, horizon: float) -> Optional[np.ndarray]:
        """Where will the ball be `horizon` seconds from now? Returns [x, y]."""
        if self.x is None:
            return None
        F = self._F(horizon)
        future = F @ self.x
        return future[:2]

    def position_velocity(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.x is None:
            return None
        return self.x[:2].copy(), self.x[2:].copy()
