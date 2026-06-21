"""Simulated sense -> think -> move latency.

Two mechanisms:
  * DelayLine: the detector measurement the filter receives is stale by
    `sense` ms (the camera/transport delay). We store timestamped samples and
    return the one closest to (now - delay).
  * PredictionLog: a prediction made now targets a position `horizon` ms in the
    future. We stash (target_time, predicted_xy) and, once simulation time
    reaches target_time, pop it to compare against where the ball actually is.
    That comparison is the residual the controller learns from.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


class LatencyPipeline:
    def __init__(self, sense_ms: float, think_ms: float, move_ms: float):
        self.sense = sense_ms / 1000.0
        self.think = think_ms / 1000.0
        self.move = move_ms / 1000.0
        self.horizon = self.sense + self.think + self.move

        self._samples: Deque[Tuple[float, np.ndarray]] = deque(maxlen=512)
        self._predictions: Deque[Tuple[float, np.ndarray]] = deque(maxlen=512)

    # -- sense delay ------------------------------------------------------
    def push_measurement(self, t: float, meas: np.ndarray) -> None:
        self._samples.append((t, meas.copy()))

    def delayed_measurement(self, t: float) -> Optional[Tuple[float, np.ndarray]]:
        """Return (timestamp, measurement) as it would arrive now: the ball as
        it was `sense` seconds ago. None if no sample is old enough yet."""
        target = t - self.sense
        if not self._samples or self._samples[0][0] > target:
            return None
        best = None
        for ts, meas in self._samples:
            if ts <= target:
                best = (ts, meas)
            else:
                break
        return best

    # -- prediction / residual -------------------------------------------
    def log_prediction(self, measurement_time: float, predicted_xy: np.ndarray) -> None:
        """The prediction extrapolates from the (delayed) measurement at
        `measurement_time` by `horizon` seconds, so it is valid at
        measurement_time + horizon of wall-clock time."""
        self._predictions.append((measurement_time + self.horizon, predicted_xy.copy()))

    def due_prediction(self, t: float) -> Optional[np.ndarray]:
        """Pop the oldest prediction whose target time has arrived."""
        if self._predictions and self._predictions[0][0] <= t:
            _, pred = self._predictions.popleft()
            return pred
        return None

    def clear(self) -> None:
        self._samples.clear()
        self._predictions.clear()
