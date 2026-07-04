"""One-step-ahead forecast-miss detection, per city per metric.

A third, deliberately distinct technique from the other detectors: z-score
(anomaly.py) flags a value far from its own recent average; correlation
(correlation.py) flags two metrics no longer moving together; this flags
when a metric diverges from what simple exponential smoothing of its own
recent trend would have predicted -- catching, e.g., a value that's
trending sharply in one direction, not just one that's an outlier against
a flat average.

Single exponential smoothing (no seasonality, no trend component) is a
deliberately simple forecaster -- the point is that "the forecast missed"
stays fully explainable (level, prediction, and residual are all plain
numbers), not that this is a serious forecasting model.
"""
from __future__ import annotations

import statistics
from collections import defaultdict, deque

from .anomaly import METRICS
from .signals import Signal, forecast_miss_to_signal

_ALPHA = 0.3  # smoothing factor: how much weight the newest reading gets
_WINDOW = 20
_RESIDUAL_Z_THRESHOLD = 2.5


class ForecastTracker:
    def __init__(self, alpha: float = _ALPHA, window: int = _WINDOW):
        self._alpha = alpha
        self._level: dict[tuple[str, str], float] = {}
        self._errors: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=window))

    def evaluate(self, city: str, reading: dict) -> list[Signal]:
        """Update the smoothed forecast for every metric of `reading`, and
        return a signal for any metric whose actual value landed far from
        what the smoothed trend predicted. The very first reading for a
        city/metric only initializes the level -- there's no prior
        forecast to have missed yet."""
        signals: list[Signal] = []

        for metric in METRICS:
            key = (city, metric)
            value = reading[metric]

            if key in self._level:
                predicted = self._level[key]
                error = value - predicted
                errs = self._errors[key]

                if len(errs) >= 5:
                    spread = statistics.pstdev(errs) or 1e-6
                    z = abs(error) / spread
                    if z >= _RESIDUAL_Z_THRESHOLD:
                        signals.append(forecast_miss_to_signal(city, metric, value, predicted, z))

                errs.append(error)
                self._level[key] = self._alpha * value + (1 - self._alpha) * predicted
            else:
                self._level[key] = value

        return signals
