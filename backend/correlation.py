"""Rolling cross-metric correlation-break detection, per city.

Different technique from anomaly.py on purpose: z-score flags a single
metric drifting from its own history; this flags when the *relationship*
between two metrics that normally move together stops holding, even if
neither metric alone looks unusual. Plain Pearson correlation + ordinary
least-squares regression -- both textbook statistics, implemented directly
in Python (a handful of readings per city; no need for numpy here either).

Two pairs are tracked, chosen for having a real meteorological
relationship rather than being arbitrary: pressure tends to correlate with
temperature over short windows in many climates, and humidity/cloud cover
are almost always positively related. A break in either is a genuinely
different observation than "temperature is unusually high."
"""
from __future__ import annotations

import statistics
from collections import defaultdict, deque

from .signals import Signal, correlation_break_to_signal

# Metric pairs worth tracking, and the minimum |r| for an established
# relationship to even be worth checking for a break -- a pair with no
# real historical correlation has no "usual relationship" to break.
TRACKED_PAIRS = [
    ("pressure_hpa", "temperature_c"),
    ("humidity_pct", "cloud_cover_pct"),
]
_MIN_CORRELATION = 0.5
_RESIDUAL_Z_THRESHOLD = 2.5


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def _linreg(xs: list[float], ys: list[float]) -> tuple[float, float]:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0.0
    intercept = my - slope * mx
    return slope, intercept


class CorrelationTracker:
    """Tracks one (metric_a, metric_b) pair's rolling relationship per
    city, and flags when the newest reading's residual from that fitted
    relationship is unusually large -- i.e. the two metrics stopped moving
    together the way they normally do for this city."""

    def __init__(self, metric_a: str, metric_b: str, window: int = 20):
        self._a = metric_a
        self._b = metric_b
        self._window = window
        self._history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def evaluate(self, city: str, reading: dict) -> Signal | None:
        series = self._history[city]
        series.append((reading[self._a], reading[self._b]))
        if len(series) < 8:
            return None

        xs = [p[0] for p in series]
        ys = [p[1] for p in series]
        r = _pearson(xs, ys)
        if abs(r) < _MIN_CORRELATION:
            return None  # no established relationship here to break

        slope, intercept = _linreg(xs, ys)
        residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
        spread = statistics.pstdev(residuals) or 1e-6
        z = abs(residuals[-1]) / spread

        if z >= _RESIDUAL_Z_THRESHOLD:
            return correlation_break_to_signal(city, self._a, self._b, r, z)
        return None
