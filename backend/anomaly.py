"""Rolling z-score anomaly detection, per city and per metric.

Simple and fully explainable on purpose: for each metric we keep a sliding
window of recent values, compute mean/stdev, and flag a new value as
anomalous if it's more than `z_threshold` standard deviations from that
rolling baseline. No black-box model required for a demo whose point is to
show the pipeline (ingest -> detect -> explain), not to win a forecasting
competition.
"""
from __future__ import annotations

import statistics
from collections import defaultdict, deque

METRICS = ["temperature_c", "humidity_pct", "wind_speed_kmh", "pressure_hpa", "cloud_cover_pct"]


class AnomalyDetector:
    def __init__(self, window: int = 20, z_threshold: float = 2.5):
        self._window = window
        self._z_threshold = z_threshold
        self._history: dict[str, dict[str, deque]] = defaultdict(
            lambda: {m: deque(maxlen=window) for m in METRICS}
        )

    def evaluate(self, city: str, reading: dict) -> list[dict]:
        """Update history for `city` and return any anomalies found in this
        reading (empty list if none)."""
        anomalies = []
        hist = self._history[city]
        for metric in METRICS:
            value = reading[metric]
            series = hist[metric]
            if len(series) >= 5:
                mean = statistics.fmean(series)
                stdev = statistics.pstdev(series) or 1e-6
                z = (value - mean) / stdev
                if abs(z) >= self._z_threshold:
                    anomalies.append({
                        "metric": metric,
                        "value": value,
                        "z_score": round(z, 2),
                        "baseline_mean": round(mean, 2),
                    })
            series.append(value)
        return anomalies
