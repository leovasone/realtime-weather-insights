"""Lightweight in-memory vector store for weather-pattern similarity search.

Each incoming reading is embedded as a 5-dimensional numeric vector (temp,
humidity, wind, pressure, cloud cover). On every new reading we brute-force
search for the nearest historical neighbors -- across OTHER cities -- to
surface "this looks like what city X was experiencing on date Y" as an
AI-generated insight.

This started out on top of ChromaDB, but for a dataset this small (a
handful of 5-float vectors per city, growing slowly over time) a full
vector database is unnecessary weight: its Rust bindings pushed memory past
what a free-tier Railway container has available and the process kept
getting OOM-killed and restarted. A plain Python/heapq brute-force search
is O(n) per query, which is irrelevant at this scale (bounded to
`max_size` entries below) and removes an entire dependency plus its
startup cost. Same public interface (`add` / `nearest_similar`), so nothing
else in the app needed to change.
"""
from __future__ import annotations

import heapq
import time
from collections import deque

from .weather_client import WeatherReading


# Fixed, realistic min/max bounds per metric, used only to scale each
# dimension into a comparable 0-1 range before computing distance. Without
# this, raw values sit on wildly different scales -- pressure varies across
# ~80 hPa around 1013, while temperature varies across ~65 degrees, humidity
# and cloud cover across 100 points each. Unscaled, a middling humidity gap
# can outweigh a real temperature gap in the total distance, which produces
# "nearest neighbor" matches that don't actually look similar to a human
# reading the dashboard (see README "Honest notes" for a concrete example).
# These are broad meteorological bounds, not tuned to this dataset -- the
# goal is comparable scale across dimensions, not precision.
_FEATURE_RANGES = {
    "temperature_c": (-20.0, 45.0),
    "humidity_pct": (0.0, 100.0),
    "wind_speed_kmh": (0.0, 100.0),
    "pressure_hpa": (970.0, 1050.0),
    "cloud_cover_pct": (0.0, 100.0),
}


def _vector(r: WeatherReading) -> list[float]:
    raw = [
        ("temperature_c", r.temperature_c),
        ("humidity_pct", r.humidity_pct),
        ("wind_speed_kmh", r.wind_speed_kmh),
        ("pressure_hpa", r.pressure_hpa),
        ("cloud_cover_pct", r.cloud_cover_pct),
    ]
    normalized = []
    for name, value in raw:
        lo, hi = _FEATURE_RANGES[name]
        scaled = (value - lo) / (hi - lo)
        normalized.append(max(0.0, min(1.0, scaled)))
    return normalized


def _l2_distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


class WeatherVectorStore:
    """Bounded in-memory store; `persist_dir` is accepted for interface
    compatibility but unused -- history resets on process restart, which is
    fine for a demo (see README)."""

    def __init__(self, persist_dir: str = "chroma_data", max_size: int = 2000):
        self._max_size = max_size
        self._entries = deque(maxlen=max_size)

    def add(self, reading: WeatherReading) -> str:
        doc_id = f"{reading.city}-{reading.timestamp}-{time.time_ns()}"
        meta = {
            "city": reading.city,
            "timestamp": reading.timestamp,
            "temperature_c": reading.temperature_c,
            "humidity_pct": reading.humidity_pct,
            "wind_speed_kmh": reading.wind_speed_kmh,
            "pressure_hpa": reading.pressure_hpa,
            "cloud_cover_pct": reading.cloud_cover_pct,
        }
        self._entries.append((doc_id, _vector(reading), meta))
        return doc_id

    def nearest_similar(self, reading: WeatherReading, exclude_id: str, k: int = 3):
        # Excluding just `exclude_id` (the reading we just added) is not
        # enough: weather barely changes between 60s polls, so a city's own
        # previous reading is almost always its own nearest neighbor --
        # every insight ends up being "Cairo is similar to Cairo (dist=0)",
        # which is true but tells you nothing. The interesting signal is
        # cross-city similarity, so we exclude the querying city's entire
        # history, not just the one exact reading.
        query = _vector(reading)
        candidates = [
            (dist, meta)
            for doc_id, vec, meta in self._entries
            if doc_id != exclude_id and meta["city"] != reading.city
            for dist in [_l2_distance(query, vec)]
        ]
        if not candidates:
            return []
        nearest = heapq.nsmallest(k, candidates, key=lambda c: c[0])
        return [{**meta, "distance": round(dist, 3)} for dist, meta in nearest]
