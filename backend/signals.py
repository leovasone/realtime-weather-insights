"""Unified signal schema for the detection pipeline.

Every detector -- anomaly (z-score), similarity (vector search), and every
new source planned for v2 (air quality, cross-metric correlation breaks,
forecast misses, climate-regime changes, climatology/percentile checks,
nearby natural events) -- emits `Signal` objects in this same shape. That
is the point of this module: without a common shape, each new signal
source would need its own bespoke wiring into the narrator prompt and its
own ad-hoc scoring, and the system would grow into a pile of disconnected
features instead of one system that gets more informed over time.

Two things consume this common shape:
  - `composite_score()` below, which turns "however many signals fired for
    a city this cycle" into one explainable 0-100 number (the leaderboard
    ranking, and later the map marker color).
  - `narrator._build_prompt()`, which picks the most interesting signal(s)
    of the cycle across every type, instead of only ever knowing about
    anomalies and similarity matches.

Adding a new signal source later means writing a small `*_to_signal()`
converter (see `anomaly_to_signal` / `similarity_to_signal` below) -- the
composite score and the narrator do not need to change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Signal:
    type: str
    city: str
    severity: float  # normalized 0-1, comparable across signal types
    summary: str  # short factual description; also used as the fallback
    # narrator-prompt line for any signal type that doesn't have bespoke
    # phrasing in narrator._build_prompt() yet
    evidence: dict[str, Any] = field(default_factory=dict)


# Relative weight of each signal type in the composite score. A plain,
# explicit lookup table (not a learned weighting) so the score always
# stays explainable -- any composite number can be decomposed back into
# "which signals fired, and how much each one contributed."
#
# Similarity matches are informative rather than alarming (two cities
# happening to look alike isn't a problem the way an anomaly is), so they
# count for less. New signal types default to a middling weight until
# there's a reason to tune them individually.
_SIGNAL_WEIGHTS = {
    "anomaly": 1.0,
    "similarity": 0.4,
}
_DEFAULT_WEIGHT = 0.6


def composite_score(signals: list[Signal]) -> float:
    """Combine every signal that fired for one city this cycle into a
    single 0-100 score.

    Deliberately a simple weighted sum with smooth saturation, not a
    learned model: a city with many simultaneous signals should approach
    100 without ever exceeding it, but the exact number must always be
    justifiable by naming which signals fired and their severities --
    there's no black box between "what happened" and "the number shown."
    """
    if not signals:
        return 0.0
    total = sum(s.severity * _SIGNAL_WEIGHTS.get(s.type, _DEFAULT_WEIGHT) for s in signals)
    return round(100 * (1 - math.exp(-total)), 1)


def anomaly_to_signal(anomaly: dict, city: str) -> Signal:
    """z-score anomaly (see anomaly.py) -> unified signal.

    Severity is 0 right at the detector's own trigger threshold (z=2.5)
    and saturates at 1.0 by z=7.5, so a single wild outlier doesn't
    dominate the composite score disproportionately more than a cluster
    of moderate anomalies would.
    """
    z = abs(anomaly["z_score"])
    severity = round(min(1.0, max(0.0, (z - 2.5) / 5.0)), 2)
    return Signal(
        type="anomaly",
        city=city,
        severity=severity,
        summary=f"{anomaly['metric']} = {anomaly['value']} (z={anomaly['z_score']}, baseline {anomaly['baseline_mean']})",
        evidence=anomaly,
    )


def similarity_to_signal(city: str, match: dict, closeness: str, gaps: list[str]) -> Signal:
    """Cross-city similarity match (see vector_store.py) -> unified signal.

    Severity is the inverse of normalized vector distance (closer match =
    higher severity), clipped to 0-1 since distance can theoretically
    exceed 1 across five normalized dimensions. `closeness` and `gaps` are
    passed in rather than recomputed here because they depend on the
    original `WeatherReading`, which this function intentionally doesn't
    need to know about -- keeps the conversion itself trivial to test.
    """
    distance = match["distance"]
    severity = round(max(0.0, min(1.0, 1.0 - distance)), 2)
    return Signal(
        type="similarity",
        city=city,
        severity=severity,
        summary=f"resembles {match['city']} (distance {distance})",
        evidence={
            "matches": match["city"],
            "distance": distance,
            "closeness_label": closeness,
            "notable_gaps": gaps,
        },
    )
