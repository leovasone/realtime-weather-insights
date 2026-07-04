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
    "air_quality": 0.8,  # a real operational/health concern, weighted close to anomaly
    "correlation_break": 0.6,  # an interesting pattern, not inherently alarming on its own
    "forecast_miss": 0.8,  # a real divergence from expected trend, similar weight to anomaly
    "regime_change": 0.3,  # informational/categorical, the lightest of the v2 signal types
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


def air_quality_to_signal(city: str, european_aqi: float, pm2_5: float, band: str) -> Signal:
    """Air quality reading (see air_quality_client.py) -> unified signal.

    Only meaningful once European AQI crosses into "moderada" (>40) --
    below that, air quality isn't operationally interesting. Severity
    saturates at 1.0 by AQI 100 ("muito ruim"), the top of the defined
    EAQI band range.
    """
    severity = round(min(1.0, max(0.0, (european_aqi - 40) / 60)), 2)
    return Signal(
        type="air_quality",
        city=city,
        severity=severity,
        summary=f"European AQI {european_aqi} ({band}), PM2.5 {pm2_5} µg/m³",
        evidence={"european_aqi": european_aqi, "pm2_5": pm2_5, "band": band},
    )


def correlation_break_to_signal(city: str, metric_a: str, metric_b: str, correlation: float, residual_z: float) -> Signal:
    """Cross-metric correlation break (see correlation.py) -> unified
    signal. Severity uses the same saturation curve as anomaly_to_signal
    -- both are residual z-scores past a 2.5 trigger threshold, just of
    different quantities (a raw value vs. a relationship's residual)."""
    severity = round(min(1.0, max(0.0, (residual_z - 2.5) / 5.0)), 2)
    return Signal(
        type="correlation_break",
        city=city,
        severity=severity,
        summary=f"{metric_a}/{metric_b} relationship broke (usual r={correlation:.2f}, residual z={residual_z:.2f})",
        evidence={
            "metric_a": metric_a,
            "metric_b": metric_b,
            "correlation": round(correlation, 2),
            "residual_z": round(residual_z, 2),
        },
    )


def forecast_miss_to_signal(city: str, metric: str, actual: float, predicted: float, residual_z: float) -> Signal:
    """Forecast miss (see forecast.py) -> unified signal. Same severity
    curve as the other residual-based signals (anomaly, correlation
    break) for consistency across the composite score."""
    severity = round(min(1.0, max(0.0, (residual_z - 2.5) / 5.0)), 2)
    return Signal(
        type="forecast_miss",
        city=city,
        severity=severity,
        summary=f"{metric} = {actual}, forecast expected ~{predicted:.1f} (z={residual_z:.2f})",
        evidence={
            "metric": metric,
            "actual": actual,
            "predicted": round(predicted, 1),
            "residual_z": round(residual_z, 2),
        },
    )


def regime_to_signal(city: str, old_label: str, new_label: str) -> Signal:
    """Climate-regime change (see regime.py) -> unified signal. Severity
    is a fixed 0.5, not a computed one: a regime shift is a categorical
    event (which cluster a city belongs to), not a quantity with a natural
    magnitude the way a z-score or correlation residual has -- there's no
    honest way to say one regime change is "more severe" than another."""
    return Signal(
        type="regime_change",
        city=city,
        severity=0.5,
        summary=f"shifted from '{old_label}' to '{new_label}' regime",
        evidence={"old_regime": old_label, "new_regime": new_label},
    )
