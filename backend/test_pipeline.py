"""Local smoke test for the anomaly detector + vector store.

This bypasses the network client entirely (generates synthetic-but-realistic
readings with one injected temperature spike) so it can run anywhere,
including sandboxes without outbound access to api.open-meteo.com. It proves
the detection + similarity-search logic is correct independent of the live
API call, which is a thin, separately-testable adapter (see weather_client.py).

Uses two cities on purpose: the similarity search is only useful if it finds
matches *across* cities (see the "same-city dist=0" bug documented in
vector_store.py and the README) -- a single-city test would pass even if that
exclusion were broken, since it wouldn't have a second city to wrongly match
against in the first place.

Run: python -m backend.test_pipeline
"""
from __future__ import annotations

import random
import shutil
from dataclasses import asdict

from .air_quality_client import eaqi_band
from .anomaly import AnomalyDetector
from .correlation import CorrelationTracker
from .forecast import ForecastTracker
from .regime import RegimeTracker, _label_centroid, kmeans
from .signals import air_quality_to_signal, anomaly_to_signal, composite_score, similarity_to_signal
from .vector_store import WeatherVectorStore, closeness_label, notable_gaps
from .weather_client import WeatherReading


def synthetic_readings(city: str, n: int = 25, spike_at: int | None = None, base_temp: float = 22.0):
    for i in range(n):
        temp = base_temp + random.uniform(-1.5, 1.5)
        if spike_at is not None and i == spike_at:
            temp += 12  # inject an obvious anomaly
        yield WeatherReading(
            city=city, latitude=0, longitude=0,
            temperature_c=round(temp, 1),
            humidity_pct=round(50 + random.uniform(-5, 5), 1),
            wind_speed_kmh=round(10 + random.uniform(-3, 3), 1),
            pressure_hpa=round(1013 + random.uniform(-2, 2), 1),
            cloud_cover_pct=round(40 + random.uniform(-10, 10), 1),
            timestamp=f"2026-07-01T{10 + i // 60:02d}:{i % 60:02d}",
        )


def test_closeness_and_gaps():
    """closeness_label/notable_gaps are the fix for the narrator overclaiming
    similarity (see README "Honest notes"): it once called a distance-0.24
    match "quase idênticas" despite an in-prompt instruction not to. These
    are now computed in code so the LLM can't misjudge the threshold, so
    it's worth pinning their behavior directly rather than only indirectly
    through the narrator (which needs a live API key to test end-to-end)."""
    assert closeness_label(0.0) == "praticamente idênticas"
    assert closeness_label(0.24) != "praticamente idênticas"
    assert "diferenças reais" in closeness_label(0.24)
    assert "sem grande semelhança" in closeness_label(1.0)

    # Same real-world pair that prompted this fix: Tokyo's calm 3.7 km/h
    # wind vs Sydney's 22 km/h -- a big real gap that a low aggregate
    # distance was hiding.
    tokyo = WeatherReading(
        city="Tokyo", latitude=0, longitude=0,
        temperature_c=21.6, humidity_pct=94.0, wind_speed_kmh=3.7,
        pressure_hpa=1009.8, cloud_cover_pct=84.0, timestamp="2026-07-01T12:00",
    )
    sydney_match = {
        "city": "Sydney", "temperature_c": 15.9, "humidity_pct": 81.0,
        "wind_speed_kmh": 22.0, "pressure_hpa": 1010.4, "cloud_cover_pct": 95.0,
    }
    gaps = notable_gaps(tokyo, sydney_match)
    assert any("vento" in g for g in gaps), f"expected a wind gap to be flagged, got: {gaps}"
    print(f"[CLOSENESS]  Tokyo/Sydney example -> notable_gaps: {gaps}")
    print("OK: closeness_label and notable_gaps behave as expected.\n")


def test_signals_and_composite_score():
    """signals.py is the v2 groundwork: every detector's output gets
    converted into a common `Signal` shape before the narrator or the
    composite index ever see it. Pin the conversions and the scoring
    formula directly, since nothing else exercises them end-to-end without
    a live ANTHROPIC_API_KEY."""
    borderline_anomaly = {"metric": "temperature_c", "value": 30.0, "z_score": 2.5, "baseline_mean": 20.0}
    wild_anomaly = {"metric": "temperature_c", "value": 40.0, "z_score": 9.0, "baseline_mean": 20.0}

    borderline_sig = anomaly_to_signal(borderline_anomaly, "São Paulo")
    wild_sig = anomaly_to_signal(wild_anomaly, "São Paulo")
    assert borderline_sig.type == "anomaly"
    assert borderline_sig.severity == 0.0, "z-score right at the detector's own threshold should score 0 severity"
    assert wild_sig.severity == 1.0, "z-score far past threshold should saturate at 1.0, not grow unbounded"

    close_match = {"city": "Cairo", "distance": 0.049}
    far_match = {"city": "London", "distance": 0.9}
    close_sig = similarity_to_signal("Tokyo", close_match, closeness_label(0.049), [])
    far_sig = similarity_to_signal("Sydney", far_match, closeness_label(0.9), ["vento: 15.0km/h de diferença"])
    assert close_sig.severity > far_sig.severity, "a closer vector match must score higher severity than a distant one"
    assert far_sig.evidence["notable_gaps"], "notable_gaps must survive the conversion into the signal's evidence"

    assert composite_score([]) == 0.0, "no signals firing must score exactly 0, not some non-zero floor"
    quiet_score = composite_score([borderline_sig])
    loud_score = composite_score([wild_sig, close_sig])
    assert quiet_score < loud_score, "more/stronger simultaneous signals must push the composite score up"
    assert 0.0 <= quiet_score <= 100.0 and 0.0 <= loud_score <= 100.0, "composite score must always stay in 0-100"

    print(f"[SIGNALS]  quiet_score={quiet_score}  loud_score={loud_score}")
    print("OK: signal conversion and composite scoring behave as expected.\n")


def test_air_quality():
    """eaqi_band's thresholds are a fixed lookup, not a computed formula --
    worth pinning directly. air_quality_to_signal should stay quiet (0
    severity) right at the "moderada" boundary and saturate at 1.0 once AQI
    is deep into "muito ruim", mirroring how anomaly_to_signal saturates."""
    assert eaqi_band(10) == "boa"
    assert eaqi_band(40) == "razoável"
    assert eaqi_band(60) == "moderada"
    assert eaqi_band(100) == "muito ruim"
    assert eaqi_band(150) == "extremamente ruim"

    calm_sig = air_quality_to_signal("Cairo", european_aqi=40, pm2_5=12.0, band="razoável")
    bad_sig = air_quality_to_signal("Cairo", european_aqi=100, pm2_5=80.0, band="muito ruim")
    assert calm_sig.severity == 0.0, "AQI right at the moderada boundary should score 0 severity"
    assert bad_sig.severity == 1.0, "AQI at the top of the defined band range should saturate at 1.0"
    assert calm_sig.type == "air_quality"
    print(f"[AIR QUALITY]  calm_severity={calm_sig.severity}  bad_severity={bad_sig.severity}")
    print("OK: eaqi_band and air_quality_to_signal behave as expected.\n")


def test_correlation_break():
    """Feed a city a strongly-correlated pressure/temperature series (a
    straight line) so the tracker establishes a relationship, then inject
    one reading that breaks it. The break must be strong enough to produce
    a large residual but not so extreme that it drags the *window's own*
    correlation below the tracker's _MIN_CORRELATION threshold -- r is
    recomputed over the whole window including the breaking point, so an
    overly wild single outlier can crush |r| below 0.5 and make the
    detector correctly decline to fire (there's no "established
    relationship" left once the outlier is folded into r itself). An
    18-point line with a -15 offset on the 19th point keeps r ~0.8 while
    still producing a residual z-score well past the 2.5 trigger."""
    tracker = CorrelationTracker("pressure_hpa", "temperature_c", window=20)
    fired = False
    for i in range(18):
        pressure = 1000 + i  # perfectly linear vs. temperature below
        temp = 20 + i  # pressure_hpa and temperature_c move in lockstep
        sig = tracker.evaluate("Testville", {"pressure_hpa": pressure, "temperature_c": temp})
        if sig:
            fired = True
    assert not fired, "a perfectly linear, unbroken relationship must never fire a break"

    # Now break it: pressure keeps climbing on trend but temperature falls
    # well short of what the established line would predict.
    break_sig = tracker.evaluate("Testville", {"pressure_hpa": 1018, "temperature_c": 23})
    assert break_sig is not None, "a sharp divergence from the established linear relationship should fire"
    assert break_sig.type == "correlation_break"
    assert break_sig.evidence["metric_a"] == "pressure_hpa"
    assert abs(break_sig.evidence["correlation"]) >= 0.5, "the break itself must not have crushed the window's own correlation below the tracker's threshold"
    print(f"[CORRELATION]  break_severity={break_sig.severity}  r={break_sig.evidence['correlation']}")
    print("OK: correlation-break detection stays quiet on a stable relationship and fires on a real break.\n")


def test_forecast_miss():
    """Feed a city a flat, stable metric so the smoothed forecast tracks it
    closely, then inject a sharp jump. The jump should fire a forecast_miss
    signal once there's enough residual history to judge it against.

    Uses a small deterministic alternating jitter (+/-0.1) on every metric
    rather than an exactly-repeating constant: feeding the exact same float
    dozens of times hits a genuine floating-point edge case in the
    smoothing formula (`alpha*value + (1-alpha)*level` doesn't reproduce
    `value` bit-for-bit) that makes the residual history collapse toward a
    near-zero spread, which can make even a ~1e-13 rounding artifact look
    like a z-score spike. Real sensor data always has some jitter, so this
    edge case doesn't come up in production -- but a literal constant in a
    test does hit it, so the jitter here is what real data would look
    like, not a workaround for a production bug."""
    tracker = ForecastTracker()
    fired = False
    for i in range(15):
        jitter = 0.1 if i % 2 == 0 else -0.1
        signals = tracker.evaluate("Testville", {
            "temperature_c": 20.0 + jitter, "humidity_pct": 50.0 + jitter,
            "wind_speed_kmh": 10.0 + jitter, "pressure_hpa": 1013.0 + jitter,
            "cloud_cover_pct": 40.0 + jitter,
        })
        if signals:
            fired = True
    assert not fired, "a stable, flat metric must never fire a forecast miss"

    signals = tracker.evaluate("Testville", {
        "temperature_c": 45.0,  # sharp jump the smoothed level would not predict
        "humidity_pct": 50.0, "wind_speed_kmh": 10.0,
        "pressure_hpa": 1013.0, "cloud_cover_pct": 40.0,
    })
    assert signals, "a sharp jump away from the smoothed trend should fire a forecast_miss"
    miss = signals[0]
    assert miss.type == "forecast_miss"
    assert miss.evidence["metric"] == "temperature_c"
    print(f"[FORECAST]  actual={miss.evidence['actual']}  predicted={miss.evidence['predicted']}")
    print("OK: forecast-miss detection stays quiet on a flat trend and fires on a sharp jump.\n")


def test_regime_clustering():
    """kmeans on two obviously-separated synthetic clusters should recover
    two distinct centroids close to the true cluster centers. Then
    RegimeTracker.assign_one should report a regime_change when a city's
    vector moves from one cluster to the other between calls, and should
    NOT report anything on a city's very first observation."""
    cold_cluster = [[0.1, 0.5, 0.2, 0.5, 0.3] for _ in range(10)]
    hot_cluster = [[0.9, 0.5, 0.2, 0.5, 0.3] for _ in range(10)]
    centroids = kmeans(cold_cluster + hot_cluster, k=2, seed=1)
    temps = sorted(c[0] for c in centroids)
    assert temps[0] < 0.3 and temps[1] > 0.7, f"expected two well-separated centroids, got {centroids}"
    assert _label_centroid([0.9, 0.5, 0.2, 0.5, 0.3]) != _label_centroid([0.1, 0.5, 0.2, 0.5, 0.3])

    class _FakeStore:
        """assign_one only needs `all_vectors()`; a minimal stand-in avoids
        pulling the real WeatherVectorStore (and its WeatherReading
        objects) into a test about clustering, not embeddings."""
        def __init__(self, pairs):
            self._pairs = pairs

        def all_vectors(self):
            return self._pairs

    pairs = [(v, {}) for v in cold_cluster + hot_cluster]
    store = _FakeStore(pairs)
    tracker = RegimeTracker(k=2, refresh_every=100)

    first = tracker.assign_one(store, "Testville", [0.1, 0.5, 0.2, 0.5, 0.3])
    assert first is None, "a city's first-ever regime observation must not fire a change"

    same = tracker.assign_one(store, "Testville", [0.12, 0.5, 0.2, 0.5, 0.3])
    assert same is None, "staying in the same regime must not fire a change"

    changed = tracker.assign_one(store, "Testville", [0.9, 0.5, 0.2, 0.5, 0.3])
    assert changed is not None, "moving to the other cluster must fire a regime_change"
    assert changed.type == "regime_change"
    print(f"[REGIME]  {changed.evidence['old_regime']} -> {changed.evidence['new_regime']}")
    print("OK: k-means separates clusters and RegimeTracker detects real regime changes.\n")


def main():
    test_closeness_and_gaps()
    test_signals_and_composite_score()
    test_air_quality()
    test_correlation_break()
    test_forecast_miss()
    test_regime_clustering()
    persist_dir = "chroma_data_test"
    shutil.rmtree(persist_dir, ignore_errors=True)  # fresh run every time

    detector = AnomalyDetector(window=20, z_threshold=2.5)
    store = WeatherVectorStore(persist_dir=persist_dir)

    found_anomaly = False
    found_cross_city_similar = False
    found_same_city_leak = False

    # CityA gets the anomaly spike. CityB is generated around the same
    # baseline (22C) so it should be the nearest cross-city neighbor for
    # CityA's non-spike readings.
    for r in synthetic_readings("CityA", spike_at=20):
        rd = asdict(r)
        anomalies = detector.evaluate("CityA", rd)
        doc_id = store.add(r)
        similar = store.nearest_similar(r, exclude_id=doc_id)
        if anomalies:
            found_anomaly = True
            print(f"[ANOMALY]  {r.timestamp}  temp={r.temperature_c}  ->  {anomalies}")
        if any(s["city"] == "CityA" for s in similar):
            found_same_city_leak = True
        if any(s["city"] == "CityB" for s in similar):
            found_cross_city_similar = True

    for r in synthetic_readings("CityB", base_temp=22.0):
        rd = asdict(r)
        detector.evaluate("CityB", rd)
        doc_id = store.add(r)
        similar = store.nearest_similar(r, exclude_id=doc_id)
        if any(s["city"] == "CityA" for s in similar):
            found_cross_city_similar = True
        if any(s["city"] == "CityB" for s in similar):
            found_same_city_leak = True

    assert found_anomaly, "expected the injected temperature spike to be flagged"
    assert found_cross_city_similar, "expected at least one genuine cross-city nearest-neighbor match"
    assert not found_same_city_leak, "a city's own readings must never appear in its own similar_patterns"
    print("\nOK: anomaly detection works, and similarity search only ever surfaces other cities.")


if __name__ == "__main__":
    main()
