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

from .anomaly import AnomalyDetector
from .signals import anomaly_to_signal, composite_score, similarity_to_signal
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


def main():
    test_closeness_and_gaps()
    test_signals_and_composite_score()
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
