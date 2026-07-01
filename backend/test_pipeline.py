"""Local smoke test for the anomaly detector + vector store.

This bypasses the network client entirely (generates synthetic-but-realistic
readings with one injected temperature spike) so it can run anywhere,
including sandboxes without outbound access to api.open-meteo.com. It proves
the detection + similarity-search logic is correct independent of the live
API call, which is a thin, separately-testable adapter (see weather_client.py).

Run: python -m backend.test_pipeline
"""
from __future__ import annotations

import random
import shutil
from dataclasses import asdict

from .anomaly import AnomalyDetector
from .vector_store import WeatherVectorStore
from .weather_client import WeatherReading


def synthetic_readings(city: str, n: int = 25, spike_at: int = 20):
    base_temp = 22.0
    for i in range(n):
        temp = base_temp + random.uniform(-1.5, 1.5)
        if i == spike_at:
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


def main():
    persist_dir = "chroma_data_test"
    shutil.rmtree(persist_dir, ignore_errors=True)  # fresh run every time

    detector = AnomalyDetector(window=20, z_threshold=2.5)
    store = WeatherVectorStore(persist_dir=persist_dir)

    found_anomaly = False
    found_similar = False
    for r in synthetic_readings("TestCity"):
        rd = asdict(r)
        anomalies = detector.evaluate("TestCity", rd)
        doc_id = store.add(r)
        similar = store.nearest_similar(r, exclude_id=doc_id)
        if anomalies:
            found_anomaly = True
            print(f"[ANOMALY]  {r.timestamp}  temp={r.temperature_c}  ->  {anomalies}")
        if similar:
            found_similar = True

    assert found_anomaly, "expected the injected temperature spike to be flagged"
    assert found_similar, "expected at least one nearest-neighbor match once history built up"
    print("\nOK: anomaly detection + vector similarity search both working correctly.")


if __name__ == "__main__":
    main()
