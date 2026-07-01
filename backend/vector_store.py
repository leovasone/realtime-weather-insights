"""Vector store wrapper around ChromaDB for weather-pattern similarity search.

Each incoming reading is embedded as a 5-dimensional numeric vector (temp,
humidity, wind, pressure, cloud cover) and stored in a persistent ChromaDB
collection. On every new reading we query for the nearest historical
neighbors -- across all cities -- to surface "this looks like what city X
was experiencing on date Y" as an AI-generated insight, without needing a
trained embedding model for a handful of well-understood numeric features.
"""
from __future__ import annotations

import time

import chromadb

from .weather_client import WeatherReading


def _vector(r: WeatherReading) -> list[float]:
    return [
        r.temperature_c,
        r.humidity_pct,
        r.wind_speed_kmh,
        r.pressure_hpa,
        r.cloud_cover_pct,
    ]


class WeatherVectorStore:
    def __init__(self, persist_dir: str = "chroma_data"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="weather_readings",
            metadata={"hnsw:space": "l2"},
        )

    def add(self, reading: WeatherReading) -> str:
        doc_id = f"{reading.city}-{reading.timestamp}-{time.time_ns()}"
        self._collection.add(
            ids=[doc_id],
            embeddings=[_vector(reading)],
            metadatas=[{
                "city": reading.city,
                "timestamp": reading.timestamp,
                "temperature_c": reading.temperature_c,
                "humidity_pct": reading.humidity_pct,
                "wind_speed_kmh": reading.wind_speed_kmh,
                "pressure_hpa": reading.pressure_hpa,
                "cloud_cover_pct": reading.cloud_cover_pct,
            }],
        )
        return doc_id

    def nearest_similar(self, reading: WeatherReading, exclude_id: str, k: int = 3) -> list[dict]:
        """Return up to k historically similar readings (any city), excluding
        the reading that was just inserted."""
        count = self._collection.count()
        if count <= 1:
            return []
        result = self._collection.query(
            query_embeddings=[_vector(reading)],
            n_results=min(k + 1, count),
        )
        matches = []
        for doc_id, meta, dist in zip(
            result["ids"][0], result["metadatas"][0], result["distances"][0]
        ):
            if doc_id == exclude_id:
                continue
            matches.append({**meta, "distance": round(dist, 3)})
        return matches[:k]
