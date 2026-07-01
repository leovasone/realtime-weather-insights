"""Client for the free, no-auth Open-Meteo current-weather API.

Kept as its own small class -- rather than inline requests scattered through
the app -- so it can be swapped for a mock/fixture client in tests without
touching the rest of the pipeline. Same pluggable-adapter pattern used for
the OCR engine in the document-extraction project.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass
class WeatherReading:
    city: str
    latitude: float
    longitude: float
    temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float
    pressure_hpa: float
    cloud_cover_pct: float
    timestamp: str


class OpenMeteoClient:
    """Thin async wrapper around the Open-Meteo current-weather endpoint."""

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    async def fetch(self, city: str, latitude: float, longitude: float) -> WeatherReading:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,pressure_msl,cloud_cover",
            "timezone": "auto",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(OPEN_METEO_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        current = data["current"]
        return WeatherReading(
            city=city,
            latitude=latitude,
            longitude=longitude,
            temperature_c=current["temperature_2m"],
            humidity_pct=current["relative_humidity_2m"],
            wind_speed_kmh=current["wind_speed_10m"],
            pressure_hpa=current["pressure_msl"],
            cloud_cover_pct=current["cloud_cover"],
            timestamp=current["time"],
        )
