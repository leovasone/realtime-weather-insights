"""Client for the free, no-auth Open-Meteo Air Quality API.

Same pluggable-adapter shape as weather_client.py: an isolated class, easy
to swap or mock, that the rest of the pipeline doesn't need to know the
internals of. A separate client (not folded into weather_client.py)
because it's a genuinely different Open-Meteo API host and could fail or
be swapped independently of the core weather fetch.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


@dataclass
class AirQualityReading:
    city: str
    european_aqi: float
    pm2_5: float
    pm10: float


# European Air Quality Index bands -- fixed reference thresholds from the
# public EAQI scale, not tuned to this dataset. "ruim" (Poor, >60) is
# where this module starts treating it as signal-worthy; the WHO/EU
# reference itself, not an arbitrary cutoff picked for this demo.
_EAQI_BANDS = [
    (20, "boa"),
    (40, "razoável"),
    (60, "moderada"),
    (80, "ruim"),
    (100, "muito ruim"),
]


def eaqi_band(aqi: float) -> str:
    for threshold, label in _EAQI_BANDS:
        if aqi <= threshold:
            return label
    return "extremamente ruim"


class OpenMeteoAirQualityClient:
    """Thin async wrapper around the Open-Meteo air-quality endpoint."""

    def __init__(self, timeout: float = 10.0):
        self._timeout = timeout

    async def fetch(self, city: str, latitude: float, longitude: float) -> AirQualityReading:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "pm2_5,pm10,european_aqi",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(AIR_QUALITY_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        current = data["current"]
        return AirQualityReading(
            city=city,
            european_aqi=current["european_aqi"],
            pm2_5=current["pm2_5"],
            pm10=current["pm10"],
        )
