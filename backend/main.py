"""FastAPI app: polls real weather data for a fixed set of cities, runs
anomaly detection + vector similarity search on each reading, and streams
the results to connected WebSocket clients in real time.
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import narrator
from .air_quality_client import OpenMeteoAirQualityClient, eaqi_band
from .anomaly import AnomalyDetector
from .cities import CITIES
from .correlation import TRACKED_PAIRS, CorrelationTracker
from .forecast import ForecastTracker
from .regime import RegimeTracker
from .retail_signals import (
    DISCLAIMER as RETAIL_DISCLAIMER,
    AlphaVantageRetailClient,
    ClimateRetailCorrelationTracker,
)
from .signals import (
    Signal,
    air_quality_to_signal,
    anomaly_to_signal,
    composite_score,
    similarity_to_signal,
)
from .vector_store import WeatherVectorStore, closeness_label, notable_gaps
from .weather_client import OpenMeteoClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("weather-insights")

POLL_INTERVAL_SECONDS = 60
# Alpha Vantage's free tier caps at 25 requests/day; with 3 tickers, a 4h
# cadence is 3 x 6 = 18 calls/day -- well under the cap with margin for
# retries. Deliberately a separate, much slower loop from the weather poll,
# not folded into poll_once().
RETAIL_POLL_INTERVAL_SECONDS = 4 * 60 * 60

app = FastAPI(title="Realtime Weather Insights")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


class ConnectionManager:
    def __init__(self):
        self.active: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()
vector_store = WeatherVectorStore()
detector = AnomalyDetector()
weather_client = OpenMeteoClient()
air_quality_client = OpenMeteoAirQualityClient()
regime_tracker = RegimeTracker(k=3)
forecaster = ForecastTracker()
correlation_trackers = [CorrelationTracker(a, b) for a, b in TRACKED_PAIRS]

_alpha_vantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
retail_client = AlphaVantageRetailClient(_alpha_vantage_key) if _alpha_vantage_key else None
climate_retail_correlation = ClimateRetailCorrelationTracker()
# retail_once() only runs every RETAIL_POLL_INTERVAL_SECONDS (4h), not every
# 60s poll cycle. Without this cache, a browser tab that connects between two
# retail cycles never receives a "retail" message at all until the next one
# fires -- up to 4h of staring at the frontend's generic "aguardando dados"
# placeholder even though the server already has real data in hand. Caching
# the last broadcast and replaying it to each newly-connected socket (see
# ws_endpoint below) closes that gap.
_last_retail_message: dict | None = None


@app.get("/")
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"status": "ok", "hint": "frontend not bundled in this deployment"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "clients": len(manager.active),
        "narrator_enabled": narrator.is_enabled(),
        "retail_panel_enabled": retail_client is not None,
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    if _last_retail_message is not None:
        # Replay the last known retail snapshot immediately so this client
        # doesn't have to wait for the next 4h retail cycle just to see data
        # (and its real day-count-so-far correlation note) the server
        # already has. Sent only to this socket, not broadcast.
        try:
            await websocket.send_json(_last_retail_message)
        except Exception:
            pass
    try:
        while True:
            # The client doesn't need to send anything; this just keeps the
            # connection open and detects disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def poll_once():
    """Poll every city once, run every detector, broadcast each reading,
    then (at most once for the whole cycle) ask the narrator for a
    one-sentence summary if anything noteworthy happened. Split out from
    poll_loop so it's easy to call directly from tests.

    Every detector's raw output is converted into a `Signal` (see
    signals.py) before anything else happens with it. That's what lets
    `composite_score()` and the narrator treat every signal type
    uniformly -- anomaly and similarity (v1), plus air quality,
    correlation breaks, forecast misses, and regime changes (v2 phase 2)
    -- without each one needing its own bespoke wiring here. Climatology
    and nearby-natural-event signals (v2 phase 3) plug in the same way
    later.
    """
    cycle_signals: list[Signal] = []

    for city in CITIES:
        try:
            reading, aq_reading = await asyncio.gather(
                weather_client.fetch(city["name"], city["latitude"], city["longitude"]),
                air_quality_client.fetch(city["name"], city["latitude"], city["longitude"]),
                return_exceptions=True,
            )
        except Exception as exc:  # pragma: no cover - asyncio.gather itself shouldn't raise here
            log.warning("fetch failed for %s: %s", city["name"], exc)
            continue

        if isinstance(reading, Exception):
            log.warning("weather fetch failed for %s: %s", city["name"], reading)
            continue
        if isinstance(aq_reading, Exception):
            # Air quality is additive, not load-bearing: the rest of the
            # cycle proceeds exactly as it would without it, same
            # "degrade, don't break" pattern as the narrator and Chart.js.
            log.warning("air quality fetch failed for %s: %s", city["name"], aq_reading)
            aq_reading = None

        reading_dict = asdict(reading)
        anomalies = detector.evaluate(city["name"], reading_dict)
        doc_id = vector_store.add(reading)
        vector = vector_store.vector_for(reading)
        similar = vector_store.nearest_similar(reading, exclude_id=doc_id)

        city_signals: list[Signal] = [
            anomaly_to_signal(a, city["name"]) for a in anomalies
        ]
        for a in anomalies:
            if a["metric"] == "temperature_c":
                # Feeds the retail correlation tracker's "climate" side --
                # see retail_signals.py for why only temperature (not
                # every metric) counts here.
                climate_retail_correlation.record_temperature_anomaly()
        for s in similar[:1]:
            city_signals.append(similarity_to_signal(
                city=city["name"],
                match=s,
                closeness=closeness_label(s["distance"]),
                gaps=notable_gaps(reading, s),
            ))

        if aq_reading is not None:
            band = eaqi_band(aq_reading.european_aqi)
            if aq_reading.european_aqi > 40:  # below "moderada" isn't signal-worthy
                city_signals.append(air_quality_to_signal(
                    city["name"], aq_reading.european_aqi, aq_reading.pm2_5, band
                ))

        for tracker in correlation_trackers:
            sig = tracker.evaluate(city["name"], reading_dict)
            if sig:
                city_signals.append(sig)

        city_signals.extend(forecaster.evaluate(city["name"], reading_dict))

        regime_sig = regime_tracker.assign_one(vector_store, city["name"], vector)
        if regime_sig:
            city_signals.append(regime_sig)

        cycle_signals.extend(city_signals)

        await manager.broadcast({
            "type": "reading",
            "reading": reading_dict,
            "anomalies": anomalies,
            "similar_patterns": similar,
            # air_quality/composite_score aren't rendered by the frontend
            # yet (that's phase 4 of the v2 plan -- leaderboard + map).
            # Broadcasting them now is free and means the UI work later
            # doesn't need a backend change too.
            "air_quality": asdict(aq_reading) if aq_reading is not None else None,
            "composite_score": composite_score(city_signals),
        })

    if narrator.is_enabled():
        text = await narrator.narrate(cycle_signals)
        if text:
            await manager.broadcast({"type": "narrative", "text": text})


async def poll_loop():
    while True:
        await poll_once()
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def retail_once():
    """Fetch the small fixed set of weather-sensitive retail stocks,
    record today's price move against today's temperature-anomaly count,
    and broadcast both the quote and its (possibly still-accumulating)
    real correlation. Kept fully separate from poll_once()/cycle_signals
    -- see retail_signals.py for why this stays outside the Signal/
    composite_score system the other detectors share."""
    if retail_client is None:
        return
    try:
        quotes = await retail_client.fetch_all()
    except Exception as exc:
        log.warning("retail quote fetch failed: %s", exc)
        return
    if quotes:
        payload = []
        for q in quotes:
            climate_retail_correlation.record_quote(q.symbol, q.change_percent)
            quote_dict = asdict(q)
            quote_dict["correlation"] = climate_retail_correlation.correlation_for(q.symbol)
            payload.append(quote_dict)
        global _last_retail_message
        _last_retail_message = {
            "type": "retail",
            "quotes": payload,
            "disclaimer": RETAIL_DISCLAIMER,
        }
        await manager.broadcast(_last_retail_message)


async def retail_loop():
    while True:
        await retail_once()
        await asyncio.sleep(RETAIL_POLL_INTERVAL_SECONDS)


@app.on_event("startup")
async def start_background_task():
    asyncio.create_task(poll_loop())
    if retail_client is not None:
        asyncio.create_task(retail_loop())
