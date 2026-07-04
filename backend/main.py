"""FastAPI app: polls real weather data for a fixed set of cities, runs
anomaly detection + vector similarity search on each reading, and streams
the results to connected WebSocket clients in real time.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import narrator
from .anomaly import AnomalyDetector
from .cities import CITIES
from .signals import Signal, anomaly_to_signal, composite_score, similarity_to_signal
from .vector_store import WeatherVectorStore, closeness_label, notable_gaps
from .weather_client import OpenMeteoClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("weather-insights")

POLL_INTERVAL_SECONDS = 60

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
    }


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # The client doesn't need to send anything; this just keeps the
            # connection open and detects disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def poll_once():
    """Poll every city once, run detection + similarity search, broadcast
    each reading, then (at most once for the whole cycle) ask the narrator
    for a one-sentence summary if anything noteworthy happened. Split out
    from poll_loop so it's easy to call directly from tests.

    Every detector's raw output is converted into a `Signal` (see
    signals.py) before anything else happens with it. That's what lets
    `composite_score()` and the narrator treat anomaly and similarity
    signals uniformly today, and treat whatever v2 adds (air quality,
    correlation breaks, forecast misses, regime changes, climatology,
    nearby natural events) the same way tomorrow -- new sources plug into
    this same list instead of needing their own bespoke wiring here.
    """
    cycle_signals: list[Signal] = []

    for city in CITIES:
        try:
            reading = await weather_client.fetch(
                city["name"], city["latitude"], city["longitude"]
            )
        except Exception as exc:
            log.warning("fetch failed for %s: %s", city["name"], exc)
            continue

        reading_dict = asdict(reading)
        anomalies = detector.evaluate(city["name"], reading_dict)
        doc_id = vector_store.add(reading)
        similar = vector_store.nearest_similar(reading, exclude_id=doc_id)

        city_signals: list[Signal] = [
            anomaly_to_signal(a, city["name"]) for a in anomalies
        ]
        for s in similar[:1]:
            city_signals.append(similarity_to_signal(
                city=city["name"],
                match=s,
                closeness=closeness_label(s["distance"]),
                gaps=notable_gaps(reading, s),
            ))
        cycle_signals.extend(city_signals)

        await manager.broadcast({
            "type": "reading",
            "reading": reading_dict,
            "anomalies": anomalies,
            "similar_patterns": similar,
            # Not rendered by the frontend yet (that's phase 4 of the v2
            # plan -- leaderboard + map). Broadcasting it now is free and
            # means the UI work later doesn't need a backend change too.
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


@app.on_event("startup")
async def start_background_task():
    asyncio.create_task(poll_loop())
