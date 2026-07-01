# Realtime Weather Insights

A working real-time insights dashboard: it ingests live public weather data,
runs it through an AI pipeline (anomaly detection + vector similarity
search), and streams the results to the browser over WebSockets as they
happen.

Built as a demonstration of a real-time data platform architecture —
ingestion, streaming, AI-driven pattern detection, vector search — using a
public, free, no-auth data source instead of synthetic/mocked data.

## Why weather data

No API key, no rate-limit headaches, no client data involved, and weather
naturally produces the kind of numeric time series that anomaly detection
and similarity search are built for.

## Architecture

```
Open-Meteo API  →  poll loop (60s)  →  anomaly detector (z-score)
                                     →  vector store (in-memory)  →  similarity search
                                     →  narrator (Claude, optional)
                                     ↓
                              WebSocket broadcast  →  browser dashboard (live charts + insights feed)
```

- **`backend/weather_client.py`** — thin async client for the free
  [Open-Meteo](https://open-meteo.com) current-weather API. Isolated behind
  its own class so it's swappable for a different data source without
  touching the rest of the pipeline (same pluggable-adapter pattern used in
  the [document-extraction project](https://github.com/leovasone/document-extraction-demo)).
- **`backend/anomaly.py`** — rolling z-score anomaly detection per city, per
  metric (temperature, humidity, wind, pressure, cloud cover). Fully
  explainable: flags a reading when it's more than 2.5 standard deviations
  from its own recent rolling baseline.
- **`backend/vector_store.py`** — every reading is embedded as a 5-dimensional
  numeric vector and kept in a bounded in-memory store. On each new reading, a
  brute-force similarity search finds the closest historical match *in a
  different city* (any time) — surfacing insights like "current conditions
  in São Paulo are similar to what Cairo experienced earlier." (Originally
  built on ChromaDB; swapped for a plain Python implementation after its
  Rust bindings pushed memory past what a free-tier container had available
  — see "Honest notes".)
- **`backend/narrator.py`** — the one part of this pipeline that's actually
  a language model, as opposed to statistics/linear algebra dressed up as
  "AI". Once per poll cycle (not per city), if anything noteworthy happened,
  it sends the structured anomalies + similarity matches to Claude (Haiku)
  and asks for a single plain-Portuguese sentence highlighting what matters.
  Entirely optional: with no `ANTHROPIC_API_KEY` set, it's a no-op and the
  rest of the dashboard is unaffected — same "degrade, don't break" pattern
  as the Chart.js loader.
- **`backend/main.py`** — FastAPI app: a background task polls every city
  every 60 seconds, runs both detectors, broadcasts each reading, and — at
  most once per cycle — asks the narrator for a summary sentence.
- **`frontend/index.html`** — single-page dashboard: live per-city cards,
  a temperature chart (Chart.js), a highlighted "AI Narrator" line when
  enabled, and a live-updating raw insights feed.

## Running locally

```bash
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

Then open `http://localhost:8000`.

## Testing without live network access

The anomaly detector and vector store are pure logic with no network
dependency, so they're tested directly with synthetic data (including one
deliberately injected temperature spike, across two synthetic cities so
cross-city similarity search has something real to find):

```bash
python -m backend.test_pipeline
```

This was how the pipeline was validated during development, since the
sandbox it was built in has no outbound access to `api.open-meteo.com`. The
live API call itself is a five-line, fully isolated adapter
(`weather_client.py`) — the part worth testing rigorously is the detection
and search logic sitting behind it, which is what `test_pipeline.py` covers.
On a normal host (including Railway) the real API call works as-is.

## Deploying to Railway

1. Push this repo to GitHub.
2. In Railway, **New Project → Deploy from GitHub repo** and select it.
   Railway auto-detects the `Dockerfile` and builds from it.
3. Open-Meteo itself needs no API key. The AI narrator is optional: to
   enable it, add an `ANTHROPIC_API_KEY` environment variable in Railway's
   service Variables tab (get a key at
   [console.anthropic.com](https://console.anthropic.com)). Without it, the
   dashboard runs exactly the same, just without the narrator line.
4. Railway assigns a public URL and a `PORT` env var automatically; the
   Dockerfile's `CMD` already binds to `$PORT`.
5. Pattern history lives in memory (bounded to the last 2,000 readings) and
   resets on every restart — fine for a demo; swap `vector_store.py` for a
   persisted backend (Postgres + pgvector, Qdrant, etc.) if you want history
   to survive restarts in a real deployment.

## Honest notes

- Poll interval is 60 seconds — weather doesn't change second-to-second, so
  faster polling would just be noise (and unkind to a free public API).
- The 2.5 standard-deviation threshold is a reasonable default, not a tuned
  hyperparameter — there's no labeled anomaly dataset for real-world
  weather to tune against here.
- No authentication, rate limiting, or multi-tenant support — this is a
  single shared public dashboard, not a production SaaS.
- First deploy used ChromaDB for the vector store. In production on Railway
  it caused the container to get killed and restarted roughly every 60-70
  seconds (no traceback, consistent with an OOM kill), which repeatedly
  dropped every open WebSocket connection. Swapped it for a small in-memory
  brute-force search — at this data volume (a handful of 5-float vectors per
  city) a full vector database was solving a scale problem this app doesn't
  have. Worth knowing before reaching for a vector DB on a resource-limited
  host: check whether brute-force search is actually fast enough first.
- Chart.js was originally loaded via a single blocking `<script>` tag
  pointing at cdnjs. In at least one real browser session it never loaded
  (`ReferenceError: Chart is not defined`, even after fixing an earlier
  case-sensitivity typo in the URL), leaving the chart panel blank with no
  visible explanation. Rewrote loading to be dynamic and decoupled from the
  rest of the page: it tries cdnjs, falls back to jsdelivr if that fails,
  an