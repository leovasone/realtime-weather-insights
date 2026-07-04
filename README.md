# Realtime Weather Insights

🔴 [Live demo](https://realtime-weather-insights-production.up.railway.app) ·
🔗 [Case study](https://vasone.com.br/realtime-insights.html) on
[vasone.com.br](https://vasone.com.br) — Leonardo Vasone's AI/ML & Data
Engineering portfolio.

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
- **`backend/signals.py`** — unified `Signal` schema that every detector's
  output gets converted into before anything else sees it, plus
  `composite_score()`, a simple, fully explainable weighted sum that turns
  however many signals fired for a city into one 0-100 number. This is the
  v2 groundwork: new sources (air quality, cross-metric correlation
  breaks, forecast misses, climate-regime clustering, climatology,
  nearby natural events) each need only a small `*_to_signal()` converter
  to plug into the same composite score and the same narrator, instead of
  becoming their own disconnected panel.
- **`backend/main.py`** — FastAPI app: a background task polls every city
  every 60 seconds, runs both detectors, converts their output into
  `Signal`s, broadcasts each reading (plus its composite score), and — at
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
5. Pattern history lives in memory (bounded to ~24h of readings across all
   cities, currently 8,800 entries) and resets on every restart — fine for
   a demo; swap `vector_store.py` for a persisted backend (Postgres +
   pgvector, Qdrant, etc.) if you want history to survive restarts in a
   real deployment.

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
  and if both fail it swaps in a visible "chart unavailable" message instead
  of empty space — none of this blocks the WebSocket connection or the city
  cards, which never depended on Chart.js in the first place.
- The similarity search's nearest-neighbor exclusion only filtered out the
  exact reading just added, not the rest of that city's own history. Since
  weather barely changes between 60-second polls, each city's own previous
  reading was almost always its nearest vector neighbor — every "similar
  pattern" insight was really just "this city is similar to itself a minute
  ago" (distance ≈ 0), which is true but tells you nothing. Fixed by
  excluding the whole querying city, not just one reading, so only genuine
  cross-city matches are ever surfaced. A single-city test wouldn't have
  caught this, which is why `test_pipeline.py` now uses two synthetic
  cities and explicitly asserts a city never appears in its own results.
- Being upfront about naming: z-score anomaly detection and vector-distance
  similarity search are legitimate techniques, but they're statistics and
  linear algebra, not machine learning models. The narrator (Claude) is the
  only genuinely generative-AI piece of this pipeline. It's called at most
  once per 60-second poll cycle across all six cities combined — not once
  per city — both to keep cost negligible and because "something changed
  somewhere this minute" is a more useful unit of narration than six
  separate one-line summaries every cycle.
- The vector embedding originally used raw values (temperature in °C,
  pressure in hPa, humidity/cloud cover in %) with no scaling, so distance
  mixed dimensions on incompatible scales — a moderate humidity gap could
  outweigh a real temperature gap in the total. Caught this after the
  narrator described a São Paulo/New York match as "praticamente idênticas"
  when the underlying readings were ~25°C vs ~33°C and 9 points apart on
  humidity: the distance really was the smallest among that cycle's pairs,
  but "smallest" isn't the same as "similar" when the units aren't
  comparable. Fixed by min-max scaling each dimension into 0-1 using fixed
  meteorological bounds before computing L2 distance, and tightened the
  narrator's prompt to only claim two cities are "quase idênticas" below a
  strict distance threshold, describing merely-closest pairs more precisely
  otherwise (naming the actual gap when it's the reason two cities aren't a
  real match despite being the closest pair that cycle).
- That prompt-level threshold instruction didn't hold up: shortly after,
  Haiku called a Tokyo/Sydney match (distance 0.24) "quase idênticas"
  anyway, ignoring the "only below 0.05" rule written into the prompt. A
  small, cheap model doesn't reliably apply a numeric threshold buried in
  prose. The underlying numbers made it worse than it sounds -- Tokyo's
  wind was 3.7 km/h against Sydney's 22 km/h, an 18.3 km/h gap that barely
  moved the aggregate distance because it's one of five equally-weighted
  normalized dimensions. Fixed by moving the judgment call out of the
  prompt entirely: `closeness_label()` and `notable_gaps()` in
  `vector_store.py` compute a fixed qualitative phrase and any large
  single-metric gaps in code, and the narrator is now instructed to use
  that exact phrase verbatim and name a concrete gap when one is flagged,
  rather than deciding the wording itself.
- v2 planning raised the question of whether a real database was needed
  for the next round of features (24h heatmap, historical percentile,
  correlation/forecast/regime detectors). It isn't: the data volumes
  involved are tiny (a handful of numbers per city per minute), so the
  24h lookback only needed a bigger in-memory bound, and the historical
  comparison only needs an on-demand call to a climate API, not a
  database of our own. A real database's only genuine benefit here would
  be surviving restarts during active development -- deliberately left
  out for now rather than reached for by default, the same call made
  about ChromaDB above.
