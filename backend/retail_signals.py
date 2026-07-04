"""Client for Alpha Vantage's free GLOBAL_QUOTE endpoint, plus a real (if
modest) statistical correlation between daily temperature anomalies and
each tracked stock's daily price move -- the "clima -> varejo" panel.

Still deliberately NOT wired into the Signal/composite_score system used
by anomaly.py, correlation.py, forecast.py, and regime.py: those detectors
fire the moment something unusual happens *this cycle*. A single day's
stock move can't honestly be called "unusual because of today's weather"
the way a live z-score anomaly can -- there just isn't a "this cycle"
version of that claim for a national stock. What CAN be said honestly is
a real Pearson correlation computed across many real days once the
dashboard has been running long enough to accumulate them (see
`ClimateRetailCorrelationTracker` below) -- so that's what this module
computes and reports, candidly, including when there isn't enough data
yet to mean anything.

Free-tier Alpha Vantage caps at 25 requests/day. With 3 tickers, main.py
calls this on its own slow loop (every few hours, not every 60s poll
cycle like the weather fetch) to stay well under that limit.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass

import httpx

from .statutils import pearson

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# Fixed, small set of US-listed retail/consumer stocks with a real,
# publicly documented (if informal) seasonal or event-driven sensitivity to
# weather -- chosen for having an actual public narrative behind them
# (e.g. Generac's own investor materials discuss storm/outage-driven
# demand), not picked to make a correlation "work."
TICKERS = [
    {
        "symbol": "GNRC",
        "name": "Generac Holdings",
        "context": "Fabricante de geradores residenciais -- demanda historicamente sobe com tempestades e apagões.",
    },
    {
        "symbol": "HD",
        "name": "Home Depot",
        "context": "Rede de materiais de construção -- demanda historicamente sobe com reparos pós-tempestade e jardinagem em ondas de calor.",
    },
    {
        "symbol": "PEP",
        "name": "PepsiCo",
        "context": "Bebidas e alimentos -- consumo historicamente sobe em ondas de calor.",
    },
]

DISCLAIMER = (
    "Contexto informativo com base em sensibilidade histórica documentada. "
    "A correlação abaixo (quando calculada) usa apenas anomalias de "
    "temperatura das 6 cidades deste dashboard -- não cobre tempestades, "
    "vento ou outros fatores reais. Correlação não implica causalidade, e "
    "nada aqui é recomendação de investimento."
)

# A correlation over a handful of days is noise, not signal -- below this
# many overlapping days of (anomaly count, price move) pairs, every ticker
# reports "not enough data yet" instead of a number that looks meaningful
# but isn't. Matches the spirit of correlation.py's own window gating.
MIN_DAYS_FOR_CORRELATION = 15
_MAX_HISTORY_DAYS = 120


@dataclass
class RetailQuote:
    symbol: str
    name: str
    context: str
    price: float
    change_percent: float
    latest_trading_day: str


class ClimateRetailCorrelationTracker:
    """Tracks, per calendar day, how many temperature anomalies fired
    across all 6 tracked cities and each ticker's daily price change, then
    computes a real Pearson correlation between the two once there's
    enough overlapping daily history for the number to mean anything.

    Deliberately conservative: below `MIN_DAYS_FOR_CORRELATION` days of
    paired data, `correlation_for()` returns `r: None` and a plain note
    instead of a number, so a demo that's only been running a few days
    never shows a big, spurious-looking correlation. In-memory only --
    resets on restart, same tradeoff as vector_store.py's history (see
    README "Honest notes"); the correlation becomes more meaningful the
    longer the dashboard stays running.

    This only tracks TEMPERATURE anomalies as the "climate" side of the
    correlation -- it does not know about storms, wind, or the other
    real-world drivers behind GNRC/HD's own investor narratives, so a
    weak correlation here doesn't mean there's no real relationship, only
    that this particular proxy doesn't capture it.
    """

    def __init__(self, min_days: int = MIN_DAYS_FOR_CORRELATION, max_history_days: int = _MAX_HISTORY_DAYS):
        self._min_days = min_days
        self._max_history_days = max_history_days
        self._anomaly_counts: dict[str, int] = {}  # ISO date -> count
        self._ticker_changes: dict[str, dict[str, float]] = {}  # symbol -> {ISO date: change_percent}

    def _trim(self, series: dict) -> None:
        if len(series) <= self._max_history_days:
            return
        oldest = sorted(series.keys())[: len(series) - self._max_history_days]
        for day in oldest:
            series.pop(day, None)

    def record_temperature_anomaly(self, day: str | None = None) -> None:
        """Call once for each temperature-metric anomaly signal that fires
        this poll cycle, across any city. A day with anomalies in 3
        different cities counts as 3 -- more anomalies that day, more
        weight in the correlation, rather than a flat yes/no per day.

        `day` defaults to None (today, real wall-clock date) -- production
        code never passes it. Tests pass explicit ISO date strings to
        simulate many days of history without waiting for real days to
        pass, the same reason `transport` is injectable on
        AlphaVantageRetailClient above."""
        day = day or datetime.date.today().isoformat()
        self._anomaly_counts[day] = self._anomaly_counts.get(day, 0) + 1
        self._trim(self._anomaly_counts)

    def record_quote(self, symbol: str, change_percent: float, day: str | None = None) -> None:
        """Call once per ticker each time a fresh quote is fetched. Alpha
        Vantage's GLOBAL_QUOTE only updates once per trading day, so
        fetching more than once a day just overwrites today's entry with
        the same value -- harmless, not double-counted. `day` is
        test-only, same as in `record_temperature_anomaly`."""
        day = day or datetime.date.today().isoformat()
        self._ticker_changes.setdefault(symbol, {})[day] = change_percent
        self._trim(self._ticker_changes[symbol])

    def correlation_for(self, symbol: str) -> dict:
        changes = self._ticker_changes.get(symbol, {})
        # Anchor on every day we have a recorded price change for, and
        # treat a day with no recorded anomaly as a genuine zero, not a
        # missing data point. Excluding zero-anomaly ("quiet") days would
        # silently drop exactly the days most likely to show a small or
        # no price move, biasing the correlation toward whatever the
        # noisier days happened to look like.
        days = sorted(changes)
        n = len(days)

        if n < self._min_days:
            return {
                "r": None,
                "n": n,
                "min_days": self._min_days,
                "note": f"Ainda não há dias suficientes para calcular correlação ({n}/{self._min_days} dias).",
            }

        xs = [self._anomaly_counts.get(d, 0) for d in days]
        ys = [changes[d] for d in days]
        r = round(pearson(xs, ys), 2)
        magnitude = abs(r)
        strength = "fraca" if magnitude < 0.3 else "moderada" if magnitude < 0.6 else "forte"
        return {
            "r": r,
            "n": n,
            "min_days": self._min_days,
            "note": f"Correlação {strength} entre anomalias de temperatura e variação diária (n={n} dias). Correlação não implica causalidade.",
        }


class AlphaVantageRetailClient:
    """Thin async wrapper around Alpha Vantage's GLOBAL_QUOTE endpoint.

    `transport` defaults to None (httpx's real network transport) and only
    exists so tests can inject an `httpx.MockTransport` and exercise the
    parsing/degrade-don't-break logic below without a live API key or
    network access -- same spirit as `_FakeStore` in test_pipeline.py."""

    def __init__(self, api_key: str, timeout: float = 10.0, transport: httpx.AsyncBaseTransport | None = None):
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport

    async def fetch_one(self, ticker: dict) -> RetailQuote | None:
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": ticker["symbol"],
            "apikey": self._api_key,
        }
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            resp = await client.get(ALPHA_VANTAGE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        quote = data.get("Global Quote") or {}
        if not quote or "05. price" not in quote:
            # Rate-limited, bad symbol, or a transient API hiccup -- same
            # "degrade, don't break" pattern as air quality and the
            # narrator: skip this ticker this cycle instead of raising.
            return None

        change_pct_raw = quote.get("10. change percent", "0%").strip().rstrip("%")
        try:
            price = float(quote["05. price"])
            change_percent = float(change_pct_raw or 0.0)
        except ValueError:
            return None

        return RetailQuote(
            symbol=ticker["symbol"],
            name=ticker["name"],
            context=ticker["context"],
            price=price,
            change_percent=change_percent,
            latest_trading_day=quote.get("07. latest trading day", ""),
        )

    async def fetch_all(self) -> list[RetailQuote]:
        quotes: list[RetailQuote] = []
        for ticker in TICKERS:
            try:
                q = await self.fetch_one(ticker)
                if q:
                    quotes.append(q)
            except Exception:
                # One bad ticker or network hiccup shouldn't drop the
                # other two -- same tolerance as the rest of the pipeline.
                continue
        return quotes
