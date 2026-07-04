"""Client for Alpha Vantage's free GLOBAL_QUOTE endpoint, used for a small,
purely informational "clima -> varejo" panel on the dashboard.

Deliberately NOT wired into the Signal/composite_score system used by
anomaly.py, correlation.py, forecast.py, and regime.py. Those detectors all
compute something from THIS app's own live data (the weather readings for
these exact 6 cities) and can honestly claim "this is unusual relative to
what we've been observing." A US-listed stock's daily price move cannot
honestly be attributed to a single city's live weather reading -- these are
national/global companies, and one day's price action has dozens of causes
that have nothing to do with weather. So this module stays a plain
informational feed: today's price for a handful of stocks with a
documented seasonal/event-driven sensitivity to weather, displayed next to
(not claimed to be caused by) the weather data. No severity score, no
anomaly detection, no composite_score contribution.

Free-tier Alpha Vantage caps at 25 requests/day. With 3 tickers, main.py
calls this on its own slow loop (every few hours, not every 60s poll
cycle like the weather fetch) to stay well under that limit.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

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
    "Contexto informativo com base em sensibilidade histórica documentada -- "
    "não é uma correlação calculada com o clima exibido acima, nem "
    "recomendação de investimento."
)


@dataclass
class RetailQuote:
    symbol: str
    name: str
    context: str
    price: float
    change_percent: float
    latest_trading_day: str


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
