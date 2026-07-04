"""Turns structured anomaly / similarity signals into a short natural
language sentence using the Claude API.

Everything else in this app -- z-score anomaly detection, vector distance
similarity search -- is legitimate but not actually a language model; this
module is the one part that's genuinely generative AI. It's kept as a thin,
optional, additive layer on top of the existing pipeline: nothing else in
the app depends on it, and it degrades to "disabled" cleanly if no API key
is configured, exactly like the Chart.js frontend fallback.

Cost control: this is called at most once per poll cycle (every 60s), not
once per city, and only when there's something to report (at least one
signal -- of any type, see signals.py -- fired that cycle). Uses Haiku,
the cheapest/fastest Claude model, with a small prompt and a short
max_tokens.
"""
from __future__ import annotations

import asyncio
import logging
import os

from .signals import Signal

log = logging.getLogger("weather-insights.narrator")

_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 100

_client = None
_enabled = bool(os.environ.get("ANTHROPIC_API_KEY"))

if _enabled:
    try:
        import anthropic
        _client = anthropic.Anthropic()
    except Exception as exc:  # pragma: no cover - defensive, missing package/bad key
        log.warning("Anthropic client unavailable, narrator disabled: %s", exc)
        _enabled = False


def is_enabled() -> bool:
    return _enabled


def _build_prompt(signals: list[Signal]) -> str:
    lines = []
    for sig in signals:
        if sig.type == "anomaly":
            e = sig.evidence
            lines.append(
                f"- Anomaly in {sig.city}: {e['metric']} = {e['value']} "
                f"(z-score {e['z_score']}, recent baseline {e['baseline_mean']})"
            )
        elif sig.type == "similarity":
            e = sig.evidence
            gaps = e.get("notable_gaps") or []
            gaps_note = f" Notable real gaps despite this: {'; '.join(gaps)}." if gaps else ""
            lines.append(
                f"- {sig.city} currently resembles {e['matches']} "
                f"(vector distance {e['distance']}). Pre-computed closeness "
                f"(use this exact phrase, do not invent your own wording): "
                f"\"{e['closeness_label']}\".{gaps_note}"
            )
        else:
            # Future signal types (air quality, correlation breaks,
            # forecast misses, regime changes, climatology, nearby
            # natural events) fall back to their own plain-language
            # `summary` until they earn bespoke phrasing here the way
            # anomaly/similarity have -- new sources work immediately,
            # tuned prompting for them can follow later.
            lines.append(f"- {sig.type} signal in {sig.city}: {sig.summary}")
    return (
        "You are annotating a live weather-monitoring dashboard for a portfolio "
        "demo. Given these raw signals from the last 60-second polling cycle, "
        "write ONE short sentence (max ~30 words) in Brazilian Portuguese "
        "pointing out the single most interesting thing happening right now. "
        "Be specific and concrete, no filler, no bullet points, no preamble -- "
        "just the one sentence.\n\n"
        "Ground the sentence only in the numbers given below -- do not imply "
        "a historical record or 'all-time' comparison, since you were only "
        "given this one cycle. For any similarity match, you MUST describe "
        "how close the two cities are using the exact pre-computed closeness "
        "phrase given in quotes -- do not substitute your own judgment (e.g. "
        "do not say 'quase idênticas' unless that literal phrase was given "
        "to you). If a match lists notable real gaps, name at least one of "
        "them concretely instead of glossing over it -- a low aggregate "
        "distance can still hide a large gap in one specific metric like "
        "wind or temperature.\n\n"
        + "\n".join(lines)
    )


async def narrate(signals: list[Signal]) -> str | None:
    """`signals` is the flat, unified list gathered across one full poll
    cycle (all cities, every signal type -- see signals.py). Returns None
    if the narrator is disabled, there's nothing noteworthy this cycle, or
    the API call fails for any reason -- callers should treat that as "no
    narration this cycle", not an error."""
    if not _enabled or not signals:
        return None

    prompt = _build_prompt(signals)
    try:
        resp = await asyncio.to_thread(
            _client.messages.create,
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        return text or None
    except Exception as exc:
        log.warning("Narrator call failed, skipping this cycle: %s", exc)
        return None
