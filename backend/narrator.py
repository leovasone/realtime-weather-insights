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
anomaly or cross-city similarity match that cycle). Uses Haiku, the
cheapest/fastest Claude model, with a small prompt and a short max_tokens.
"""
from __future__ import annotations

import asyncio
import logging
import os

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


def _build_prompt(anomalies: list[dict], similar: list[dict]) -> str:
    lines = []
    for a in anomalies:
        lines.append(
            f"- Anomaly in {a['city']}: {a['metric']} = {a['value']} "
            f"(z-score {a['z_score']}, recent baseline {a['baseline_mean']})"
        )
    for s in similar:
        gaps = s.get("notable_gaps") or []
        gaps_note = f" Notable real gaps despite this: {'; '.join(gaps)}." if gaps else ""
        lines.append(
            f"- {s['city']} currently resembles {s['matches']} "
            f"(vector distance {s['distance']}). Pre-computed closeness "
            f"(use this exact phrase, do not invent your own wording): "
            f"\"{s['closeness_label']}\".{gaps_note}"
        )
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


async def narrate(anomalies: list[dict], similar: list[dict]) -> str | None:
    """anomalies/similar are flat lists gathered across one full poll cycle
    (all cities), each tagged with its own city. Returns None if the
    narrator is disabled, there's nothing noteworthy this cycle, or the API
    call fails for any reason -- callers should treat that as "no narration
    this cycle", not an error."""
    if not _enabled or not (anomalies or similar):
        return None

    prompt = _build_prompt(anomalies, similar)
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
