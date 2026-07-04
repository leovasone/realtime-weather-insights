"""Small, shared, plain-Python statistics helpers.

Deliberately not numpy/scipy: every user of this module (correlation.py,
retail_signals.py) works with at most a few dozen data points, so a
textbook implementation in pure Python is simpler to read, easier to
audit, and has no dependency to go wrong -- same reasoning already
documented in the README for why regime.py's k-means and vector_store.py's
similarity search are hand-rolled instead of pulling in scikit-learn/numpy.
"""
from __future__ import annotations


def pearson(xs: list[float], ys: list[float]) -> float:
    """Textbook Pearson correlation coefficient. Returns 0.0 (no
    relationship) rather than raising if either series has zero variance
    -- a flat series has no correlation to report, not an error."""
    n = len(xs)
    if n == 0:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)
