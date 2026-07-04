"""Unsupervised weather-regime clustering -- the one genuinely
unsupervised-ML piece of the v2 pipeline (z-score and correlation are
statistics; forecasting here is simple smoothing; this is k-means).

Deliberately implemented from scratch in plain Python rather than pulling
in scikit-learn/numpy: the data is a few thousand 5-dimensional points at
most (see vector_store.py's bound), so a dependency that size would be the
same mistake ChromaDB was for a similarity search this small (see
README "Honest notes") -- solving a scale problem this app doesn't have.

Clustering runs over the vector store's full recent history, not just the
current cycle's 6 points: with only 6 data points, cluster assignments
would be unstable and prone to k-means' well-known label-switching problem
between independent re-runs. Re-clustering also doesn't need to happen
every single reading -- climate regimes don't shift minute to minute -- so
`RegimeTracker` only recomputes centroids periodically and otherwise just
assigns each new reading to the nearest already-known one.

Clusters are labeled by their own centroid characteristics (e.g. "quente e
úmido") rather than by arbitrary index, so a label stays meaningful and
comparable across re-clusterings even though k-means itself has no memory
of a previous run's cluster numbering.
"""
from __future__ import annotations

import random

from .signals import Signal, regime_to_signal


def _dist2(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _nearest(v: list[float], centroids: list[list[float]]) -> int:
    return min(range(len(centroids)), key=lambda i: _dist2(v, centroids[i]))


def kmeans(vectors: list[list[float]], k: int, iterations: int = 25, seed: int = 42) -> list[list[float]]:
    """Lloyd's algorithm. Returns final centroids; caller assigns points to
    their nearest one. Stops early once centroids stop moving."""
    rnd = random.Random(seed)
    if len(vectors) <= k:
        return [v[:] for v in vectors]
    centroids = [v[:] for v in rnd.sample(vectors, k)]

    for _ in range(iterations):
        clusters: list[list[list[float]]] = [[] for _ in centroids]
        for v in vectors:
            clusters[_nearest(v, centroids)].append(v)

        new_centroids = []
        for i, pts in enumerate(clusters):
            if pts:
                dims = len(pts[0])
                new_centroids.append([sum(p[d] for p in pts) / len(pts) for d in range(dims)])
            else:
                new_centroids.append(centroids[i])  # empty cluster keeps its old center

        if new_centroids == centroids:
            break
        centroids = new_centroids

    return centroids


def _label_centroid(centroid: list[float]) -> str:
    """Turn a centroid's normalized [temp, humidity, wind, pressure, cloud]
    values (see vector_store._vector's ordering) into a short, descriptive
    regime name. Fixed thresholds against the 0-1 normalized scale, chosen
    to read naturally in Portuguese -- not a tuned classifier."""
    temp, humidity, wind, _pressure, cloud = centroid

    temp_word = "quente" if temp > 0.55 else "frio" if temp < 0.35 else "ameno"
    parts = [temp_word]
    if humidity > 0.6:
        parts.append("úmido")
    elif humidity < 0.35:
        parts.append("seco")
    if wind > 0.3:
        parts.append("ventoso")
    if cloud > 0.7:
        parts.append("nublado")
    return " e ".join(parts)


class RegimeTracker:
    """Maintains climate-regime clusters over a vector store's recent
    history and detects when a city's assignment changes between cycles.

    `assign_one` is meant to be called once per city per poll cycle, right
    after that city's new reading has been added to the vector store --
    see main.py's poll_once(). Centroids refresh from the full history
    every `refresh_every` calls; in between, a new reading is just
    assigned to the nearest already-known centroid, which is effectively
    free.
    """

    def __init__(self, k: int = 3, refresh_every: int = 10):
        self._k = k
        self._refresh_every = refresh_every
        self._centroids: list[list[float]] = []
        self._labels: list[str] = []
        self._last_regime: dict[str, str] = {}
        self._calls_since_refresh = refresh_every  # force a refresh on first call

    def _maybe_refresh(self, store) -> None:
        if self._calls_since_refresh < self._refresh_every and self._centroids:
            self._calls_since_refresh += 1
            return
        pairs = store.all_vectors()
        if len(pairs) >= self._k:
            vectors = [vec for vec, _meta in pairs]
            self._centroids = kmeans(vectors, self._k)
            self._labels = [_label_centroid(c) for c in self._centroids]
        self._calls_since_refresh = 1

    def assign_one(self, store, city: str, vector: list[float]) -> Signal | None:
        """Returns a `regime_change` Signal if this city's regime differs
        from what it was assigned last time this was called for it, else
        None. The first observation of any city never produces a signal --
        there's nothing to compare against yet."""
        self._maybe_refresh(store)
        if not self._centroids:
            return None

        label = self._labels[_nearest(vector, self._centroids)]
        previous = self._last_regime.get(city)
        self._last_regime[city] = label

        if previous is not None and previous != label:
            return regime_to_signal(city, previous, label)
        return None
