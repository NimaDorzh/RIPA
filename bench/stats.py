"""Statistics helpers (NEW during hackathon)."""
from __future__ import annotations

import math
import random


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n (assumes independent trials)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cluster_bootstrap_ci(values: list[int], clusters: list, B: int = 4000, seed: int = 0,
                         alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for a mean when observations are grouped (here: 4 wrappers share one base text).
    Resamples whole clusters, so the interval reflects the true number of independent bases."""
    if not values:
        return (0.0, 0.0)
    groups: dict = {}
    for v, c in zip(values, clusters):
        groups.setdefault(c, []).append(v)
    gs = list(groups.values())
    rng = random.Random(seed)
    means = []
    for _ in range(B):
        sample = [gs[rng.randrange(len(gs))] for _ in gs]
        tot = sum(len(g) for g in sample)
        means.append(sum(sum(g) for g in sample) / tot)
    means.sort()
    return (means[int(alpha / 2 * B)], means[int((1 - alpha / 2) * B) - 1])


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pairs (b: A only, c: B only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * p)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def ci_with_fallback(values: list[int], clusters: list, **kw) -> tuple[float, float]:
    """Cluster bootstrap CI; if every value is identical (e.g. all zeros) the bootstrap collapses to a point,
    so fall back to a Wilson interval whose n is the NUMBER OF CLUSTERS (conservative)."""
    lo, hi = cluster_bootstrap_ci(values, clusters, **kw)
    if hi - lo == 0.0 and values:
        n = len(set(clusters))
        return wilson(round(sum(values) / len(values) * n), n)
    return lo, hi
