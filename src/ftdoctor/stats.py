"""Robust statistics. Shared lineage with rldoctor's stats module.

Training curves are noisy and heavy-tailed, and a handful of bad batches should
not decide which checkpoint you keep. Everything here is rank- or median-based,
and the only distributional result needed -- the normal CDF -- comes from
:func:`math.erf`, so there is no SciPy dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

_RNG = np.random.default_rng(0)


def fmt_p(p: float) -> str:
    """Format a p-value honestly.

    Returns the relational operator too, so call sites read ``f"p{fmt_p(x)}"``.
    The normal approximation underflows to exactly 0.0 for strong trends, and
    printing "p=0" in a report aimed at researchers is a fast way to lose them.
    """
    if p <= 0.0:
        return "<1e-16"
    if p < 1e-4:
        return f"={p:.1e}"
    return f"={p:.3g}"


def normal_cdf(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class Trend:
    """Result of a robust trend test over a series."""

    slope: float  #: Theil-Sen slope, units of y per unit of x
    intercept: float
    p_value: float  #: Mann-Kendall two-sided p-value
    n: int
    tau: float  #: Kendall's tau-b, a scale-free effect size in [-1, 1]

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05 and self.n >= 8

    @property
    def direction(self) -> str:
        if not self.significant:
            return "flat"
        return "up" if self.slope > 0 else "down"

    def predict(self, x: float) -> float:
        return self.intercept + self.slope * x


def theil_sen(x: np.ndarray, y: np.ndarray, max_pairs: int = 40_000) -> Tuple[float, float]:
    """Median-of-pairwise-slopes regression.

    Returns ``(slope, intercept)``.  Breakdown point ~29%, so a handful of
    catastrophic steps cannot flip the reported direction of a trend.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 2:
        return 0.0, float(y[0]) if n else 0.0

    n_pairs = n * (n - 1) // 2
    if n_pairs <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        # Subsample pairs; the median slope estimate is stable well below the
        # full pair set and this keeps long runs interactive.
        i = _RNG.integers(0, n, size=max_pairs)
        j = _RNG.integers(0, n, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]

    dx = x[j] - x[i]
    valid = dx != 0
    if not np.any(valid):
        return 0.0, float(np.median(y))
    slopes = (y[j][valid] - y[i][valid]) / dx[valid]
    slope = float(np.median(slopes))
    intercept = float(np.median(y) - slope * np.median(x))
    return slope, intercept


def mann_kendall(y: np.ndarray) -> Tuple[float, float]:
    """Mann-Kendall trend test with tie correction.

    Returns ``(p_value, tau_b)``.  Non-parametric, so it makes no assumption
    that reward curves are normal or homoscedastic -- both of which are false.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 4:
        return 1.0, 0.0

    # S = sum of sign(y_j - y_i) for i < j, computed pairwise.
    diff = np.sign(y[None, :] - y[:, None])
    s = float(np.sum(np.triu(diff, k=1)))

    # Variance with correction for tied groups.
    _, counts = np.unique(y, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if var_s <= 0:
        return 1.0, 0.0

    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * (1.0 - normal_cdf(abs(z)))

    # tau-b denominator accounts for ties on both sides.
    n0 = n * (n - 1) / 2.0
    n1 = float(np.sum(counts * (counts - 1) / 2.0))
    denom = math.sqrt(max(n0 - n1, 1e-12) * n0)
    tau = s / denom if denom > 0 else 0.0
    return float(min(max(p, 0.0), 1.0)), float(max(min(tau, 1.0), -1.0))


def trend(x: np.ndarray, y: np.ndarray) -> Trend:
    """Combined Theil-Sen slope + Mann-Kendall significance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return Trend(0.0, float(y[0]) if y.size else 0.0, 1.0, int(x.size), 0.0)
    slope, intercept = theil_sen(x, y)
    p, tau = mann_kendall(y)
    return Trend(slope, intercept, p, int(x.size), tau)


def mad(y: np.ndarray) -> float:
    """Median absolute deviation, scaled to be a consistent sigma estimator."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(y - np.median(y))))


def robust_z(y: np.ndarray) -> np.ndarray:
    """Per-point robust z-scores using median/MAD."""
    y = np.asarray(y, dtype=float)
    scale = mad(y)
    centred = y - np.median(y)
    if scale <= 0:
        # More than half the points are identical, so the spread is zero and any
        # point that differs is infinitely many robust sigmas out -- not zero.
        # Returning zeros here made a perfectly flat loss with sharp jumps look
        # spike-free.
        z = np.zeros_like(centred)
        z[centred > 0] = np.inf
        z[centred < 0] = -np.inf
        return z
    return centred / scale


def diff_sigma(y: np.ndarray) -> float:
    """Noise scale estimated from successive differences.

    ``mad(y)`` is the wrong scale whenever the series has structure: a step or a
    trend inflates it, and the feature then hides inside its own noise estimate.
    Differencing removes any level and most of a slow trend, and for i.i.d.
    noise ``std(diff) = sqrt(2) * sigma``.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return 0.0
    return float(mad(np.diff(y)) / math.sqrt(2.0))
