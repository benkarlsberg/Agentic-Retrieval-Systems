"""Statistical utilities: paired bootstrap CIs."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class BootstrapCI(BaseModel):
    mean: float
    low: float
    high: float
    n: int
    n_bootstrap: int
    alpha: float


def mean_ci(
    values: list[float], n_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42
) -> BootstrapCI:
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return BootstrapCI(mean=0.0, low=0.0, high=0.0, n=0, n_bootstrap=n_bootstrap, alpha=alpha)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        means.append(float(sample.mean()))
    means_arr = np.sort(np.asarray(means))
    low = float(np.quantile(means_arr, alpha / 2))
    high = float(np.quantile(means_arr, 1 - alpha / 2))
    return BootstrapCI(
        mean=float(arr.mean()),
        low=low,
        high=high,
        n=n,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )


def paired_bootstrap_ci(
    a: list[float],
    b: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """CI on mean(a - b)."""
    if len(a) != len(b):
        raise ValueError("paired_bootstrap_ci requires equal-length inputs")
    diff = [x - y for x, y in zip(a, b, strict=True)]
    return mean_ci(diff, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed)
