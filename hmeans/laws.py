"""e-means constants, batch/step laws and schedule (torch-free; shared by both backends).

Values are those shipped in faiss ``contrib/e_means`` (faiss-cpu 1.16.0), not the paper's rounded ones.
"""
from __future__ import annotations

import math

ALPHA_MIN_FACTOR = 0.1
CHECK_UNUSED_EVERY = 5
INIT_USAGE = 2.0**-16
SPLIT_EPSILON = 1e-4
UNUSED_THRESHOLD = 0.1
NO_REVIVAL_MAX_D = 2
AUTO_QUALITY_MAX_D = 16
BATCH_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}
ALPHA_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)


def round_power_of_two(value: float) -> int:
    if value <= 1:
        return 1
    lo = 2 ** math.floor(math.log2(value))
    hi = lo * 2
    return int(lo if value / lo < hi / value else hi)


def step_law(n, k, nepochs, batch_size) -> float:
    c, ak, ab, an, at = ALPHA_LAW
    a0 = c * k**ak * batch_size**ab * n**an * nepochs**at
    return min(max(a0, 1e-4), 1.0)


def resolve_batch_law(batch_law: str, d: int) -> str:
    if batch_law != "auto":
        return batch_law
    return "quality" if d <= AUTO_QUALITY_MAX_D else "throughput"


def resolve_recipe(n, k, nepochs, batch_law, batch_size=None, alpha0=None):
    if batch_size is None:
        c, bk, bn, bt = BATCH_LAWS[batch_law]
        raw = c * k**bk * n**bn * nepochs**bt
        batch_size = round_power_of_two(raw)
        if batch_law == "throughput":
            batch_size = min(batch_size, round_power_of_two(n / 2))
        else:
            batch_size = min(batch_size, n // 2)
        batch_size = max(1, min(batch_size, n))
    else:
        batch_size = min(int(batch_size), n)
    if alpha0 is None:
        alpha0 = step_law(n, k, max(nepochs, 1e-9), batch_size)
    return int(batch_size), float(alpha0)


def epoch_rows(n: int, nepochs: float) -> list[int]:
    whole = math.floor(nepochs)
    remainder = int(round((nepochs - whole) * n))
    if whole == 0:
        return [remainder]
    return [n] * (whole - 1) + ([remainder] if remainder else []) + [n]


def scheduled_alpha(alpha0: float, step: float, total_steps: float) -> float:
    fraction = min(step, total_steps) / max(1, total_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)


def epoch_seed(seed: int, iteration: int) -> int:
    return (seed * 1000003 + iteration) % 2**32


def auto_ema_frac(n: int, k: int) -> float:
    """Fraction of the passes run in EMA mode as a function of points per cluster.

    Below 24 points per cluster the exact Hartigan passes carry the gain (0.4); between 24 and 64
    the EMA passes matter more (0.8); above 64 the exact tail no longer helps and can hurt on
    separable data, so all passes stay in EMA mode (1.0: warm start + Hartigan-corrected e-means).
    Thresholds were set on the measurements in REPORT.md §3 (see its selection caveat)."""
    r = n / max(k, 1)
    return 0.4 if r < 24 else 0.8 if r < 64 else 1.0
