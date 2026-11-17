"""Device-agnostic (CPU / Apple MPS / CUDA) distance and assignment primitives.

Every function that computes point-to-centroid distances increments the global
``OPS`` counter by the number of (point, centroid) pairs it evaluates, so that
algorithms can be compared on distance computations as well as wall-clock time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class OpCounter:
    """Counts distance computations (point-centroid pairs) and timings."""

    pairs: float = 0.0
    macs: float = 0.0  # multiply-accumulates: pairs x dimension actually used
    by_tag: dict = field(default_factory=dict)

    def add(self, n_pairs: float, tag: str = "assign", dim: int | None = None) -> None:
        self.pairs += float(n_pairs)
        if dim is not None:
            self.macs += float(n_pairs) * dim
        self.by_tag[tag] = self.by_tag.get(tag, 0.0) + float(n_pairs)

    def reset(self) -> None:
        self.pairs = 0.0
        self.macs = 0.0
        self.by_tag = {}

    def snapshot(self) -> dict:
        return {"pairs": self.pairs, "macs": self.macs, **{"pairs_" + k: v for k, v in self.by_tag.items()}}


OPS = OpCounter()


def pick_device(name: str | None = None) -> torch.device:
    if name is not None and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    """Wall-clock timer that synchronizes the accelerator at start and stop."""

    def __init__(self, device: torch.device):
        self.device = device
        self.t = 0.0

    def __enter__(self):
        synchronize(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        synchronize(self.device)
        self.t = time.perf_counter() - self.t0


def to_device(x, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x))
    return x.to(device=device, dtype=dtype).contiguous()


def sq_norms(x: torch.Tensor) -> torch.Tensor:
    return (x * x).sum(dim=1)


def assign_exact(
    x: torch.Tensor,
    c: torch.Tensor,
    c_norms: torch.Tensor | None = None,
    chunk_rows: int = 32768,
    count: bool = True,
    tag: str = "assign",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Brute-force nearest centroid for every row of ``x``.

    Returns (labels int64, d2 float32) where d2 is the squared distance computed
    through the expansion ||x||^2 - 2 x.c + ||c||^2 (clamped at 0). The expansion
    is fast on matmul hardware but loses precision; use :func:`exact_sse` for
    reported objectives.
    """
    n, k = x.shape[0], c.shape[0]
    if c_norms is None:
        c_norms = sq_norms(c)
    labels = torch.empty(n, dtype=torch.int64, device=x.device)
    d2 = torch.empty(n, dtype=torch.float32, device=x.device)
    # keep the (rows x k) distance block around 2^28 floats (1 GB) at most
    rows = max(256, min(chunk_rows, (1 << 28) // max(k, 1)))
    for b in range(0, n, rows):
        xb = x[b : b + rows]
        dist = torch.addmm(c_norms[None, :], xb, c.T, alpha=-2.0)
        v, i = dist.min(dim=1)
        labels[b : b + rows] = i
        d2[b : b + rows] = v + sq_norms(xb)
    if count:
        OPS.add(n * k, tag, x.shape[1])
    return labels, d2.clamp_min_(0.0)


def assign_topm(
    x: torch.Tensor,
    c: torch.Tensor,
    m: int,
    c_norms: torch.Tensor | None = None,
    chunk_rows: int = 32768,
    count: bool = True,
    tag: str = "assign",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Brute-force m nearest centroids. Returns (indices [n, m], d2 [n, m]) sorted ascending."""
    n, k = x.shape[0], c.shape[0]
    m = min(m, k)
    if c_norms is None:
        c_norms = sq_norms(c)
    idx = torch.empty((n, m), dtype=torch.int64, device=x.device)
    d2 = torch.empty((n, m), dtype=torch.float32, device=x.device)
    rows = max(256, min(chunk_rows, (1 << 28) // max(k, 1)))
    for b in range(0, n, rows):
        xb = x[b : b + rows]
        dist = torch.addmm(c_norms[None, :], xb, c.T, alpha=-2.0)
        v, i = torch.topk(dist, m, dim=1, largest=False, sorted=True)
        idx[b : b + rows] = i
        d2[b : b + rows] = v + sq_norms(xb)[:, None]
    if count:
        OPS.add(n * k, tag, x.shape[1])
    return idx, d2.clamp_min_(0.0)


def assign_candidates(
    x: torch.Tensor,
    c: torch.Tensor,
    cand: torch.Tensor,
    count: bool = True,
    tag: str = "assign_cand",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest centroid among a per-point candidate list ``cand`` [n, m] (int64).

    Distances are computed directly (no expansion), chunked so the gathered
    centroid block stays around 2^27 floats.
    """
    n, m = cand.shape
    d = x.shape[1]
    labels = torch.empty(n, dtype=torch.int64, device=x.device)
    d2 = torch.empty(n, dtype=torch.float32, device=x.device)
    rows = max(256, (1 << 27) // max(m * d, 1))
    for b in range(0, n, rows):
        cb = cand[b : b + rows]
        xb = x[b : b + rows]
        diff = c[cb] - xb[:, None, :]
        dist = (diff * diff).sum(dim=2)
        v, i = dist.min(dim=1)
        labels[b : b + rows] = torch.gather(cb, 1, i[:, None])[:, 0]
        d2[b : b + rows] = v
    if count:
        OPS.add(n * m, tag, d)
    return labels, d2


def batch_stats(xb: torch.Tensor, labels: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-cluster counts (float32 [k]) and sums (float32 [k, d]) of a batch."""
    counts = torch.bincount(labels, minlength=k).to(torch.float32)
    sums = torch.zeros((k, xb.shape[1]), dtype=torch.float32, device=xb.device)
    sums.index_add_(0, labels, xb)
    return counts, sums


def exact_sse(
    x: torch.Tensor,
    c: torch.Tensor,
    labels: torch.Tensor | None = None,
    chunk_rows: int = 65536,
) -> tuple[float, torch.Tensor]:
    """Sum of squared distances to the assigned (nearest) centroid.

    Labels come from the matmul argmin (not counted as algorithmic distance
    computations); the distances themselves are recomputed directly as
    ||x - c_label||^2 in float32 and summed in float64 on the CPU, which avoids
    the cancellation error of the expansion. Returns (sse, labels).
    """
    if labels is None:
        labels, _ = assign_exact(x, c, count=False)
    total = 0.0
    n = x.shape[0]
    for b in range(0, n, chunk_rows):
        xb = x[b : b + chunk_rows]
        diff = xb - c[labels[b : b + chunk_rows]]
        part = (diff * diff).sum(dim=1)
        total += float(part.double().sum().item()) if part.device.type != "mps" else float(part.sum().item())
    return total, labels


def exact_mse(x: torch.Tensor, c: torch.Tensor, labels: torch.Tensor | None = None) -> float:
    sse, _ = exact_sse(x, c, labels)
    return sse / x.shape[0]


def random_rows_init(x: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    """k distinct rows of x, from a NumPy RNG so the draw does not depend on the device."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(x.shape[0])[:k]
    return x[torch.from_numpy(idx).to(x.device)].clone()


def kmeanspp_init(
    x: torch.Tensor,
    k: int,
    seed: int,
    greedy: bool = True,
    n_local_trials: int | None = None,
    sample: int | None = None,
    chunk_rows: int = 65536,
) -> torch.Tensor:
    """(Greedy) k-means++ seeding on the device, Arthur & Vassilvitskii (2007).

    ``greedy=True`` follows scikit-learn: at every step draw ``n_local_trials``
    (default 2 + log k) candidates proportionally to D^2 and keep the one that
    lowers the potential the most. ``sample`` restricts the seeding to a random
    subset of rows (the usual practical shortcut for large n*k). Distance
    computations are counted under the tag "init".
    """
    rng = np.random.RandomState(seed)
    if sample is not None and sample < x.shape[0]:
        sub = rng.permutation(x.shape[0])[:sample]
        x = x[torch.from_numpy(sub).to(x.device)]
    n, d = x.shape
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(k)) if greedy else 1
    gen = torch.Generator(device="cpu").manual_seed(int(rng.randint(2**31 - 1)))
    centers = torch.empty((k, d), dtype=x.dtype, device=x.device)
    first = int(rng.randint(n))
    centers[0] = x[first]
    xn = sq_norms(x)
    closest = (xn - 2 * (x @ centers[0]) + (centers[0] * centers[0]).sum()).clamp_min_(0.0)
    OPS.add(n, "init", d)
    pot = closest.sum()
    for i in range(1, k):
        # sample candidates proportional to D^2 (on CPU for reproducibility)
        probs = torch.clamp(closest.cpu().double(), min=0)
        if not torch.isfinite(probs).all() or probs.sum() <= 0:  # all remaining rows coincide with a centre
            probs = torch.ones_like(probs)
        probs /= probs.sum()
        cand = torch.multinomial(probs, n_local_trials, replacement=True, generator=gen).to(x.device)
        cc = x[cand]  # [L, d]
        dist = torch.addmm(sq_norms(cc)[None, :], x, cc.T, alpha=-2.0).add_(xn[:, None]).clamp_min_(0.0)  # [n, L]
        OPS.add(n * n_local_trials, "init", d)
        new_closest = torch.minimum(dist, closest[:, None])
        new_pot = new_closest.sum(dim=0)
        best = int(torch.argmin(new_pot).item())
        centers[i] = cc[best]
        closest = new_closest[:, best].contiguous()
        pot = new_pot[best]
    return centers


def pca_projection(x: torch.Tensor, dim: int | None = None, var_frac: float = 0.9, sample: int = 200_000,
                   seed: int = 0, min_dim: int = 8) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Orthonormal PCA basis P [d, d'] (and mean) from a sample of rows.

    ``dim`` fixes d'; otherwise d' is the smallest dimension retaining ``var_frac`` of the
    variance, clamped to [min_dim, d // 2]. Returns (P, mean, d').
    """
    n, d = x.shape
    if sample < n:
        idx = torch.from_numpy(np.random.RandomState(seed).permutation(n)[:sample]).to(x.device)
        xs = x[idx]
    else:
        xs = x
    xs64 = xs.cpu().double()
    mean = xs64.mean(dim=0)
    xc = xs64 - mean
    cov = xc.T @ xc / max(1, xc.shape[0] - 1)
    evals, evecs = torch.linalg.eigh(cov)  # ascending
    evals, evecs = evals.flip(0), evecs.flip(1)
    if dim is None:
        frac = torch.cumsum(evals.clamp_min(0), 0) / evals.clamp_min(0).sum()
        dim = int((frac < var_frac).sum().item()) + 1
        dim = max(min_dim, min(dim, max(min_dim, d // 2)))
    dim = min(dim, d)
    P = evecs[:, :dim].contiguous().float().to(x.device)
    return P, mean.float().to(x.device), dim


class PCAFilter:
    """Exact nearest-centroid search by PCA lower bounds and re-ranking.

    For an orthonormal P (d x d'), ||P^T (x - c)||^2 <= ||x - c||^2. Distances to all k
    centroids are computed in d' dimensions (k d' MACs per point), the ``cand`` smallest lower
    bounds are re-ranked exactly (cand d MACs), and the answer is certified if the next lower
    bound is not below the best exact distance. Uncertified points fall back to a full exact
    scan. The projected data ``xp`` is computed once per dataset.
    """

    def __init__(self, x: torch.Tensor, dim: int | None = None, cand: int = 32, var_frac: float = 0.9, seed: int = 0):
        self.P, self.mean, self.dim = pca_projection(x, dim, var_frac, seed=seed)
        self.cand = int(cand)
        self.xp = (x @ self.P).contiguous()
        self.xp_norms = sq_norms(self.xp)
        self.n_fallback = 0
        self.n_points = 0

    def project_centroids(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cp = c @ self.P
        return cp, sq_norms(cp)

    def assign(self, idx: torch.Tensor, xb: torch.Tensor, c: torch.Tensor, cp=None, cp_norms=None,
               chunk_rows: int = 32768, tag: str = "assign") -> tuple[torch.Tensor, torch.Tensor]:
        n, k = xb.shape[0], c.shape[0]
        d, dp = c.shape[1], self.dim
        if cp is None:
            cp, cp_norms = self.project_centroids(c)
        cc = min(self.cand, k - 1)
        labels = torch.empty(n, dtype=torch.int64, device=xb.device)
        d2 = torch.empty(n, dtype=torch.float32, device=xb.device)
        rows = max(256, min(chunk_rows, (1 << 28) // max(k, 1)))
        fallback_rows = []
        for b in range(0, n, rows):
            ib = idx[b : b + rows]
            xpb = self.xp[ib]
            lb = torch.addmm(cp_norms[None, :], xpb, cp.T, alpha=-2.0).add_(self.xp_norms[ib][:, None]).clamp_min_(0.0)
            vals, cand = torch.topk(lb, cc + 1, dim=1, largest=False, sorted=True)
            lab, best = assign_candidates(xb[b : b + rows], c, cand[:, :cc], count=False)
            labels[b : b + rows] = lab
            d2[b : b + rows] = best
            # certified when the (cc+1)-th lower bound is at least the best exact distance (with fp slack)
            ok = vals[:, cc] >= best * (1.0 - 1e-4) - 1e-6 * (1.0 + best)
            bad = torch.nonzero(~ok).flatten()
            if bad.numel():
                fallback_rows.append(bad + b)
        OPS.add(n * k, tag, dp)
        OPS.add(n * cc, tag, d)
        self.n_points += n
        if fallback_rows:
            bad = torch.cat(fallback_rows)
            lab, dd = assign_exact(xb[bad], c, sq_norms(c), tag=tag)
            # exact distances direct (not the expansion) for consistency
            labels[bad] = lab
            d2[bad] = ((xb[bad] - c[lab]) ** 2).sum(dim=1)
            self.n_fallback += bad.numel()
        return labels, d2
