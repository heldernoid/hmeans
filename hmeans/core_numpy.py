"""h-means on NumPy only (no torch): the final recipe, mirroring ``HMeansCore`` step for step.

Warm start (k/16 coarse EMA half-pass, SSE-proportional child allocation, two local Lloyd
iterations), e-means EMA passes with Hartigan-corrected assignment, exact Hartigan passes with
incremental sums, dead-cluster revival. Same random seeds and batch orders as the torch core, so the
two backends give the same result up to floating-point summation order. Distances go through BLAS
(``x @ c.T``); everything else is vectorised NumPy.
"""
from __future__ import annotations

import time

import numpy as np

from .laws import (
    CHECK_UNUSED_EVERY,
    INIT_USAGE,
    NO_REVIVAL_MAX_D,
    SPLIT_EPSILON,
    UNUSED_THRESHOLD,
    auto_ema_frac,
    epoch_rows,
    epoch_seed,
    resolve_batch_law,
    resolve_recipe,
    scheduled_alpha,
)

F32 = np.float32


def _sq_norms(a: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, a)


def _scatter_add(out: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    """out[idx] += vals (rows), without np.add.at's per-element loop."""
    if idx.size == 0:
        return
    o = np.argsort(idx, kind="stable")
    ls = idx[o]
    uniq, start = np.unique(ls, return_index=True)
    out[uniq] += np.add.reduceat(vals[o], start, axis=0)


def batch_stats(xb: np.ndarray, labels: np.ndarray, k: int):
    counts = np.bincount(labels, minlength=k).astype(F32)
    sums = np.zeros((k, xb.shape[1]), dtype=F32)
    _scatter_add(sums, labels, xb)
    return counts, sums


def assign_exact(x: np.ndarray, c: np.ndarray, c_norms: np.ndarray | None = None, chunk_rows: int = 32768):
    """Nearest centroid through ||x||^2 - 2 x.c + ||c||^2 (BLAS). Returns (labels int64, d2 float32)."""
    n, k = x.shape[0], c.shape[0]
    if c_norms is None:
        c_norms = _sq_norms(c)
    labels = np.empty(n, dtype=np.int64)
    d2 = np.empty(n, dtype=F32)
    rows = max(256, min(chunk_rows, (1 << 27) // max(k, 1)))
    ct = np.ascontiguousarray(c.T)
    for b in range(0, n, rows):
        xb = x[b : b + rows]
        dist = xb @ ct
        dist *= -2.0
        dist += c_norms[None, :]
        i = dist.argmin(axis=1)
        labels[b : b + rows] = i
        d2[b : b + rows] = dist[np.arange(xb.shape[0]), i] + _sq_norms(xb)
    np.maximum(d2, 0.0, out=d2)
    return labels, d2


def labels_and_sse(x: np.ndarray, c: np.ndarray, chunk_rows: int = 65536):
    """Nearest-centroid labels and the exact sum of squared distances (float64 accumulation)."""
    labels, _ = assign_exact(x, c, chunk_rows=chunk_rows)
    total = 0.0
    for b in range(0, x.shape[0], chunk_rows):
        diff = x[b : b + chunk_rows].astype(np.float64) - c[labels[b : b + chunk_rows]].astype(np.float64)
        total += float(np.einsum("ij,ij->", diff, diff))
    return labels, total


def exact_sse(x: np.ndarray, c: np.ndarray, chunk_rows: int = 65536) -> float:
    """Exact sum of squared distances to the nearest centroid (float64 accumulation)."""
    return labels_and_sse(x, c, chunk_rows)[1]


class HMeansNumpy:
    """See :class:`hmeans.HMeansCore` for the parameters; this class has the subset that defines the
    final recipe (``warm_start`` True/False, ``hartigan_ema``, ``ema_frac``, ``tail``, ``donor``)."""

    def __init__(self, k: int, nepochs: float = 5, batch_size=None, alpha0=None, batch_law="auto", unused_threshold=None,
                 seed: int = 1234, warm_start: bool | dict = True, donor: str = "sse", tail: str | None = "hartigan",
                 ema_frac: float | str = "auto", hartigan_ema: bool = True, verbose: bool = False):
        self.k = int(k)
        self.nepochs = float(nepochs)
        self.batch_size, self.alpha0, self.batch_law = batch_size, alpha0, batch_law
        self.unused_threshold = unused_threshold
        self.seed = int(seed)
        self.warm_start = dict(factor=16, epochs=0.5, local_iters=2, init_mass=1.0) if warm_start is True else (dict(warm_start) if warm_start else None)
        self.donor, self.tail = donor, tail
        self.ema_frac = ema_frac if ema_frac == "auto" else float(ema_frac)
        self.hartigan_ema = bool(hartigan_ema)
        self.verbose = verbose
        if ema_frac != "auto":
            self._set_split(self.ema_frac)
        self.ops_pairs_ = 0.0
        self.n_splits_ = 0
        self.history_ = []

    def _set_split(self, ema_frac: float):
        self.ema_frac_ = float(ema_frac)
        if self.tail is None:
            self.ema_epochs, self.tail_epochs = self.nepochs, 0.0
        else:
            self.ema_epochs = float(max(1, round(self.ema_frac_ * self.nepochs)))
            self.tail_epochs = self.nepochs - self.ema_epochs

    # ------------------------------------------------------------------ pieces
    def _revive(self, usage, sse, centroids, rng, tau):
        u = usage.astype(np.float64)
        threshold = tau * u.sum() / u.shape[0]
        unused = u < threshold
        donors = (~unused) & (u > 0)
        nsplit = min(int(unused.sum()), int(donors.sum()))
        if nsplit == 0:
            return 0
        weight = u if self.donor == "usage" else np.maximum(sse.astype(np.float64), 0) + 1e-30
        uniform = np.maximum(rng.random_sample(u.shape[0]), np.finfo("float64").tiny)
        keys = np.full(u.shape[0], -np.inf)
        keys[donors] = np.log(weight[donors]) - np.log(-np.log(uniform[donors]))
        src = np.argsort(keys)[-nsplit:][::-1].copy()
        tgt = np.flatnonzero(unused)[:nsplit]
        c = centroids[src]
        sign = np.ones(c.shape[1], dtype=F32)
        sign[1::2] = -1.0
        offset = F32(SPLIT_EPSILON) * sign
        donor_c = c * (1.0 - offset)
        child_c = c * (1.0 + offset)
        same = (donor_c == child_c).all(axis=1)
        if same.any():
            child_c[same, 0] = np.nextafter(child_c[same, 0], F32(np.inf))
        half = usage[src] / 2.0
        usage[src] = half
        usage[tgt] = half
        half_sse = sse[src] / 2.0
        sse[src] = half_sse
        sse[tgt] = half_sse
        centroids[src] = donor_c
        centroids[tgt] = child_c
        self.n_splits_ += nsplit
        return nsplit

    def _hartigan_assign(self, xb, c, N, old):
        nb, k = xb.shape[0], c.shape[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(N > 0, N / (N + 1.0), np.inf).astype(F32)
        empty_cols = ~np.isfinite(g)
        g_fin = np.where(empty_cols, 0.0, g).astype(F32)
        c_norms = _sq_norms(c)
        W = np.concatenate([g_fin[:, None], -2.0 * g_fin[:, None] * c, (g_fin * c_norms)[:, None]], axis=1).astype(F32)  # [k, d+2]
        Wt = np.ascontiguousarray(W.T)
        new = np.empty(nb, dtype=np.int64)
        d2 = np.empty(nb, dtype=F32)
        rows = max(256, (1 << 27) // max(k, 1))
        for b in range(0, nb, rows):
            xr = xb[b : b + rows]
            o = old[b : b + rows]
            xa = np.concatenate([_sq_norms(xr)[:, None], xr, np.ones((xr.shape[0], 1), dtype=F32)], axis=1)
            cost = xa @ Wt
            if empty_cols.any():
                cost[:, empty_cols] = np.inf
            diff = xr - c[o]
            d_old = np.einsum("ij,ij->i", diff, diff)
            n_old = N[o]
            stay = np.where(n_old > 1.0, d_old * n_old / np.maximum(n_old - 1.0, 1e-6), 0.0)
            cost[np.arange(xr.shape[0]), o] = stay
            j = cost.argmin(axis=1)
            new[b : b + rows] = j
            diff = xr - c[j]
            d2[b : b + rows] = np.einsum("ij,ij->i", diff, diff)
        self.ops_pairs_ += nb * k
        return new, d2

    def _ema_passes(self, x, centroids, usage, nepochs, bs, a0, tau, seed, stage="fine"):
        n, d = x.shape
        k = centroids.shape[0]
        rows = epoch_rows(n, nepochs)
        total_rows = sum(rows)
        sse = np.zeros(k, dtype=F32)
        split_rng = np.random.RandomState(seed)
        step = rows_done = 0
        t0 = time.perf_counter()
        last_labels = pass_counts = pass_sums = None
        prev_labels = np.full(n, -1, dtype=np.int64) if (self.hartigan_ema and stage == "fine") else None
        for e, r in enumerate(rows):
            last = e == len(rows) - 1
            order = np.random.RandomState(epoch_seed(seed, e)).permutation(n)
            pass_counts = np.zeros(k, dtype=F32)
            pass_sums = np.zeros((k, d), dtype=F32) if last else None
            if last:
                last_labels = np.empty(n, dtype=np.int64)
            objective = 0.0
            c_norms = _sq_norms(centroids)
            b0 = 0
            while b0 < r:
                idx = order[b0 : min(b0 + bs, r)]
                xb = x[idx]
                if prev_labels is not None and (prev_labels[idx] >= 0).all():
                    labels, d2 = self._hartigan_assign(xb, centroids, usage * F32(n / bs), prev_labels[idx])
                else:
                    labels, d2 = assign_exact(xb, centroids, c_norms)
                    self.ops_pairs_ += xb.shape[0] * k
                objective += float(d2.sum())
                if prev_labels is not None:
                    prev_labels[idx] = labels
                counts, sums = batch_stats(xb, labels, k)
                bsse = np.bincount(labels, weights=d2, minlength=k).astype(F32)
                pass_counts += counts
                if last:
                    pass_sums += sums
                    last_labels[idx] = labels
                alpha = F32(scheduled_alpha(a0, min(1.0, rows_done / total_rows), 1.0))
                touched = counts > 0
                usage *= 1.0 - alpha
                sse *= 1.0 - alpha
                mass = usage[touched] + alpha * counts[touched]
                mean = sums[touched] / counts[touched, None]
                w = (alpha * counts[touched] / mass)[:, None]
                centroids[touched] += w * (mean - centroids[touched])
                usage[touched] = mass
                sse[touched] += alpha * bsse[touched]
                step += 1
                rows_done += idx.shape[0]
                b0 += idx.shape[0]
                if tau > 0 and step % CHECK_UNUSED_EVERY == 0:
                    self._revive(usage, sse, centroids, split_rng, tau)
                c_norms = _sq_norms(centroids)
            self.history_.append(dict(stage=stage, epoch=sum(rows[: e + 1]) / n, moving_obj=objective, time=time.perf_counter() - t0,
                                      splits=self.n_splits_, ops=self.ops_pairs_, steps=step))
        return centroids, pass_counts, pass_sums, last_labels, sse

    @staticmethod
    def _allocate(weight, cnt, k):
        """Largest-remainder apportionment of k children proportional to ``weight``; at least one per
        non-empty coarse cluster and at most its point count."""
        kc = weight.shape[0]
        w = weight.astype(np.float64)
        cnt = cnt.astype(np.float64)
        alive = cnt > 0
        cap = np.where(alive, np.maximum(cnt, 1.0), 0.0)
        if w.sum() <= 0:
            w = alive.astype(np.float64)
        alloc = np.zeros(kc, dtype=np.int64)
        alloc[alive] = 1
        remaining = k - int(alloc.sum())
        if remaining < 0:
            alloc[:] = 0
            alloc[np.argsort(w)[-k:]] = 1
            return alloc
        for _ in range(8):
            room = np.maximum(cap - alloc, 0)
            share = w * (room > 0)
            if share.sum() <= 0 or remaining <= 0:
                break
            quota = share / share.sum() * remaining
            give = np.minimum(np.floor(quota), room).astype(np.int64)
            alloc += give
            remaining -= int(give.sum())
            if remaining <= 0:
                break
            frac = quota - np.floor(quota)
            frac = np.where((cap - alloc) > 0, frac, -1.0)
            m = min(remaining, int((frac >= 0).sum()))
            take = np.argsort(-frac, kind="stable")[:m]
            alloc[take] += 1
            remaining -= m
        return alloc

    def _warm_start(self, x, k, bs, a0, tau):
        ws = self.warm_start
        f = int(ws.get("factor", 16))
        kc = max(1, k // f)
        ec = float(ws.get("epochs", 0.5))
        n, d = x.shape
        law = resolve_batch_law(self.batch_law, d)
        bsc, a0c = resolve_recipe(n, kc, max(ec, 0.25), law, None, None)
        t = time.perf_counter()
        coarse = x[np.random.RandomState(self.seed).permutation(n)[:kc]].copy()
        usage_c = np.full(kc, INIT_USAGE, dtype=F32)
        coarse, _, _, _, _ = self._ema_passes(x, coarse, usage_c, ec, bsc, a0c, tau, self.seed + 7, stage="coarse")
        labels_c, d2_c = assign_exact(x, coarse)
        self.ops_pairs_ += n * kc
        cnt = np.bincount(labels_c, minlength=kc).astype(F32)
        weight = np.bincount(labels_c, weights=d2_c, minlength=kc)
        weight = np.where(cnt > 0, weight, 0.0)
        nchild = self._allocate(weight, cnt, k)
        f = int(nchild.max())
        rng = np.random.RandomState(self.seed + 11)
        keys = rng.random_sample(n)
        order = np.lexsort((keys, labels_c))
        sorted_lab = labels_c[order]
        bounds = np.searchsorted(sorted_lab, np.arange(kc + 1))
        sizes = np.diff(bounds)
        child_rows = np.full((kc, f), -1, dtype=np.int64)
        rank = np.arange(f)[None, :]
        take = (rank < sizes[:, None]) & (rank < nchild[:, None])
        pos = bounds[:-1][:, None] + rank
        child_rows[take] = order[pos[take]]
        children = np.repeat(coarse[:, None, :], f, axis=1).copy()  # [kc, f, d]
        valid = child_rows >= 0
        children[valid] = x[child_rows[valid]]
        slot = rank < nchild[:, None]
        miss_mask = slot & ~valid
        if miss_mask.any():
            sign = np.ones(d, dtype=F32)
            sign[1::2] = -1.0
            mi, mj = np.nonzero(miss_mask)
            children[mi, mj] = coarse[mi] * (1 + F32(SPLIT_EPSILON) * (1 + mj[:, None].astype(F32)) * sign)
        child_counts = None
        lab_local = None
        rows = max(1024, (1 << 26) // (f * d))
        for _ in range(int(ws.get("local_iters", 2))):
            lab_local = np.empty(n, dtype=np.int64)
            for b in range(0, n, rows):
                lc = labels_c[b : b + rows]
                diff = children[lc] - x[b : b + rows, None, :]
                dist = np.einsum("ijk,ijk->ij", diff, diff)
                dist[~slot[lc]] = np.inf
                lab_local[b : b + rows] = dist.argmin(axis=1)
            self.ops_pairs_ += n * f
            flat = labels_c * f + lab_local
            child_counts, child_sums = batch_stats(x, flat, kc * f)
            nonempty = child_counts > 0
            flat_children = children.reshape(kc * f, d)
            flat_children[nonempty] = child_sums[nonempty] / child_counts[nonempty, None]
            children = flat_children.reshape(kc, f, d)
        slot_ids = np.flatnonzero(slot.reshape(-1))
        centroids = np.ascontiguousarray(children.reshape(kc * f, d)[slot_ids])
        counts = child_counts[slot_ids]
        rho = float(ws.get("init_mass", 1.0))
        usage = np.maximum(rho * counts * F32(bs / n), INIT_USAGE).astype(F32)
        self.warm_time_ = time.perf_counter() - t
        if self.verbose:
            print(f"warm start: kc={kc} f={f} time={self.warm_time_:.2f}s ops={self.ops_pairs_ / (n * k):.3f} passes")
        return centroids, usage

    def _tail_revive(self, x, c, S, N, labels, empty, rng):
        probs = np.where(N > 1, N / N.sum(), 0.0).astype(np.float64)
        if probs.sum() <= 0:
            return
        probs /= probs.sum()
        n_rev = min(empty.shape[0], int((probs > 0).sum()))
        donors = rng.choice(N.shape[0], size=n_rev, replace=False, p=probs)
        for e_idx, dnr in zip(empty[:n_rev].tolist(), donors.tolist()):
            members = np.flatnonzero(labels == dnr)
            if members.shape[0] < 2:
                continue
            diff = x[members] - c[dnr]
            far = members[np.einsum("ij,ij->i", diff, diff).argmax()]
            xv = x[far]
            S[dnr] -= xv
            N[dnr] -= 1.0
            S[e_idx] = xv
            N[e_idx] = 1.0
            labels[far] = e_idx
            c[dnr] = S[dnr] / N[dnr]
            c[e_idx] = xv
            self.n_splits_ += 1

    def _tail_passes(self, x, centroids, S, N, labels, bs, t_start):
        n, d = x.shape
        k = centroids.shape[0]
        hartigan = self.tail == "hartigan"
        nonempty = N > 0
        c = centroids.copy()
        c[nonempty] = S[nonempty] / N[nonempty, None]
        rows = epoch_rows(n, self.tail_epochs)
        seed = self.seed + 101
        split_rng = np.random.RandomState(seed)
        step = 0
        ar = None
        for e, r in enumerate(rows):
            order = np.random.RandomState(epoch_seed(seed, e)).permutation(n)
            objective = 0.0
            changed_epoch = 0
            for b0 in range(0, r, bs):
                idx = order[b0 : min(b0 + bs, r)]
                xb = x[idx]
                old = labels[idx]
                if hartigan:
                    new, d2 = self._hartigan_assign(xb, c, N, old)
                else:
                    new, d2 = assign_exact(xb, c)
                    self.ops_pairs_ += xb.shape[0] * k
                objective += float(d2.sum())
                moved = new != old
                nm = int(moved.sum())
                if nm:
                    xm = xb[moved]
                    om, nw = old[moved], new[moved]
                    _scatter_add(S, om, -xm)
                    _scatter_add(S, nw, xm)
                    N -= np.bincount(om, minlength=k).astype(F32)
                    N += np.bincount(nw, minlength=k).astype(F32)
                    labels[idx[moved]] = nw
                    touched = np.unique(np.concatenate([om, nw]))
                    ne = touched[N[touched] > 0]
                    c[ne] = S[ne] / N[ne, None]
                    changed_epoch += nm
                step += 1
                if step % CHECK_UNUSED_EVERY == 0:
                    empty = np.flatnonzero(N <= 0)
                    if empty.shape[0]:
                        self._tail_revive(x, c, S, N, labels, empty, split_rng)
            self.history_.append(dict(stage="tail", epoch=self.ema_epochs + sum(rows[: e + 1]) / n, moving_obj=objective,
                                      time=time.perf_counter() - t_start, ops=self.ops_pairs_, changed=changed_epoch / n))
        nonempty = N > 0
        c[nonempty] = S[nonempty] / N[nonempty, None]
        return c, labels

    # --------------------------------------------------------------------- API
    def fit(self, x):
        x = np.ascontiguousarray(x, dtype=F32)
        n, d = x.shape
        k = self.k
        if self.ema_frac == "auto":
            self._set_split(auto_ema_frac(n, k))
        law = resolve_batch_law(self.batch_law, d)
        bs, a0 = resolve_recipe(n, k, self.ema_epochs, law, self.batch_size, self.alpha0)
        tau = (0.0 if d <= NO_REVIVAL_MAX_D else UNUSED_THRESHOLD) if self.unused_threshold is None else self.unused_threshold
        self.recipe_ = dict(batch_size=bs, alpha0=a0, batch_law=law, tau=tau, ema_frac=self.ema_frac_, ema_epochs=self.ema_epochs, tail_epochs=self.tail_epochs, tail=self.tail)
        t_start = time.perf_counter()
        self.history_ = []
        self.ops_pairs_ = 0.0
        if self.warm_start:
            centroids, usage = self._warm_start(x, k, bs, a0, tau)
        else:
            centroids = x[np.random.RandomState(self.seed).permutation(n)[:k]].copy()
            usage = np.full(k, INIT_USAGE, dtype=F32)
        centroids, pass_counts, pass_sums, labels, _ = self._ema_passes(x, centroids, usage, self.ema_epochs, bs, a0, tau, self.seed)
        nonempty = pass_counts > 0
        final = centroids.copy()
        final[nonempty] = pass_sums[nonempty] / pass_counts[nonempty, None]
        if self.tail and self.tail_epochs > 0:
            final, labels = self._tail_passes(x, final, pass_sums.astype(F32).copy(), pass_counts.copy(), labels, bs, t_start)
        self.centroids_ = final
        self.labels_ = labels
        self.fit_time_ = time.perf_counter() - t_start
        return self

    def predict(self, x):
        return assign_exact(np.ascontiguousarray(x, dtype=F32), self.centroids_)[0]
