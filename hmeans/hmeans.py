"""h-means: hierarchical warm start + e-means EMA passes + exact Hartigan passes.

Recipe (defaults): (1) cluster into k/16 coarse centroids with half an EMA pass,
allocate children to coarse clusters proportionally to their SSE and place them
with two local Lloyd iterations (~0.1 pass of distance computations); (2) run
e-means EMA passes (its laws, cosine schedule, revival with SSE-proportional
donors) for 40% of the pass budget; (3) finish with exact Hartigan passes: every
cluster's sum and count over all points are maintained exactly, each batch is
re-assigned with Hartigan's rule (move x from a to b iff
N_a/(N_a-1) d2(x,a) > N_b/(N_b+1) d2(x,b), one augmented matmul) and the affected
sums are updated. With ``warm_start=None, donor="usage", tail=None`` it is exactly
``EMeansRef``. Other switches below are kept for the ablations in NOTES.md.

Components (each has an ablation switch):

* ``warm_start``   hierarchical seeding: cluster into k/f coarse centroids with the
                   same EMA loop for a fraction of an epoch, split every coarse
                   centroid into f children by a few local Lloyd iterations
                   (each point only sees the f children of its coarse cluster),
                   and give the children an initial EMA mass so the first noisy
                   mini-batches do not erase the seeding.
* ``assign="bounded"`` exact nearest-centroid assignment that reuses each
                   point's cached candidate list from its previous visit: every
                   centroid's cumulative path length is tracked, the few
                   centroids that moved most since the point's last full scan
                   are checked explicitly ("hot" set), and a triangle-inequality
                   bound certifies the cached candidates for the rest. Points
                   failing the bound get a full scan. No approximation.
* ``polish``       after finalization, p cheap exact Lloyd iterations using the
                   same bounded assignment (each costs a small fraction of a pass).
* ``donor``        revival donors sampled proportionally to EMA mass (e-means) or
                   to the cluster's running SSE (``"sse"``).
* ``batch_grow``   geometric batch-size warm-up during the first pass, with the
                   step law applied to the current batch size.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from . import backend as B
from .laws import auto_ema_frac
from .emeans_ref import (
    CHECK_UNUSED_EVERY,
    INIT_USAGE,
    NO_REVIVAL_MAX_D,
    SPLIT_EPSILON,
    UNUSED_THRESHOLD,
    epoch_rows,
    epoch_seed,
    resolve_batch_law,
    resolve_recipe,
    scheduled_alpha,
    step_law,
)


class _CachedAssigner:
    """Assignment through per-point cached candidate lists (see module doc).

    mode="bounded": exact. A point's cached candidates (its m nearest centroids at
    its last full scan) plus the h centroids that moved most since then are
    checked explicitly; a triangle-inequality bound certifies the result, and
    uncertified points get a full scan.
    mode="cached": approximate. Same candidate evaluation, but uncertified points
    keep the best candidate; only points whose candidates all drifted beyond the
    old m-th distance (a sign the neighbourhood changed) and a random fraction
    ``refresh`` of each batch get a full scan.
    """

    def __init__(self, n, k, m, hot, max_steps, device, mode="bounded", refresh=0.0, seed=0):
        self.n, self.k, self.m, self.h = n, k, min(m, k), min(hot, max(k - m, 0))
        self.device, self.mode, self.refresh = device, mode, float(refresh)
        self.cand = torch.zeros((n, self.m), dtype=torch.int64, device=device)
        self.dm = torch.full((n,), -1.0, dtype=torch.float32, device=device)
        self.t0 = torch.full((n,), -1, dtype=torch.int64, device=device)
        self.cum = torch.zeros((max_steps + 2, k), dtype=torch.float32, device=device)
        self.step = 0
        self.n_full = 0
        self.n_cheap = 0
        self.n_certified = 0
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def record_move(self, old_c, new_c):
        """Call after every centroid update (including splits): advances the step clock."""
        move = (new_c - old_c).norm(dim=1)
        if self.step + 1 >= self.cum.shape[0]:
            self.cum = torch.cat([self.cum, torch.zeros_like(self.cum)], dim=0)
        self.cum[self.step + 1] = self.cum[self.step] + move
        self.step += 1

    def full_scan(self, idx, xb, c, c_norms):
        cand, d2 = B.assign_topm(xb, c, self.m, c_norms)
        self.cand[idx] = cand
        self.dm[idx] = d2[:, -1]
        self.t0[idx] = self.step
        self.n_full += idx.numel()
        return cand[:, 0], d2[:, 0]

    def assign(self, idx, xb, c, c_norms, exact=False):
        """Returns (labels, d2) for the rows ``xb`` (global ids ``idx``)."""
        nb = idx.shape[0]
        labels = torch.empty(nb, dtype=torch.int64, device=self.device)
        d2 = torch.empty(nb, dtype=torch.float32, device=self.device)
        t0 = self.t0[idx]
        fresh = t0 < 0
        if self.refresh > 0 and not exact:
            fresh = fresh | (torch.rand(nb, generator=self.gen) < self.refresh).to(self.device)
        if bool(fresh.all()):
            labels[:], d2[:] = self.full_scan(idx, xb, c, c_norms)
            return labels, d2
        if bool(fresh.any()):
            f = torch.nonzero(fresh).flatten()
            labels[f], d2[f] = self.full_scan(idx[f], xb[f], c, c_norms)
        seen = torch.nonzero(~fresh).flatten()
        groups, inverse = torch.unique(t0[seen], return_inverse=True)
        order = torch.argsort(inverse)
        seen = seen[order]
        counts = torch.bincount(inverse[order], minlength=groups.numel()).tolist()
        start = 0
        failed = []
        for g, cnt in zip(groups.tolist(), counts):
            rows = seen[start : start + cnt]  # positions within the batch
            gids = idx[rows]  # global row ids (index the per-point cache)
            start += cnt
            mv = self.cum[self.step] - self.cum[g]  # movement since that scan, per centroid
            if self.h > 0:
                top = torch.topk(mv, self.h + 1)
                hot = top.indices[: self.h]
                theta = top.values[-1]
                cand = torch.cat([self.cand[gids], hot[None, :].expand(rows.numel(), -1)], dim=1)
            else:
                theta = mv.max()
                cand = self.cand[gids]
            lab, best = B.assign_candidates(xb[rows], c, cand)
            dm = self.dm[gids]
            certified = best.sqrt() <= dm.sqrt() - theta
            self.n_certified += int(certified.sum().item())
            if self.mode == "bounded" or exact:
                ok = certified
            else:
                ok = best <= dm  # suspicious if every cached candidate drifted beyond the old m-th distance
            labels[rows] = lab
            d2[rows] = best
            bad = rows[~ok]
            if bad.numel():
                failed.append(bad)
            self.n_cheap += int(ok.sum().item())
        if failed:
            bad = torch.cat(failed)
            labels[bad], d2[bad] = self.full_scan(idx[bad], xb[bad], c, c_norms)
        return labels, d2


_BoundedAssigner = _CachedAssigner


class _GraphAssigner:
    """Assignment through a k-NN graph over the centroids (see module doc).

    The graph ``G`` [k, m] holds every centroid's m nearest other centroids. It
    is rebuilt exactly every ``rebuild_every`` updates (k*k distance pairs) and
    refined in between with one NN-descent step (candidates = neighbours and
    neighbours of neighbours, k*(m+m^2) pairs). A point is assigned by evaluating
    its previous centroid and that centroid's m neighbours (m+1 pairs); if the
    best changed, a second hop from the new best is taken. Points without a
    previous label get a full scan. ``probe`` rows per batch are also assigned
    exactly (not counted) to measure the miss rate.
    """

    def __init__(self, n, k, m, device, rebuild_every=5, refine=True, probe=1024, seed=0, hops=2, gamma=0.0):
        self.n, self.k, self.m = n, k, min(m, k - 1)
        self.device, self.rebuild_every, self.refine, self.probe, self.hops = device, rebuild_every, refine, probe, hops
        self.gamma = float(gamma)  # fallback to a full scan when d2 > gamma * cluster mean d2 (0 disables)
        self.r2 = None  # per-cluster mean squared distance, set by the caller before each batch
        self.n_fallback = 0
        # optional two-level structure for the first visit of a point: parents [kc, d] (mean of their
        # children) and children_of [kc, f]; first visits probe the children of the nprobe nearest parents
        self.children_of = None
        self.nprobe = 4
        self.n_ivf = 0
        self.prev = torch.full((n,), -1, dtype=torch.int64, device=device)
        self.G = None
        self.step = 0
        self.n_full = 0
        self.n_cheap = 0
        self.n_hop2 = 0
        self.missed = []
        self.miss_excess = []
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    # ------------------------------------------------------------- graph
    def rebuild(self, c):
        k, m = self.k, self.m
        c_norms = B.sq_norms(c)
        G = torch.empty((k, m), dtype=torch.int64, device=self.device)
        rows = max(256, (1 << 27) // k)
        for b in range(0, k, rows):
            dist = torch.addmm(c_norms[None, :], c[b : b + rows], c.T, alpha=-2.0)
            ar = torch.arange(b, min(b + rows, k), device=self.device)
            dist[torch.arange(ar.numel(), device=self.device), ar] = float("inf")  # exclude self
            G[b : b + rows] = torch.topk(dist, m, dim=1, largest=False).indices
        B.OPS.add(k * k, "graph")
        self.G = G

    def refine_step(self, c):
        """One NN-descent step: candidates are neighbours and neighbours of neighbours."""
        k, m = self.k, self.m
        G = self.G
        cand = torch.cat([G, G[G].reshape(k, m * m)], dim=1)  # [k, m + m^2]
        rows = max(64, (1 << 26) // (cand.shape[1] * c.shape[1]))
        newG = torch.empty_like(G)
        for b in range(0, k, rows):
            cb = cand[b : b + rows]
            diff = c[cb] - c[b : b + rows, None, :]
            dist = (diff * diff).sum(dim=2)
            self_mask = cb == torch.arange(b, min(b + rows, k), device=self.device)[:, None]
            dist[self_mask] = float("inf")
            # duplicates: sort by (dist) then unique-by-id is costly; approximate by sorting distances and
            # removing repeated ids with a stable trick (id + tiny rank) -> use topk on dist and dedupe after
            v, i = torch.topk(dist, min(3 * m, cb.shape[1]), dim=1, largest=False)
            ids = torch.gather(cb, 1, i)
            # dedupe per row: mark later duplicates by comparing against sorted ids
            s_ids, s_ord = torch.sort(ids, dim=1)
            dup = torch.zeros_like(ids, dtype=torch.bool)
            dup_sorted = torch.zeros_like(s_ids, dtype=torch.bool)
            dup_sorted[:, 1:] = s_ids[:, 1:] == s_ids[:, :-1]
            dup.scatter_(1, s_ord, dup_sorted)
            v = torch.where(dup, torch.full_like(v, float("inf")), v)
            v2, i2 = torch.topk(v, m, dim=1, largest=False)
            newG[b : b + rows] = torch.gather(ids, 1, i2)
        B.OPS.add(k * cand.shape[1], "graph")
        self.G = newG

    def on_split(self, src, tgt):
        """Children inherit the donor's neighbours and point back to the donor; donors learn the child."""
        if self.G is None:
            return
        self.G[tgt] = self.G[src]
        self.G[tgt, -1] = src
        self.G[src, -1] = tgt

    def after_update(self, c):
        self.step += 1
        if self.G is None or (self.rebuild_every and self.step % self.rebuild_every == 0):
            self.rebuild(c)
        elif self.refine:
            self.refine_step(c)

    # -------------------------------------------------------------- assign
    def _hop(self, xb, c, start):
        cand = torch.cat([start[:, None], self.G[start]], dim=1)  # [b, m+1]
        return B.assign_candidates(xb, c, cand)

    def assign(self, idx, xb, c, c_norms):
        if self.G is None:
            self.rebuild(c)
        nb = idx.shape[0]
        labels = torch.empty(nb, dtype=torch.int64, device=self.device)
        d2 = torch.empty(nb, dtype=torch.float32, device=self.device)
        prev = self.prev[idx]
        fresh = prev < 0
        if bool(fresh.any()):
            f = torch.nonzero(fresh).flatten()
            if self.children_of is not None:
                lab, dd = self._ivf_assign(xb[f], c)
                self.n_ivf += f.numel()
            else:
                lab, dd = B.assign_exact(xb[f], c, c_norms)
                self.n_full += f.numel()
            labels[f], d2[f] = lab, dd
        seen = torch.nonzero(~fresh).flatten()
        if seen.numel():
            xs = xb[seen]
            lab, best = self._hop(xs, c, prev[seen])
            for _ in range(self.hops - 1):
                changed = torch.nonzero(lab != prev[seen]).flatten() if _ == 0 else torch.nonzero(lab != lab_prev).flatten()
                if changed.numel() == 0:
                    break
                lab_prev = lab.clone()
                lab2, best2 = self._hop(xs[changed], c, lab[changed])
                lab[changed], best[changed] = lab2, best2
                self.n_hop2 += changed.numel()
            if self.gamma > 0 and self.r2 is not None:
                far = torch.nonzero(best > self.gamma * self.r2[lab]).flatten()
                if far.numel():
                    lab_f, d_f = B.assign_exact(xs[far], c, c_norms)
                    lab[far], best[far] = lab_f, d_f
                    self.n_fallback += far.numel()
            labels[seen], d2[seen] = lab, best
            self.n_cheap += seen.numel()
            if self.probe and seen.numel() > 0:
                p = min(self.probe, seen.numel())
                sel = seen[torch.randperm(seen.numel(), generator=self.gen)[:p].to(self.device)]
                lab_e, d2_e = B.assign_exact(xb[sel], c, c_norms, count=False)
                miss = lab_e != labels[sel]
                self.missed.append(float(miss.float().mean().item()))
                self.miss_excess.append(float(((d2[sel] - d2_e).clamp_min(0) / d2_e.clamp_min(1e-12)).mean().item()))
        self.prev[idx] = labels
        return labels, d2

    def _ivf_assign(self, xf, c):
        """Two-level search: nearest ``nprobe`` parents (parent = mean of its children), then their children."""
        kc, f = self.children_of.shape
        parents = c[self.children_of.clamp_max(self.k - 1)].mean(dim=1)  # [kc, d]
        p_norms = B.sq_norms(parents)
        labels = torch.empty(xf.shape[0], dtype=torch.int64, device=self.device)
        d2 = torch.empty(xf.shape[0], dtype=torch.float32, device=self.device)
        rows = max(256, (1 << 27) // max(kc, 1))
        npb = min(self.nprobe, kc)
        for b in range(0, xf.shape[0], rows):
            xr = xf[b : b + rows]
            dist = torch.addmm(p_norms[None, :], xr, parents.T, alpha=-2.0)
            cells = torch.topk(dist, npb, dim=1, largest=False).indices  # [rows, npb]
            cand = self.children_of[cells].reshape(xr.shape[0], npb * f).clamp_max(self.k - 1)
            lab, dd = B.assign_candidates(xr, c, cand)
            labels[b : b + rows], d2[b : b + rows] = lab, dd
        B.OPS.add(xf.shape[0] * kc, "assign")
        if self.gamma > 0 and self.r2 is not None:
            far = torch.nonzero(d2 > self.gamma * self.r2[labels]).flatten()
            if far.numel():
                lab_f, d_f = B.assign_exact(xf[far], c)
                labels[far], d2[far] = lab_f, d_f
                self.n_fallback += far.numel()
        return labels, d2

    def stats(self):
        return dict(assign_full=self.n_full, assign_ivf=self.n_ivf, assign_cheap=self.n_cheap, assign_hop2=self.n_hop2, assign_fallback=self.n_fallback,
                    miss_rate=float(np.mean(self.missed)) if self.missed else None,
                    miss_excess=float(np.mean(self.miss_excess)) if self.miss_excess else None)



class HMeansCore:
    """The h-means algorithm on torch tensors, with every ablation switch. ``hmeans.HMeans`` is the
    scikit-learn style estimator built on it."""
    DEFAULT_WARM_START = dict(factor=16, epochs=0.5, local_iters=2, alloc="sse", init_mass=1.0)

    def __init__(
        self,
        k: int,
        nepochs: float = 5,
        batch_size: int | None = None,
        alpha0: float | None = None,
        batch_law: str = "auto",
        unused_threshold: float | None = None,
        seed: int = 1234,
        device: str | torch.device = "auto",
        verbose: bool = False,
        # h-means components (defaults = the h-means recipe; pass warm_start=None, donor="usage",
        # tail=None to recover e-means)
        warm_start: dict | None | str = "default",
        assign: str = "exact",
        m: int = 32,
        hot: int = 32,
        refresh: float = 0.0,
        graph_rebuild_every: int = 5,
        graph_hops: int = 2,
        graph_gamma: float = 2.0,
        pca_dim: int | None = None,
        pca_cand: int = 32,
        pca_var: float = 0.9,
        polish: int = 0,
        polish_mode: str = "exact",
        donor: str = "usage",
        batch_grow: dict | None = None,
        finalize: str = "last_epoch_mean",
        tail: str | None = "hartigan",
        tail_epochs: float | None = None,
        ema_frac: float | str = "auto",
        tail_batch: int | None = None,
        hartigan_ema: bool = True,
    ):
        """``nepochs`` is the total number of passes over the data (not counting the warm start,
        which costs about 0.1 pass). The first ``ema_epochs = max(1, round(ema_frac * nepochs))``
        passes are e-means EMA passes (``tail_epochs`` given explicitly overrides this split), the
        remaining passes are exact Hartigan passes. ``hartigan_ema`` applies Hartigan's N/(N±1)
        correction to the assignment inside the EMA passes too, using usage·n/B as the size
        estimate. The defaults are the final recipe of REPORT.md ("v3"); v1 = hartigan_ema=False,
        v2 = ema_frac=0.8."""
        if warm_start == "default":
            warm_start = dict(self.DEFAULT_WARM_START)
        if donor == "sse" and warm_start is None and tail is None:
            pass
        self.k = int(k)
        self.nepochs = float(nepochs)
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.batch_law = batch_law
        self.unused_threshold = unused_threshold
        self.seed = int(seed)
        self.device = B.pick_device(device) if isinstance(device, str) else device
        self.verbose = verbose
        self.warm_start = warm_start
        self.assign = assign
        self.m, self.hot, self.refresh = int(m), int(hot), float(refresh)
        self.graph_rebuild_every, self.graph_hops, self.graph_gamma = int(graph_rebuild_every), int(graph_hops), float(graph_gamma)
        self.pca_dim, self.pca_cand, self.pca_var = pca_dim, int(pca_cand), float(pca_var)
        self.polish, self.polish_mode = int(polish), polish_mode
        self.donor = donor
        self.batch_grow = batch_grow
        self.finalize = finalize
        self.tail, self.tail_batch = tail, tail_batch
        self.hartigan_ema = bool(hartigan_ema)
        self.ema_frac = ema_frac if ema_frac == "auto" else float(ema_frac)
        self._tail_epochs_arg = tail_epochs
        if ema_frac != "auto":
            self._set_split(self.ema_frac)
        self.history_ = []
        self.n_splits_ = 0
        self.extra_ = {}

    def _set_split(self, ema_frac: float):
        """EMA / exact-tail split of the pass budget."""
        self.ema_frac_ = float(ema_frac)
        if self.tail is None:
            self.ema_epochs, self.tail_epochs = self.nepochs, 0.0
        elif self._tail_epochs_arg is not None:
            self.tail_epochs = float(self._tail_epochs_arg)
            self.ema_epochs = self.nepochs if self.tail_epochs == 0 else self.nepochs - self.tail_epochs
            if self.ema_epochs < 1:  # legacy call style: nepochs = EMA passes, tail added on top
                self.ema_epochs, self.nepochs = self.nepochs, self.nepochs + self.tail_epochs
        else:
            self.ema_epochs = float(max(1, round(self.ema_frac_ * self.nepochs)))
            self.tail_epochs = self.nepochs - self.ema_epochs

    # ------------------------------------------------------------------ revival
    def _revive(self, usage, sse, centroids, rng, tau, assigner):
        u = usage.cpu().numpy().astype(np.float64)
        threshold = tau * u.sum() / usage.shape[0]
        unused = u < threshold
        donors = (~unused) & (u > 0)
        nsplit = min(int(unused.sum()), int(donors.sum()))
        if nsplit == 0:
            return 0
        weight = u if self.donor == "usage" else np.maximum(sse.cpu().numpy().astype(np.float64), 0) + 1e-30
        uniform = np.maximum(rng.random_sample(usage.shape[0]), np.finfo("float64").tiny)
        keys = np.full(usage.shape[0], -np.inf)
        keys[donors] = np.log(weight[donors]) - np.log(-np.log(uniform[donors]))
        source = np.argsort(keys)[-nsplit:][::-1].copy()
        target = np.flatnonzero(unused)[:nsplit]
        src = torch.from_numpy(source).to(self.device)
        tgt = torch.from_numpy(target).to(self.device)
        old = centroids.clone() if isinstance(assigner, _CachedAssigner) else None
        c = centroids[src]
        sign = torch.ones(c.shape[1], device=self.device)
        sign[1::2] = -1.0
        offset = SPLIT_EPSILON * sign
        donor_c = c * (1.0 - offset)
        child_c = c * (1.0 + offset)
        same = (donor_c == child_c).all(dim=1)
        if bool(same.any()):
            child_c[same, 0] = torch.nextafter(child_c[same, 0], torch.full_like(child_c[same, 0], float("inf")))
        half = usage[src] / 2.0
        usage[src] = half
        usage[tgt] = half
        half_sse = sse[src] / 2.0
        sse[src] = half_sse
        sse[tgt] = half_sse
        centroids[src] = donor_c
        centroids[tgt] = child_c
        if isinstance(assigner, _CachedAssigner):
            assigner.record_move(old, centroids)
        elif isinstance(assigner, _GraphAssigner):
            assigner.on_split(src, tgt)
        self.n_splits_ += nsplit
        return nsplit

    # ------------------------------------------------------------ EMA passes
    def _ema_passes(self, x, centroids, usage, nepochs, bs, a0, tau, assigner, seed, callback=None,
                    stage="fine", grow=None, schedule_offset=0.0):
        """Run ``nepochs`` passes of EMA mini-batch updates. Returns (centroids, pass_counts, pass_sums, labels)."""
        n, d = x.shape
        k = centroids.shape[0]
        rows = epoch_rows(n, nepochs)
        total_rows = sum(rows)
        sse = torch.zeros(k, dtype=torch.float32, device=self.device)
        split_rng = np.random.RandomState(seed)
        step = 0
        rows_done = 0
        t0 = time.perf_counter()
        last_labels = None
        pass_counts = pass_sums = None
        prev_labels = torch.full((n,), -1, dtype=torch.int64, device=self.device) if (self.hartigan_ema and stage == "fine") else None
        # batch schedule: list of batch sizes for the first pass when growing
        for e, r in enumerate(rows):
            last = e == len(rows) - 1
            order = torch.from_numpy(np.random.RandomState(epoch_seed(seed, e)).permutation(n)).to(self.device)
            pass_counts = torch.zeros(k, dtype=torch.float32, device=self.device)
            pass_sums = torch.zeros((k, d), dtype=torch.float32, device=self.device) if last else None
            if last:
                last_labels = torch.empty(n, dtype=torch.int64, device=self.device)
            objective = 0.0
            c_norms = B.sq_norms(centroids)
            b0 = 0
            cur_bs = bs
            if grow is not None and e == 0:
                cur_bs = max(1, bs // int(grow.get("start_div", 8)))
            while b0 < r:
                idx = order[b0 : min(b0 + cur_bs, r)]
                xb = x[idx]
                if assigner is None and self.hartigan_ema and stage == "fine" and prev_labels is not None and bool((prev_labels[idx] >= 0).all()):
                    # estimated cluster sizes from the EMA masses (mass ~ points per batch)
                    n_est = usage * (n / bs)
                    labels, d2 = self._hartigan_assign(xb, centroids, n_est, prev_labels[idx])
                elif assigner is None:
                    labels, d2 = B.assign_exact(xb, centroids, c_norms)
                elif isinstance(assigner, B.PCAFilter):
                    labels, d2 = assigner.assign(idx, xb, centroids)
                else:
                    if isinstance(assigner, _GraphAssigner) and assigner.gamma > 0:
                        # per-cluster mean squared distance from the EMA statistics (global mean where unknown)
                        known = usage > 1e-3
                        r2 = sse / usage.clamp_min(1e-12)
                        gmean = (sse[known].sum() / usage[known].sum()) if bool(known.any()) else torch.tensor(float("inf"), device=self.device)
                        assigner.r2 = torch.where(known, r2, gmean)
                    labels, d2 = assigner.assign(idx, xb, centroids, c_norms)
                objective += float(d2.sum().item())
                if prev_labels is not None:
                    prev_labels[idx] = labels
                counts, sums = B.batch_stats(xb, labels, k)
                bsse = torch.zeros(k, dtype=torch.float32, device=self.device).index_add_(0, labels, d2)
                pass_counts += counts
                if last:
                    pass_sums += sums
                    last_labels[idx] = labels
                # step size: cosine over rows processed, peak from the step law at the current batch size
                frac = min(1.0, schedule_offset + (1.0 - schedule_offset) * (rows_done / total_rows))
                a_peak = a0 if cur_bs == bs else step_law(n, k, max(nepochs, 1e-9), cur_bs) * (a0 / step_law(n, k, max(nepochs, 1e-9), bs))
                alpha = scheduled_alpha(a_peak, frac, 1.0)
                touched = counts > 0
                old = centroids.clone() if isinstance(assigner, _CachedAssigner) else None
                usage *= 1.0 - alpha
                sse *= 1.0 - alpha
                mass = usage[touched] + alpha * counts[touched]
                mean = sums[touched] / counts[touched, None]
                w = (alpha * counts[touched] / mass)[:, None]
                centroids[touched] = centroids[touched] + w * (mean - centroids[touched])
                usage[touched] = mass
                sse[touched] += alpha * bsse[touched]
                if isinstance(assigner, _CachedAssigner):
                    assigner.record_move(old, centroids)
                elif isinstance(assigner, _GraphAssigner):
                    assigner.after_update(centroids)
                step += 1
                rows_done += idx.numel()
                b0 += idx.numel()
                if grow is not None and e == 0 and cur_bs < bs and step % int(grow.get("every", 2)) == 0:
                    cur_bs = min(bs, cur_bs * 2)
                if tau > 0 and step % CHECK_UNUSED_EVERY == 0:
                    self._revive(usage, sse, centroids, split_rng, tau, assigner)
                c_norms = B.sq_norms(centroids)
            self.history_.append(dict(stage=stage, epoch=sum(rows[: e + 1]) / n, moving_obj=objective,
                                      time=time.perf_counter() - t0, splits=self.n_splits_, ops=B.OPS.pairs,
                                      steps=step))
            if callback is not None and stage == "fine":
                callback(e, centroids.clone() if (not last or self.tail_epochs > 0) else None)
        return centroids, pass_counts, pass_sums, last_labels, sse

    # ------------------------------------------------------------ warm start
    def _warm_start(self, x, k, bs, a0, tau, callback):
        """Hierarchical seeding. ``factor`` may be an int (one coarse level, k/factor centroids) or a
        list of factors for several levels (coarsest first), each level seeded from the previous one.
        Children are allocated to coarse clusters proportionally to their SSE (``alloc="sse"``),
        their point count (``"count"``) or uniformly (``"uniform"``)."""
        ws = self.warm_start
        factors = ws.get("factor", 16)
        if isinstance(factors, (int, float)):
            factors = [int(factors)]
        f = int(factors[-1])
        kc = max(1, k // f)
        ec = float(ws.get("epochs", 0.5))
        n, d = x.shape
        law = resolve_batch_law(self.batch_law, d)
        bsc, a0c = resolve_recipe(n, kc, max(ec, 0.25), law, None, None)
        t = time.perf_counter()
        if len(factors) > 1:
            sub = HMeans(kc, nepochs=ec, batch_law=self.batch_law, seed=self.seed + 3, device=self.device,
                         warm_start=dict(ws, factor=factors[:-1]), donor=self.donor, assign="exact")
            sub.fit(x)
            coarse = sub.centroids_
            self.n_splits_ += sub.n_splits_
            labels_c = None
        else:
            coarse = B.random_rows_init(x, kc, self.seed)
            usage_c = torch.full((kc,), INIT_USAGE, dtype=torch.float32, device=self.device)
            coarse, cnt_c, sums_c, labels_c, _ = self._ema_passes(
                x, coarse, usage_c, ec, bsc, a0c, tau, None, self.seed + 7, stage="coarse")
        # coarse labels only cover the rows of the (possibly fractional) last pass; assign the rest exactly
        d2_c = None
        if labels_c is None or ec < 1 or ws.get("alloc", "sse") == "sse":
            labels_c, d2_c = B.assign_exact(x, coarse, tag="warm_assign")
        # number of children per coarse cluster
        alloc = ws.get("alloc", "sse")
        cnt = torch.bincount(labels_c, minlength=kc).float()
        if alloc == "sse":
            weight = torch.zeros(kc, device=self.device).index_add_(0, labels_c, d2_c)
        elif alloc == "count":
            weight = cnt
        else:
            weight = torch.ones(kc, device=self.device)
        weight = torch.where(cnt > 0, weight, torch.zeros_like(weight))
        nchild = self._allocate(weight, cnt, k)
        f = int(nchild.max().item())
        # children: f random members of each coarse cluster (fallback: perturbed coarse centroid)
        rng = np.random.RandomState(self.seed + 11)
        lab_np = labels_c.cpu().numpy()
        # f random members per coarse cluster: sort rows by (label, random key) and take each segment's head
        keys = rng.random_sample(n)
        order = np.lexsort((keys, lab_np))
        sorted_lab = lab_np[order]
        bounds = np.searchsorted(sorted_lab, np.arange(kc + 1))
        sizes = np.diff(bounds)
        nchild_np = nchild.cpu().numpy()
        child_rows = np.full((kc, f), -1, dtype=np.int64)
        rank = np.arange(f)[None, :]
        take = (rank < sizes[:, None]) & (rank < nchild_np[:, None])
        pos = bounds[:-1][:, None] + rank
        child_rows[take] = order[pos[take]]
        child_rows_t = torch.from_numpy(child_rows).to(self.device)
        children = coarse[:, None, :].expand(kc, f, d).clone()
        valid = child_rows_t >= 0
        children[valid] = x[child_rows_t[valid]]
        # slots: which (coarse, rank) pairs are real children; others are padding that no point can pick
        slot = torch.from_numpy(rank < nchild_np[:, None]).to(self.device)  # [kc, f]
        # allocated but memberless slots: perturb the coarse centroid along alternating coordinates
        miss_mask = slot & ~valid
        if bool(miss_mask.any()):
            sign = torch.ones(d, device=self.device)
            sign[1::2] = -1.0
            miss = torch.nonzero(miss_mask)
            children[miss[:, 0], miss[:, 1]] = coarse[miss[:, 0]] * (1 + SPLIT_EPSILON * (1 + miss[:, 1, None].float()) * sign)
        # local Lloyd: each point sees only the children of its coarse cluster
        child_counts = None
        for _ in range(int(ws.get("local_iters", 2))):
            lab_local = torch.empty(n, dtype=torch.int64, device=self.device)
            rows = max(1024, (1 << 27) // (f * d))
            for b in range(0, n, rows):
                lc = labels_c[b : b + rows]
                ch = children[lc]  # [rows, f, d]
                diff = ch - x[b : b + rows, None, :]
                dist = (diff * diff).sum(dim=2)
                dist[~slot[lc]] = float("inf")
                lab_local[b : b + rows] = dist.argmin(dim=1)
            B.OPS.add(n * f, "warm_local", d)
            flat = labels_c * f + lab_local
            child_counts = torch.bincount(flat, minlength=kc * f).to(torch.float32)
            child_sums = torch.zeros((kc * f, d), dtype=torch.float32, device=self.device).index_add_(0, flat, x)
            nonempty = child_counts > 0
            flat_children = children.reshape(kc * f, d)
            flat_children[nonempty] = child_sums[nonempty] / child_counts[nonempty, None]
            children = flat_children.reshape(kc, f, d)
        # compact the allocated slots into k centroids
        slot_ids = torch.nonzero(slot.reshape(-1)).flatten()  # exactly k entries
        centroids = children.reshape(kc * f, d)[slot_ids].contiguous()
        counts = child_counts[slot_ids] if child_counts is not None else torch.zeros(k, device=self.device)
        remap = torch.full((kc * f,), -1, dtype=torch.int64, device=self.device)
        remap[slot_ids] = torch.arange(k, device=self.device)
        init_labels = remap[labels_c * f + lab_local] if child_counts is not None else None
        self._children_of = None
        # initial EMA mass: rho * (points this cluster would see per batch)
        rho = float(ws.get("init_mass", 1.0))
        usage = (rho * counts * (bs / n)).clamp_min(INIT_USAGE)
        self.extra_["warm_time"] = time.perf_counter() - t
        self.extra_["warm_kc"] = kc
        self.extra_["warm_ops"] = B.OPS.pairs
        if self.verbose:
            print(f"warm start: kc={kc} f={f} epochs={ec} time={self.extra_['warm_time']:.2f}s ops={B.OPS.pairs / (n * k):.3f} epoch-equiv")
        return centroids, usage, init_labels

    @staticmethod
    def _allocate(weight, cnt, k):
        """Largest-remainder apportionment of k children to coarse clusters, proportional to
        ``weight``; at least one child per non-empty cluster and at most ``cnt`` (its points)."""
        device = weight.device
        kc = weight.shape[0]
        w = weight.detach().cpu().double()
        cnt = cnt.detach().cpu().double()
        alive = cnt > 0
        cap = torch.where(alive, cnt.clamp_min(1.0), torch.zeros_like(cnt))
        if w.sum() <= 0:
            w = alive.double()
        alloc = torch.zeros(kc, dtype=torch.long)
        alloc[alive] = 1
        remaining = k - int(alloc.sum().item())
        if remaining < 0:  # more non-empty clusters than k (cannot happen when kc <= k)
            alloc[:] = 0
            alloc[torch.topk(w, k).indices] = 1
            return alloc.to(device)
        for _ in range(8):  # iterate so capped clusters hand their share to the others
            room = (cap - alloc.double()).clamp_min(0)
            share = w * (room > 0)
            if share.sum() <= 0 or remaining <= 0:
                break
            quota = share / share.sum() * remaining
            give = torch.minimum(torch.floor(quota), room).long()
            alloc += give
            remaining -= int(give.sum().item())
            if remaining <= 0:
                break
            frac = quota - torch.floor(quota)
            frac = torch.where((cap - alloc.double()) > 0, frac, torch.full_like(frac, -1.0))
            take = torch.topk(frac, min(remaining, int((frac >= 0).sum().item()))).indices
            alloc[take] += 1
            remaining -= take.numel()
        return alloc.to(device)

    # --------------------------------------------------------------------- fit
    def fit(self, x, init_centroids=None, callback=None):
        x = B.to_device(x, self.device)
        n, d = x.shape
        k = self.k
        if self.ema_frac == "auto":
            self._set_split(auto_ema_frac(n, k))
        law = resolve_batch_law(self.batch_law, d)
        T_ema = self.ema_epochs
        bs, a0 = resolve_recipe(n, k, T_ema, law, self.batch_size, self.alpha0)
        tau = (0.0 if d <= NO_REVIVAL_MAX_D else UNUSED_THRESHOLD) if self.unused_threshold is None else self.unused_threshold
        rows = epoch_rows(n, T_ema)
        self.recipe_ = dict(batch_size=bs, alpha0=a0, batch_law=law, epoch_rows=rows, tau=tau, ema_frac=self.ema_frac_,
                            ema_epochs=T_ema, tail_epochs=self.tail_epochs, tail=self.tail)
        t_start = time.perf_counter()
        self.history_ = []
        init_labels = None
        if init_centroids is not None:
            centroids = B.to_device(init_centroids, self.device).clone()
            usage = torch.full((k,), INIT_USAGE, dtype=torch.float32, device=self.device)
        elif self.warm_start:
            centroids, usage, init_labels = self._warm_start(x, k, bs, a0, tau, callback)
            if not self.warm_start.get("use_labels", False):
                init_labels = None
        else:
            centroids = B.random_rows_init(x, k, self.seed)
            usage = torch.full((k,), INIT_USAGE, dtype=torch.float32, device=self.device)
        assigner = None
        if self.assign in ("bounded", "cached"):
            max_steps = sum((r + bs - 1) // bs for r in rows) * 2 + 64
            assigner = _CachedAssigner(n, k, self.m, self.hot, max_steps, self.device, mode=self.assign,
                                       refresh=self.refresh, seed=self.seed)
        elif self.assign == "pca":
            t_p = time.perf_counter()
            assigner = B.PCAFilter(x, self.pca_dim, self.pca_cand, self.pca_var, seed=self.seed)
            B.synchronize(self.device)
            self.extra_["pca_dim"] = assigner.dim
            self.extra_["pca_setup_time"] = time.perf_counter() - t_p
        elif self.assign == "graph":
            assigner = _GraphAssigner(n, k, self.m, self.device, rebuild_every=self.graph_rebuild_every,
                                      seed=self.seed, hops=self.graph_hops, gamma=self.graph_gamma)
            if init_labels is not None:
                assigner.prev[:] = init_labels
            elif self.warm_start and self.warm_start.get("ivf_first", True) and getattr(self, "_children_of", None) is not None:
                assigner.children_of = self._children_of
                assigner.nprobe = int(self.warm_start.get("nprobe", 4))
        offset = float(self.warm_start.get("schedule_offset", 0.0)) if self.warm_start else 0.0
        centroids, pass_counts, pass_sums, labels, sse = self._ema_passes(
            x, centroids, usage, T_ema, bs, a0, tau, assigner, self.seed, callback=callback,
            grow=self.batch_grow, schedule_offset=offset)
        if self.finalize == "last_epoch_mean" and pass_sums is not None:
            nonempty = pass_counts > 0
            final = centroids.clone()
            final[nonempty] = pass_sums[nonempty] / pass_counts[nonempty, None]
        else:
            final = centroids.clone()
        # tail: incremental exact-mean passes (block coordinate descent), optionally Hartigan-corrected
        if self.tail and self.tail_epochs > 0 and pass_sums is not None and labels is not None:
            final, labels = self._tail_passes(x, final, pass_sums.clone(), pass_counts.clone(), labels, bs, t_start,
                                              callback=callback)
        # polishing: exact Lloyd iterations, cheap through the bounded assigner
        if self.polish > 0:
            if isinstance(assigner, _CachedAssigner):
                assigner.record_move(centroids, final)
            all_idx = torch.arange(n, device=self.device)
            for p in range(self.polish):
                c_norms = B.sq_norms(final)
                if isinstance(assigner, _CachedAssigner) and self.polish_mode == "bounded":
                    labels, d2 = assigner.assign(all_idx, x, final, c_norms, exact=True)
                elif isinstance(assigner, B.PCAFilter):
                    labels, d2 = assigner.assign(all_idx, x, final, tag="polish")
                elif isinstance(assigner, _GraphAssigner) and self.polish_mode == "graph":
                    assigner.rebuild(final)
                    labels, d2 = assigner.assign(all_idx, x, final, c_norms)
                else:
                    labels, d2 = B.assign_exact(x, final, c_norms, tag="polish")
                counts, sums = B.batch_stats(x, labels, k)
                nonempty = counts > 0
                new = final.clone()
                new[nonempty] = sums[nonempty] / counts[nonempty, None]
                if isinstance(assigner, _CachedAssigner):
                    assigner.record_move(final, new)
                final = new
                self.history_.append(dict(stage="polish", epoch=self.nepochs + p + 1, moving_obj=float(d2.sum().item()),
                                          time=time.perf_counter() - t_start, ops=B.OPS.pairs))
        self.centroids_ = final
        self.labels_ = labels
        self.fit_time_ = time.perf_counter() - t_start
        if isinstance(assigner, _CachedAssigner):
            self.extra_["assign_full"] = assigner.n_full
            self.extra_["assign_cheap"] = assigner.n_cheap
            self.extra_["assign_certified"] = assigner.n_certified
        elif isinstance(assigner, _GraphAssigner):
            self.extra_.update(assigner.stats())
        elif isinstance(assigner, B.PCAFilter):
            self.extra_["pca_fallback_frac"] = assigner.n_fallback / max(1, assigner.n_points)
        return self

    def _tail_passes(self, x, centroids, S, N, labels, bs, t_start, callback=None):
        """Incremental k-means passes. ``S``/``N`` are exact sums/counts of all points under
        ``labels`` (the last EMA pass's labels). Each batch is re-assigned (exactly, or with
        Hartigan's n/(n+-1) correction) and the sums of the affected clusters are updated, so
        every update is an exact partial Lloyd step and the objective is monotone."""
        n, d = x.shape
        k = centroids.shape[0]
        bs = int(self.tail_batch or bs)
        hartigan = self.tail == "hartigan"
        # centroids consistent with (S, N): the last-epoch-mean finalisation already is, for non-empty clusters
        nonempty = N > 0
        c = centroids.clone()
        c[nonempty] = S[nonempty] / N[nonempty, None]
        rows = epoch_rows(n, self.tail_epochs)
        seed = self.seed + 101
        split_rng = np.random.RandomState(seed)
        step = 0
        n_changed_total = 0
        for e, r in enumerate(rows):
            order = torch.from_numpy(np.random.RandomState(epoch_seed(seed, e)).permutation(n)).to(self.device)
            objective = 0.0
            changed_epoch = 0
            for b0 in range(0, r, bs):
                idx = order[b0 : min(b0 + bs, r)]
                xb = x[idx]
                old = labels[idx]
                if hartigan:
                    new, d2 = self._hartigan_assign(xb, c, N, old)
                else:
                    new, d2 = B.assign_exact(xb, c, B.sq_norms(c))
                objective += float(d2.sum().item())
                moved = new != old
                nm = int(moved.sum().item())
                if nm:
                    xm = xb[moved]
                    S.index_add_(0, old[moved], -xm)
                    S.index_add_(0, new[moved], xm)
                    N.index_add_(0, old[moved], torch.full((nm,), -1.0, device=self.device))
                    N.index_add_(0, new[moved], torch.ones(nm, device=self.device))
                    labels[idx[moved]] = new[moved]
                    touched = torch.unique(torch.cat([old[moved], new[moved]]))
                    ne = N[touched] > 0
                    c[touched[ne]] = S[touched[ne]] / N[touched[ne], None]
                    changed_epoch += nm
                step += 1
                # empty clusters: split the largest cluster's points? keep simple: re-seed at a far point of a donor
                empty = torch.nonzero(N <= 0).flatten()
                if empty.numel() and step % CHECK_UNUSED_EVERY == 0:
                    self._tail_revive(x, c, S, N, labels, empty, split_rng)
            n_changed_total += changed_epoch
            self.history_.append(dict(stage="tail", epoch=self.ema_epochs + sum(rows[: e + 1]) / n, moving_obj=objective,
                                      time=time.perf_counter() - t_start, ops=B.OPS.pairs, changed=changed_epoch / n))
            if callback is not None and e < len(rows) - 1:
                nonempty = N > 0
                snap = c.clone()
                snap[nonempty] = S[nonempty] / N[nonempty, None]
                callback(len(epoch_rows(n, self.ema_epochs)) + e, snap)
        self.extra_["tail_changed_frac"] = n_changed_total / max(1, n * self.tail_epochs)
        nonempty = N > 0
        c[nonempty] = S[nonempty] / N[nonempty, None]
        return c, labels

    def _hartigan_assign(self, xb, c, N, old):
        """Hartigan's rule: move x from a to b iff N_a/(N_a-1) d2(x,a) > N_b/(N_b+1) d2(x,b).

        The join costs g_j d2(x, c_j) with g_j = N_j/(N_j+1) are one matmul of augmented
        matrices: [||x||^2, x, 1] . [g_j, -2 g_j c_j, g_j ||c_j||^2]. The stay cost of the current
        cluster is written into its column (one scatter per row); empty clusters are excluded.
        """
        nb, k = xb.shape[0], c.shape[0]
        d = c.shape[1]
        g = torch.where(N > 0, N / (N + 1.0), torch.full_like(N, float("inf")))
        g_fin = torch.where(torch.isfinite(g), g, torch.zeros_like(g))
        c_norms = B.sq_norms(c)
        W = torch.cat([g_fin[:, None], -2.0 * g_fin[:, None] * c, (g_fin * c_norms)[:, None]], dim=1)  # [k, d+2]
        empty_cols = ~torch.isfinite(g)
        new = torch.empty(nb, dtype=torch.int64, device=self.device)
        d2 = torch.empty(nb, dtype=torch.float32, device=self.device)
        rows = max(256, (1 << 28) // max(k, 1))
        for b in range(0, nb, rows):
            xr = xb[b : b + rows]
            o = old[b : b + rows]
            xn = B.sq_norms(xr)
            xa = torch.cat([xn[:, None], xr, torch.ones((xr.shape[0], 1), device=self.device)], dim=1)
            cost = xa @ W.T  # [rows, k]
            if bool(empty_cols.any()):
                cost[:, empty_cols] = float("inf")
            d_old = ((xr - c[o]) ** 2).sum(dim=1)
            n_old = N[o]
            stay = torch.where(n_old > 1.0, d_old * n_old / (n_old - 1.0).clamp_min(1e-6), torch.zeros_like(d_old))
            cost.scatter_(1, o[:, None], stay[:, None])
            j = cost.argmin(dim=1)
            new[b : b + rows] = j
            d2[b : b + rows] = ((xr - c[j]) ** 2).sum(dim=1)
        B.OPS.add(nb * k, "assign", d + 2)
        return new, d2

    def _tail_revive(self, x, c, S, N, labels, empty, rng):
        """Re-seed empty clusters at the farthest point of the largest clusters (exact bookkeeping)."""
        probs = (N / N.sum()).cpu().numpy().astype(np.float64)
        probs = np.where(N.cpu().numpy() > 1, probs, 0.0)
        if probs.sum() <= 0:
            return
        probs /= probs.sum()
        n_rev = min(empty.numel(), int((probs > 0).sum()))
        donors = rng.choice(N.shape[0], size=n_rev, replace=False, p=probs)
        empty = empty[:n_rev]
        for e_idx, dnr in zip(empty.tolist(), donors.tolist()):
            members = torch.nonzero(labels == dnr).flatten()
            if members.numel() < 2:
                continue
            dd = ((x[members] - c[dnr]) ** 2).sum(dim=1)
            far = members[dd.argmax()]
            xv = x[far]
            S[dnr] -= xv
            N[dnr] -= 1.0
            S[e_idx] = xv
            N[e_idx] = 1.0
            labels[far] = e_idx
            c[dnr] = S[dnr] / N[dnr]
            c[e_idx] = xv
            self.n_splits_ += 1

    def predict(self, x):
        labels, _ = B.assign_exact(B.to_device(x, self.device), self.centroids_, count=False)
        return labels
