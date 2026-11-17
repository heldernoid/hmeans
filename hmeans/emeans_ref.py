"""Reimplementation of e-means (Bessa et al., 2026, arXiv:2611.09501) in PyTorch.

Follows the shipped faiss ``contrib/e_means`` recipe (base.py / e_means.py of
faiss-cpu 1.16.0) rather than the paper's rounded constants: shipped batch and
step laws, 2**-16 initial mass, 1e-4 split perturbation, revival every 5
updates below 0.1 x mean mass with mass-proportional donors, cosine schedule to
0.1*alpha0, last-epoch-mean finalization, remainder batches kept. The point of
the reimplementation is to run e-means on the same assignment backend
(CPU/MPS) and with the same operation counting as the other methods; the
official NumpyEmeans is used as a cross-check (see experiments/).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from . import backend as B

from .laws import (  # noqa: F401  (re-exported: the torch core imports them from here)
    ALPHA_LAW,
    ALPHA_MIN_FACTOR,
    AUTO_QUALITY_MAX_D,
    BATCH_LAWS,
    CHECK_UNUSED_EVERY,
    INIT_USAGE,
    NO_REVIVAL_MAX_D,
    SPLIT_EPSILON,
    UNUSED_THRESHOLD,
    epoch_rows,
    epoch_seed,
    resolve_batch_law,
    resolve_recipe,
    round_power_of_two,
    scheduled_alpha,
    step_law,
)


class EMeansRef:
    """e-means with a scikit-learn-like fit API on a torch device.

    Parameters mirror ``faiss.contrib.e_means.NumpyEmeans``. ``callback(epoch_idx,
    centroids)`` is invoked after every epoch (used for loss curves).
    """

    def __init__(
        self,
        k: int,
        nepochs: float = 5,
        batch_size: int | None = None,
        alpha0: float | None = None,
        batch_law: str = "auto",
        unused_threshold: float | None = None,
        seed: int = 1234,
        finalize: str = "last_epoch_mean",
        device: str | torch.device = "auto",
        verbose: bool = False,
    ):
        self.k = int(k)
        self.nepochs = float(nepochs)
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.batch_law = batch_law
        self.unused_threshold = unused_threshold
        self.seed = int(seed)
        self.finalize = finalize
        self.device = B.pick_device(device) if isinstance(device, str) else device
        self.verbose = verbose
        self.centroids_ = None
        self.history_ = []
        self.n_splits_ = 0

    # ----------------------------------------------------------------- revival
    def _revive(self, usage: torch.Tensor, centroids: torch.Tensor, rng: np.random.RandomState, tau: float) -> int:
        u = usage.cpu().numpy().astype(np.float64)
        threshold = tau * u.sum() / self.k
        unused = u < threshold
        donors = (~unused) & (u > 0)
        nsplit = min(int(unused.sum()), int(donors.sum()))
        if nsplit == 0:
            return 0
        uniform = np.maximum(rng.random_sample(self.k), np.finfo("float64").tiny)
        keys = np.full(self.k, -np.inf)
        keys[donors] = np.log(u[donors]) - np.log(-np.log(uniform[donors]))
        source = np.argsort(keys)[-nsplit:][::-1].copy()
        target = np.flatnonzero(unused)[:nsplit]
        src = torch.from_numpy(source).to(self.device)
        tgt = torch.from_numpy(target).to(self.device)
        c = centroids[src]
        sign = torch.ones(c.shape[1], device=self.device)
        sign[1::2] = -1.0
        offset = SPLIT_EPSILON * sign
        donor_c = c * (1.0 - offset)
        child_c = c * (1.0 + offset)
        # separate identical rows (zero donors) along the first coordinate
        same = (donor_c == child_c).all(dim=1)
        if bool(same.any()):
            child_c[same, 0] = torch.nextafter(child_c[same, 0], torch.full_like(child_c[same, 0], float("inf")))
        half = usage[src] / 2.0
        usage[src] = half
        usage[tgt] = half
        centroids[src] = donor_c
        centroids[tgt] = child_c
        self.n_splits_ += nsplit
        return nsplit

    # --------------------------------------------------------------------- fit
    def fit(self, x: torch.Tensor, init_centroids: torch.Tensor | None = None, callback=None):
        x = B.to_device(x, self.device)
        n, d = x.shape
        k = self.k
        law = resolve_batch_law(self.batch_law, d)
        bs, a0 = resolve_recipe(n, k, self.nepochs, law, self.batch_size, self.alpha0)
        tau = (0.0 if d <= NO_REVIVAL_MAX_D else UNUSED_THRESHOLD) if self.unused_threshold is None else self.unused_threshold
        rows = epoch_rows(n, self.nepochs)
        steps = [(r + bs - 1) // bs for r in rows]
        total_steps = sum(steps)
        self.recipe_ = dict(batch_size=bs, alpha0=a0, batch_law=law, total_steps=total_steps, epoch_rows=rows, tau=tau)
        if self.verbose:
            print(f"e-means(ref): n={n} d={d} k={k} T={self.nepochs} B={bs} alpha0={a0:.4g} law={law} steps={total_steps}")

        if init_centroids is None:
            centroids = B.random_rows_init(x, k, self.seed)
        else:
            centroids = B.to_device(init_centroids, self.device).clone()
        usage = torch.full((k,), INIT_USAGE, dtype=torch.float32, device=self.device)
        split_rng = np.random.RandomState(self.seed)
        step = 0
        t0 = time.perf_counter()
        self.history_ = []
        last_labels = None
        for e, r in enumerate(rows):
            last = e == len(rows) - 1
            order = torch.from_numpy(np.random.RandomState(epoch_seed(self.seed, e)).permutation(n)).to(self.device)
            pass_counts = torch.zeros(k, dtype=torch.float32, device=self.device)
            pass_sums = torch.zeros((k, d), dtype=torch.float32, device=self.device) if last else None
            if last:
                last_labels = torch.empty(n, dtype=torch.int64, device=self.device)
            objective = 0.0
            c_norms = B.sq_norms(centroids)
            for b0 in range(0, r, bs):
                idx = order[b0 : min(b0 + bs, r)]
                xb = x[idx]
                labels, d2 = B.assign_exact(xb, centroids, c_norms)
                objective += float(d2.sum().item())
                counts, sums = B.batch_stats(xb, labels, k)
                pass_counts += counts
                if last:
                    pass_sums += sums
                    last_labels[idx] = labels
                alpha = scheduled_alpha(a0, step, total_steps)
                # EMA update in (usage, centroid) form: N <- (1-a) N + a n ; c <- c + (a n / N_new)(mean - c)
                touched = counts > 0
                usage *= 1.0 - alpha
                mass = usage[touched] + alpha * counts[touched]
                mean = sums[touched] / counts[touched, None]
                w = (alpha * counts[touched] / mass)[:, None]
                centroids[touched] = centroids[touched] + w * (mean - centroids[touched])
                usage[touched] = mass
                step += 1
                if tau > 0 and step % CHECK_UNUSED_EVERY == 0:
                    self._revive(usage, centroids, split_rng, tau)
                c_norms = B.sq_norms(centroids)
            self.history_.append(dict(epoch=sum(rows[: e + 1]) / n, moving_obj=objective, time=time.perf_counter() - t0, splits=self.n_splits_))
            if callback is not None:
                callback(e, centroids.clone() if not last else None)
        if self.finalize == "last_epoch_mean":
            nonempty = pass_counts > 0
            final = centroids.clone()
            final[nonempty] = pass_sums[nonempty] / pass_counts[nonempty, None]
        else:
            final = centroids.clone()
        self.centroids_ = final
        self.labels_ = last_labels
        self.fit_time_ = time.perf_counter() - t0
        return self

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        labels, _ = B.assign_exact(B.to_device(x, self.device), self.centroids_, count=False)
        return labels
