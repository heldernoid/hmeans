"""Baselines on the shared torch backend: Lloyd (random / k-means++ init) and
Sculley's mini-batch k-means. Both count distance computations through
``backend.OPS``."""
from __future__ import annotations

import time

import numpy as np
import torch

from . import backend as B


class Lloyd:
    """Full-batch Lloyd iterations. ``init`` is "random", "kmeans++" (greedy, as in
    scikit-learn), or a tensor of initial centroids. Empty clusters are refilled by
    splitting the largest clusters (faiss's policy) when ``empty_policy="split"``."""

    def __init__(self, k, n_iter=25, init="random", seed=1234, device="auto", init_sample=None,
                 empty_policy="split", tol=0.0, verbose=False, init_trials=None):
        self.k, self.n_iter, self.init, self.seed = int(k), int(n_iter), init, int(seed)
        self.device = B.pick_device(device) if isinstance(device, str) else device
        self.init_sample, self.empty_policy, self.tol, self.verbose = init_sample, empty_policy, tol, verbose
        self.init_trials = init_trials
        self.history_ = []

    def fit(self, x, callback=None):
        x = B.to_device(x, self.device)
        n, d = x.shape
        k = self.k
        t0 = time.perf_counter()
        if isinstance(self.init, str):
            if self.init == "random":
                c = B.random_rows_init(x, k, self.seed)
            elif self.init == "kmeans++":
                c = B.kmeanspp_init(x, k, self.seed, greedy=True, sample=self.init_sample, n_local_trials=self.init_trials)
            elif self.init == "kmeans++-plain":
                c = B.kmeanspp_init(x, k, self.seed, greedy=False, sample=self.init_sample)
            else:
                raise ValueError(self.init)
        else:
            c = B.to_device(self.init, self.device).clone()
        B.synchronize(self.device)
        self.init_time_ = time.perf_counter() - t0
        rng = np.random.RandomState(self.seed)
        prev_obj = None
        labels = None
        for it in range(self.n_iter):
            labels, d2 = B.assign_exact(x, c)
            obj = float(d2.sum().item())
            counts, sums = B.batch_stats(x, labels, k)
            nonempty = counts > 0
            c = torch.where(nonempty[:, None], sums / counts.clamp_min(1.0)[:, None], c)
            n_empty = int((~nonempty).sum().item())
            if n_empty and self.empty_policy == "split":
                # faiss policy: split the largest clusters, perturb along alternating coordinates
                cnt = counts.cpu().numpy()
                empty_idx = np.flatnonzero(cnt == 0)
                probs = cnt / cnt.sum()
                donors = rng.choice(k, size=len(empty_idx), replace=True, p=probs)
                src = torch.from_numpy(donors).to(self.device)
                tgt = torch.from_numpy(empty_idx).to(self.device)
                sign = torch.ones(d, device=self.device)
                sign[1::2] = -1.0
                eps = 1.0 / 1024.0
                base = c[src]
                c[tgt] = base * (1 + eps * sign)
                c[src] = base * (1 - eps * sign)
            self.history_.append(dict(epoch=it + 1, moving_obj=obj, time=time.perf_counter() - t0, n_empty=n_empty))
            if callback is not None:
                callback(it, c.clone())
            if self.verbose:
                print(f"lloyd it {it+1}: obj {obj / n:.6g} empty {n_empty}")
            if prev_obj is not None and self.tol > 0 and abs(prev_obj - obj) <= self.tol * obj:
                break
            prev_obj = obj
        self.centroids_ = c
        self.labels_ = labels
        self.fit_time_ = time.perf_counter() - t0
        return self

    def predict(self, x):
        labels, _ = B.assign_exact(B.to_device(x, self.device), self.centroids_, count=False)
        return labels


class SculleyMiniBatch:
    """Sculley (2010) web-scale mini-batch k-means with per-center learning rate
    1/n_c (cumulative counts) and optional random reassignment of small clusters
    (as in scikit-learn's ``reassignment_ratio``)."""

    def __init__(self, k, nepochs=5, batch_size=4096, reassignment_ratio=0.01, seed=1234, device="auto", verbose=False):
        self.k, self.nepochs, self.batch_size = int(k), float(nepochs), int(batch_size)
        self.reassignment_ratio, self.seed, self.verbose = reassignment_ratio, int(seed), verbose
        self.device = B.pick_device(device) if isinstance(device, str) else device
        self.history_ = []

    def fit(self, x, callback=None):
        x = B.to_device(x, self.device)
        n, d = x.shape
        k = self.k
        bs = min(self.batch_size, n)
        c = B.random_rows_init(x, k, self.seed)
        counts_total = torch.zeros(k, dtype=torch.float32, device=self.device)
        rng = np.random.RandomState(self.seed)
        t0 = time.perf_counter()
        total_rows = int(round(self.nepochs * n))
        done = 0
        epoch = 0
        c_norms = B.sq_norms(c)
        while done < total_rows:
            order = torch.from_numpy(rng.permutation(n)).to(self.device)
            obj = 0.0
            labels_epoch = torch.empty(n, dtype=torch.int64, device=self.device)
            for b0 in range(0, n, bs):
                if done >= total_rows:
                    break
                idx = order[b0 : b0 + bs]
                xb = x[idx]
                labels, d2 = B.assign_exact(xb, c, c_norms)
                labels_epoch[idx] = labels
                obj += float(d2.sum().item())
                counts, sums = B.batch_stats(xb, labels, k)
                counts_total += counts
                touched = counts > 0
                lr = (counts[touched] / counts_total[touched])[:, None]
                mean = sums[touched] / counts[touched, None]
                c[touched] = c[touched] + lr * (mean - c[touched])
                # reassignment of tiny clusters (scikit-learn style)
                if self.reassignment_ratio > 0:
                    thresh = self.reassignment_ratio * counts_total.max()
                    small = counts_total < thresh
                    nsmall = int(small.sum().item())
                    if nsmall:
                        nsmall = min(nsmall, xb.shape[0])
                        small_idx = torch.nonzero(small).flatten()[:nsmall]
                        pick = torch.from_numpy(rng.choice(xb.shape[0], nsmall, replace=False)).to(self.device)
                        c[small_idx] = xb[pick]
                        counts_total[small_idx] = counts_total.min().clamp_min(1.0)
                c_norms = B.sq_norms(c)
                done += len(idx)
            epoch += 1
            self.history_.append(dict(epoch=min(done, total_rows) / n, moving_obj=obj, time=time.perf_counter() - t0))
            if callback is not None:
                callback(epoch - 1, c.clone())
        self.centroids_ = c
        self.labels_ = labels_epoch
        self.fit_time_ = time.perf_counter() - t0
        return self

    def predict(self, x):
        labels, _ = B.assign_exact(B.to_device(x, self.device), self.centroids_, count=False)
        return labels
