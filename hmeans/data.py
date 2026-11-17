"""Datasets: ANN-benchmarks HDF5 embeddings and synthetic mixtures.

Every loader returns (train, heldout) float32 NumPy arrays. The held-out split
is the last ``n_heldout`` rows of a seeded permutation of the training rows,
so it is identically distributed and disjoint from training (the paper uses a
20k deduplicated held-out set).
"""
from __future__ import annotations

import os

import numpy as np

DATA_DIR = os.environ.get("HMEANS_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))

HDF5 = {
    "sift": ("sift-128-euclidean.hdf5", False),
    "fashion-mnist": ("fashion-mnist-784-euclidean.hdf5", False),
    "glove": ("glove-100-angular.hdf5", True),
    "nytimes": ("nytimes-256-angular.hdf5", True),
    "deep": ("deep-image-96-angular.hdf5", True),
    "gist": ("gist-960-euclidean.hdf5", False),
}


def _split(x: np.ndarray, n_heldout: int, seed: int, n_train: int | None) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(x.shape[0])
    held = x[perm[:n_heldout]]
    rest = perm[n_heldout:]
    if n_train is not None:
        rest = rest[:n_train]
    return np.ascontiguousarray(x[rest], dtype=np.float32), np.ascontiguousarray(held, dtype=np.float32)


def load_hdf5(name: str, n_heldout: int = 20000, seed: int = 0, n_train: int | None = None):
    import h5py

    fname, normalize = HDF5[name]
    path = os.path.join(DATA_DIR, fname)
    with h5py.File(path, "r") as f:
        x = np.asarray(f["train"], dtype=np.float32)
    if normalize:
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    # deduplicate exact duplicates (glove / nytimes have a few) so held-out rows are not in train
    x = np.unique(x, axis=0) if name in ("glove", "nytimes", "fashion-mnist") else x
    return _split(x, n_heldout, seed, n_train)


def synthetic(
    kind: str,
    n: int,
    d: int,
    seed: int = 0,
    n_heldout: int = 20000,
    n_components: int | None = None,
    sigma_w2: float = 1.0 / 3.0,
    sigma_mu2: float = 2.0 / 3.0,
    zipf_a: float = 1.0,
    overlap: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic distributions, unit variance per coordinate (paper, Appendix D.2).

    kind:
      "uniform"   uniform cube
      "gauss"     isotropic Gaussian
      "gmm"       Gaussian mixture with ``n_components`` equal-weight components,
                  within variance sigma_w2 * overlap, center variance sigma_mu2
      "gmm-zipf"  same with Zipf(a) component weights (imbalanced clusters)
      "stretched" Gaussian with variance decaying as j^-a along coordinates (a=1)
    """
    rng = np.random.RandomState(seed)
    total = n + n_heldout
    if kind == "uniform":
        x = rng.uniform(-np.sqrt(3), np.sqrt(3), size=(total, d))
    elif kind == "gauss":
        x = rng.standard_normal((total, d))
    elif kind == "stretched":
        scale = (np.arange(1, d + 1) ** (-zipf_a)).astype(np.float64)
        scale *= np.sqrt(d / (scale**2).sum())
        x = rng.standard_normal((total, d)) * scale
    elif kind in ("gmm", "gmm-zipf"):
        g = n_components or 1024
        mu = rng.standard_normal((g, d)) * np.sqrt(sigma_mu2)
        if kind == "gmm-zipf":
            w = 1.0 / np.arange(1, g + 1) ** zipf_a
        else:
            w = np.ones(g)
        w /= w.sum()
        comp = rng.choice(g, size=total, p=w)
        x = mu[comp] + rng.standard_normal((total, d)) * np.sqrt(sigma_w2 * overlap)
    else:
        raise ValueError(kind)
    x = x.astype(np.float32)
    return x[:n], x[n:]


def load(name: str, **kw) -> tuple[np.ndarray, np.ndarray]:
    """``name`` is an HDF5 key or ``syn:<kind>:<n>:<d>[:<components>]``."""
    if name.startswith("syn:"):
        parts = name.split(":")
        kind, n, d = parts[1], int(float(parts[2])), int(parts[3])
        comps = int(parts[4]) if len(parts) > 4 else None
        return synthetic(kind, n, d, n_components=comps, **kw)
    return load_hdf5(name, **kw)
