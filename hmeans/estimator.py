"""scikit-learn style estimators: ``HMeans`` (this project) and ``EMeans`` (faithful e-means).

Both accept numpy arrays or torch tensors, run on NumPy alone (``backend="numpy"``) or on torch
(CPU, Apple MPS or CUDA; ``pip install hmeans[torch]``), and expose the usual
``fit`` / ``predict`` / ``transform`` / ``score`` methods and ``cluster_centers_`` / ``labels_`` /
``inertia_`` attributes. scikit-learn itself is optional: when it is installed the classes inherit
from ``BaseEstimator`` and ``ClusterMixin`` (``get_params``, ``set_params``, ``fit_predict``,
pipelines); without it they behave the same minus those helpers.
"""
from __future__ import annotations

import numbers

import numpy as np

from . import core_numpy as NP


def _torch():
    try:
        import torch  # noqa: F401
        from . import backend as B
        from .hmeans import HMeansCore
        return torch, B, HMeansCore
    except ImportError as e:  # pragma: no cover
        raise ImportError("the torch backend needs PyTorch: pip install 'hmeans[torch]' (or backend='numpy')") from e


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False

try:  # optional
    from sklearn.base import BaseEstimator, ClusterMixin
except ImportError:  # pragma: no cover
    class BaseEstimator:  # type: ignore[no-redef]
        def get_params(self, deep=True):
            import inspect
            return {p: getattr(self, p) for p in inspect.signature(self.__init__).parameters}

        def set_params(self, **params):
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class ClusterMixin:  # type: ignore[no-redef]
        def fit_predict(self, X, y=None, **kw):
            return self.fit(X, **kw).labels_


def _as_array(X, name="X") -> np.ndarray:
    """Validated float32 numpy array (torch tensors are moved to the CPU and converted)."""
    if type(X).__module__.startswith("torch"):
        X = X.detach().cpu().numpy()
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            raise TypeError(f"{name} must be dense; sparse input is not supported")
    except ImportError:  # pragma: no cover
        pass
    a = np.asarray(X)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_samples, n_features), got shape {a.shape}")
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.ascontiguousarray(a)


def _seed(random_state) -> int:
    if random_state is None:
        return int(np.random.SeedSequence().generate_state(1)[0] % (2**31 - 1))
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(2**31 - 1))
    if isinstance(random_state, numbers.Integral):
        return int(random_state)
    raise ValueError("random_state must be None, an int or a numpy RandomState")


class HMeans(ClusterMixin, BaseEstimator):
    """h-means clustering: e-means (mini-batch k-means with EMA statistics) plus a hierarchical,
    loss-proportional warm start and Hartigan's N/(N±1) assignment correction.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of centroids k.
    n_passes : float, default=5
        Total number of passes over the data (each pass is n·k distance computations). The warm
        start adds about 0.1 pass. 3 is a fast setting, 5 the default of the paper's e-means.
    batch_law : {"auto", "quality", "throughput"}, default="auto"
        e-means batch-size law. "auto" picks "quality" for d <= 16 and "throughput" otherwise (as
        in faiss); "quality" (8x smaller batches) is usually 0.2 to 1.5% better on accelerators.
    batch_size, alpha0 : int / float, optional
        Override the batch law and the peak step size.
    ema_frac : float or "auto", default="auto"
        Fraction of the passes run in e-means EMA mode; the rest are exact Hartigan passes. "auto"
        sets it from the points per cluster n/k: 0.4 below 24, 0.8 below 64, 1.0 (no exact passes)
        above, where the exact tail no longer helps.
    hartigan_ema : bool, default=True
        Use Hartigan's rule (with the EMA mass as the cluster-size estimate) inside the EMA passes.
    warm_start : bool, default=True
        Hierarchical warm start (k/16 coarse centroids, children allocated by SSE).
    tail : {"hartigan", "incremental", None}, default="hartigan"
        What the passes after the EMA phase do: exact Hartigan passes, exact Lloyd-rule passes, or
        nothing (all passes in EMA mode).
    donor : {"sse", "usage"}, default="sse"
        How the donor of a dead-cluster revival is sampled.
    backend : {"auto", "numpy", "torch"}, default="auto"
        "numpy" needs only NumPy (CPU, BLAS matmul). "torch" needs PyTorch (``pip install
        hmeans[torch]``) and runs on CPU, Apple MPS or CUDA. "auto" uses torch when it is installed.
    device : str, default="auto"
        torch backend only: "auto" (CUDA, then MPS, then CPU), "cpu", "cuda", "mps".
    random_state : int or None
        Seed of the random-row initialisation and the batch order.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    labels_ : ndarray of shape (n_samples,)
    inertia_ : float
        Sum of squared distances of the training points to their nearest centroid (exact).
    n_features_in_ : int
    n_passes_ : float
        Passes actually spent, warm start included, in units of n·k distance computations.
    recipe_ : dict
        The resolved batch size, peak step and pass split.

    Examples
    --------
    >>> from hmeans import HMeans
    >>> import numpy as np
    >>> X = np.random.rand(20000, 32).astype("float32")
    >>> km = HMeans(n_clusters=512, n_passes=5, random_state=0).fit(X)
    >>> km.cluster_centers_.shape, km.labels_.shape
    ((512, 32), (20000,))
    """

    _core_kwargs: dict = {}

    def __init__(self, n_clusters=8, n_passes=5, *, batch_law="auto", batch_size=None, alpha0=None, ema_frac="auto",
                 hartigan_ema=True, warm_start=True, tail="hartigan", donor="sse", backend="auto", device="auto", random_state=None):
        self.n_clusters = n_clusters
        self.n_passes = n_passes
        self.batch_law = batch_law
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.ema_frac = ema_frac
        self.hartigan_ema = hartigan_ema
        self.warm_start = warm_start
        self.tail = tail
        self.donor = donor
        self.backend = backend
        self.device = device
        self.random_state = random_state

    # ---- sklearn plumbing
    def _validate(self):
        if not isinstance(self.n_clusters, numbers.Integral) or self.n_clusters < 1:
            raise ValueError(f"n_clusters must be a positive integer, got {self.n_clusters!r}")
        if not (self.n_passes > 0):
            raise ValueError(f"n_passes must be positive, got {self.n_passes!r}")
        if self.tail not in ("hartigan", "incremental", None):
            raise ValueError(f"tail must be 'hartigan', 'incremental' or None, got {self.tail!r}")
        if self.ema_frac != "auto" and not 0 < self.ema_frac <= 1:
            raise ValueError(f"ema_frac must be 'auto' or in (0, 1], got {self.ema_frac!r}")
        if self.backend not in ("auto", "numpy", "torch"):
            raise ValueError(f"backend must be 'auto', 'numpy' or 'torch', got {self.backend!r}")

    def _resolve_backend(self) -> str:
        if self.backend == "auto":
            return "torch" if _torch_available() else "numpy"
        return self.backend

    def _make_core(self, seed: int):
        common = dict(k=int(self.n_clusters), nepochs=float(self.n_passes), batch_law=self.batch_law, batch_size=self.batch_size,
                      alpha0=self.alpha0, ema_frac=self.ema_frac, hartigan_ema=bool(self.hartigan_ema), tail=self.tail,
                      donor=self.donor, seed=seed)
        if self._resolve_backend() == "numpy":
            return NP.HMeansNumpy(warm_start=bool(self.warm_start), **common)
        _, _, HMeansCore = _torch()
        return HMeansCore(warm_start="default" if self.warm_start else None, device=self.device, **common, **self._core_kwargs)

    def fit(self, X, y=None):
        """Compute the clustering. ``y`` is ignored (API compatibility)."""
        self._validate()
        x = _as_array(X)
        n, d = x.shape
        if self.n_clusters > n:
            raise ValueError(f"n_clusters={self.n_clusters} must be <= n_samples={n}")
        seed = _seed(self.random_state)
        core = self._make_core(seed)
        self.backend_ = self._resolve_backend()
        # labels_ are the exact nearest centroids under the final centroids (scikit-learn contract:
        # labels_ == predict(X)); the training labels of the core (EMA / Hartigan rule) can differ.
        # The same pass yields inertia_, so this costs one evaluation pass, not counted in n_passes_.
        if self.backend_ == "numpy":
            core.fit(x)
            self.cluster_centers_ = core.centroids_
            self.n_passes_ = float(core.ops_pairs_ / (n * self.n_clusters))
            self.labels_, self.inertia_ = NP.labels_and_sse(x, self.cluster_centers_)
        else:
            torch, B, _ = _torch()
            ops0 = B.OPS.pairs
            xt = torch.from_numpy(x)
            core.fit(xt)
            self._centers_t_ = core.centroids_.detach()
            self.cluster_centers_ = self._centers_t_.cpu().numpy()
            self.n_passes_ = float((B.OPS.pairs - ops0) / (n * self.n_clusters))
            xd = B.to_device(xt, core.device)
            labels, _ = B.assign_exact(xd, self._centers_t_, count=False)
            self.inertia_ = float(B.exact_sse(xd, self._centers_t_, labels)[0])
            self.labels_ = labels.cpu().numpy().astype(np.int64)
        self._core_ = core
        self.n_features_in_ = int(d)
        self.recipe_ = dict(core.recipe_)
        self.n_iter_ = int(np.ceil(self.n_passes_))
        return self

    def _check_fitted(self):
        if not hasattr(self, "cluster_centers_"):
            raise AttributeError("This estimator is not fitted yet. Call 'fit' first.")

    def _check_X(self, X) -> np.ndarray:
        x = _as_array(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {x.shape[1]} features, but {type(self).__name__} was fitted with {self.n_features_in_}")
        return x

    def predict(self, X):
        """Index of the nearest centroid for each row of X."""
        self._check_fitted()
        x = self._check_X(X)
        if self.backend_ == "numpy":
            return NP.assign_exact(x, self.cluster_centers_)[0]
        torch, B, _ = _torch()
        labels, _ = B.assign_exact(B.to_device(x, self._core_.device), self._centers_t_, count=False)
        return labels.cpu().numpy().astype(np.int64)

    def transform(self, X):
        """Euclidean distance from each row of X to every centroid, shape (n_samples, n_clusters)."""
        self._check_fitted()
        x = self._check_X(X)
        c = self.cluster_centers_
        out = np.empty((x.shape[0], c.shape[0]), dtype=np.float32)
        cn = np.einsum("ij,ij->i", c, c)
        for i in range(0, x.shape[0], 32768):
            xb = x[i:i + 32768]
            d2 = cn[None, :] - 2.0 * (xb @ c.T) + np.einsum("ij,ij->i", xb, xb)[:, None]
            out[i:i + 32768] = np.sqrt(np.maximum(d2, 0.0))
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def score(self, X, y=None):
        """Negative sum of squared distances of X to its nearest centroids (higher is better)."""
        self._check_fitted()
        x = self._check_X(X)
        if self.backend_ == "numpy":
            return -NP.exact_sse(x, self.cluster_centers_)
        torch, B, _ = _torch()
        return -float(B.exact_sse(B.to_device(x, self._core_.device), self._centers_t_)[0])

    def __sklearn_tags__(self):  # sklearn >= 1.6
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        tags.input_tags.allow_nan = False
        return tags


class EMeans(HMeans):
    """e-means (Bessa et al., 2026), the faithful reimplementation of the faiss recipe, with the
    same interface as :class:`HMeans`. Equivalent to ``HMeans(warm_start=False, hartigan_ema=False,
    tail=None, donor="usage")``; those four parameters are fixed here."""

    def __init__(self, n_clusters=8, n_passes=5, *, batch_law="auto", batch_size=None, alpha0=None, backend="auto", device="auto", random_state=None):
        super().__init__(n_clusters, n_passes, batch_law=batch_law, batch_size=batch_size, alpha0=alpha0, ema_frac=1.0,
                         hartigan_ema=False, warm_start=False, tail=None, donor="usage", backend=backend, device=device, random_state=random_state)

    def get_params(self, deep=True):
        p = super().get_params(deep)
        for fixed in ("ema_frac", "hartigan_ema", "warm_start", "tail", "donor"):
            p.pop(fixed, None)
        return p
