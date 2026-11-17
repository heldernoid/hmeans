"""hmeans: h-means clustering (e-means + hierarchical warm start + Hartigan assignment).

>>> from hmeans import HMeans
>>> km = HMeans(n_clusters=1024, n_passes=5).fit(X)      # X: numpy array or torch tensor
>>> km.cluster_centers_, km.labels_, km.inertia_, km.predict(X_new)

Backends: ``backend="numpy"`` needs NumPy only; ``backend="torch"`` (``pip install hmeans[torch]``)
runs on CPU, Apple MPS or CUDA; ``"auto"`` uses torch when installed. ``EMeans`` is the faithful
e-means reimplementation with the same interface. The torch-only classes (``HMeansCore`` with every
ablation switch, ``EMeansRef``, ``Lloyd``, ``SculleyMiniBatch``, the ``backend`` module) load lazily.
"""
from __future__ import annotations

from .core_numpy import HMeansNumpy  # noqa: F401
from .estimator import EMeans, HMeans  # noqa: F401

try:
    from importlib.metadata import version as _version
    __version__ = _version("hmeans")
except Exception:  # pragma: no cover  (not installed)
    __version__ = "0+unknown"

_TORCH_EXPORTS = {
    "HMeansCore": ("hmeans", "HMeansCore"), "EMeansRef": ("emeans_ref", "EMeansRef"), "Lloyd": ("baselines", "Lloyd"),
    "SculleyMiniBatch": ("baselines", "SculleyMiniBatch"), "OPS": ("backend", "OPS"), "pick_device": ("backend", "pick_device"),
    "assign_exact": ("backend", "assign_exact"), "exact_mse": ("backend", "exact_mse"), "exact_sse": ("backend", "exact_sse"),
}
__all__ = ["HMeans", "EMeans", "HMeansNumpy", *_TORCH_EXPORTS]


def __getattr__(name):
    if name in _TORCH_EXPORTS:
        import importlib
        mod, attr = _TORCH_EXPORTS[name]
        try:
            return getattr(importlib.import_module("." + mod, __name__), attr)
        except ImportError as e:
            raise ImportError(f"hmeans.{name} needs PyTorch: pip install 'hmeans[torch]'") from e
    raise AttributeError(f"module 'hmeans' has no attribute {name!r}")
