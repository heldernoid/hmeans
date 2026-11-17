import numpy as np
import pytest

from hmeans import EMeans, HMeans

try:
    import torch
except ImportError:  # the numpy-only install
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch backend not installed")

rng = np.random.RandomState(0)
CENTERS = rng.randn(64, 16).astype("float32") * 4
X = (CENTERS[rng.randint(0, 64, 6000)] + rng.randn(6000, 16).astype("float32")).astype("float32")
# a less separable, larger problem for the quality checks (n/k = 59): 256 components, k = 512
rng2 = np.random.RandomState(1)
C2 = rng2.randn(256, 32).astype("float32") * 3
X2 = (C2[rng2.randint(0, 256, 30000)] + rng2.randn(30000, 32).astype("float32")).astype("float32")


@needs_torch
def test_fit_predict_shapes_and_types():
    km = HMeans(n_clusters=64, n_passes=3, device="cpu", random_state=0).fit(X)
    assert km.cluster_centers_.shape == (64, 16) and km.cluster_centers_.dtype == np.float32
    assert km.labels_.shape == (6000,) and km.labels_.dtype == np.int64
    assert km.labels_.min() >= 0 and km.labels_.max() < 64
    assert km.n_features_in_ == 16 and km.inertia_ > 0 and 3 <= km.n_passes_ < 4  # warm start costs ~2/k_coarse passes, large at k = 64
    p = km.predict(X[:100])
    assert p.shape == (100,) and (p == km.labels_[:100]).all()      # labels_ == predict(X) on training data
    d = km.transform(X[:10])
    assert d.shape == (10, 64) and np.allclose(d.argmin(1), km.labels_[:10])
    assert km.score(X) == pytest.approx(-km.inertia_, rel=1e-4)


@needs_torch
def test_accepts_torch_and_float64():
    a = HMeans(n_clusters=16, n_passes=2, device="cpu", random_state=1).fit(torch.from_numpy(X))
    b = HMeans(n_clusters=16, n_passes=2, device="cpu", random_state=1).fit(X.astype("float64"))
    assert np.allclose(a.cluster_centers_, b.cluster_centers_, atol=1e-4)


@needs_torch
def test_reproducible_and_seed_matters():
    a = HMeans(n_clusters=32, n_passes=2, device="cpu", random_state=3).fit(X)
    b = HMeans(n_clusters=32, n_passes=2, device="cpu", random_state=3).fit(X)
    c = HMeans(n_clusters=32, n_passes=2, device="cpu", random_state=4).fit(X)
    assert np.allclose(a.cluster_centers_, b.cluster_centers_)
    assert not np.allclose(a.cluster_centers_, c.cluster_centers_)


@needs_torch
def test_beats_random_and_close_to_lloyd():
    from hmeans import Lloyd, exact_sse
    km = HMeans(n_clusters=512, n_passes=5, device="cpu", random_state=0).fit(X2)
    ll = Lloyd(k=512, n_iter=25, seed=0, device="cpu").fit(torch.from_numpy(X2))
    lloyd = float(exact_sse(torch.from_numpy(X2), ll.centroids_)[0])
    random_centroids = float(exact_sse(torch.from_numpy(X2), torch.from_numpy(X2[:512]))[0])
    assert km.inertia_ < 0.7 * random_centroids
    assert km.inertia_ < 1.03 * lloyd
    e = EMeans(n_clusters=512, n_passes=5, device="cpu", random_state=0).fit(X2)
    assert km.inertia_ < 1.02 * e.inertia_


@needs_torch
def test_emeans_is_core_with_switches_off():
    from hmeans import HMeansCore
    e = EMeans(n_clusters=32, n_passes=3, device="cpu", random_state=0).fit(X)
    core = HMeansCore(k=32, nepochs=3, device="cpu", seed=0, warm_start=None, hartigan_ema=False, tail=None, donor="usage").fit(torch.from_numpy(X))
    assert np.allclose(e.cluster_centers_, core.centroids_.numpy(), atol=1e-5)
    assert "warm_start" not in e.get_params()


def test_validation():
    with pytest.raises(ValueError):
        HMeans(n_clusters=0).fit(X)
    with pytest.raises(ValueError):
        HMeans(n_clusters=10000).fit(X)
    with pytest.raises(ValueError):
        HMeans(n_clusters=4).fit(X[:, 0])
    km = HMeans(n_clusters=4, n_passes=1, backend="numpy").fit(X)
    with pytest.raises(ValueError):
        km.predict(X[:, :5])
    with pytest.raises(ValueError):
        HMeans(n_clusters=4, backend="jax").fit(X)
    with pytest.raises(AttributeError):
        HMeans(n_clusters=4).predict(X)


def test_sklearn_integration():
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    km = HMeans(n_clusters=8, n_passes=2, backend="numpy", random_state=0)
    assert clone(km).get_params() == km.get_params()
    pipe = make_pipeline(StandardScaler(), km).fit(X)
    assert pipe.predict(X[:5]).shape == (5,)
    labels = HMeans(n_clusters=8, n_passes=2, backend="numpy", random_state=0).fit_predict(X)
    assert labels.shape == (6000,)


# ---------------------------------------------------------------- numpy backend
@needs_torch
def test_numpy_backend_matches_torch_when_assignment_is_plain():
    kw = dict(n_clusters=64, n_passes=5, random_state=0, hartigan_ema=False, ema_frac=0.4)
    a = HMeans(backend="numpy", **kw).fit(X)
    b = HMeans(backend="torch", device="cpu", **kw).fit(X)
    assert a.backend_ == "numpy" and b.backend_ == "torch"
    assert np.allclose(a.cluster_centers_, b.cluster_centers_, atol=1e-3)
    assert a.inertia_ == pytest.approx(b.inertia_, rel=1e-4)


@needs_torch
def test_numpy_backend_full_recipe_close_to_torch():
    kw = dict(n_clusters=64, n_passes=5, random_state=0)
    a = HMeans(backend="numpy", **kw).fit(X)
    b = HMeans(backend="torch", device="cpu", **kw).fit(X)
    assert a.recipe_["ema_frac"] == b.recipe_["ema_frac"]
    assert abs(a.inertia_ / b.inertia_ - 1) < 0.03   # same algorithm; float summation order differs


def test_numpy_backend_api_and_auto_split():
    km = HMeans(n_clusters=64, n_passes=5, backend="numpy", random_state=0).fit(X)
    assert km.cluster_centers_.shape == (64, 16) and km.labels_.shape == (6000,) and km.inertia_ > 0
    assert km.recipe_["ema_frac"] == 1.0            # 6000 / 64 = 94 points per cluster -> no exact tail
    assert (km.predict(X[:50]) == km.labels_[:50]).all()
    assert km.transform(X[:5]).shape == (5, 64) and km.score(X) == pytest.approx(-km.inertia_, rel=1e-4)
    small = HMeans(n_clusters=600, n_passes=5, backend="numpy", random_state=0).fit(X)
    assert small.recipe_["ema_frac"] == 0.4         # 10 points per cluster
    e = EMeans(n_clusters=64, n_passes=5, backend="numpy", random_state=0).fit(X)
    assert e.inertia_ > 0 and e.recipe_["tail"] is None


def test_auto_split_thresholds():
    from hmeans.laws import auto_ema_frac
    assert auto_ema_frac(1000, 100) == 0.4 and auto_ema_frac(2400, 100) == 0.8 and auto_ema_frac(6400, 100) == 1.0
