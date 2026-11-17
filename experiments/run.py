"""Run one clustering configuration and append the result as one JSON line.

Example:
  python experiments/run.py --algo emeans --dataset sift --k 16384 --epochs 3 --seed 0 --out repro
  python experiments/run.py --algo lloyd --init kmeans++ --dataset sift --k 16384 --epochs 25 --seed 0 --out repro

Results are appended to experiments/results/<out>.jsonl and never overwritten.
Loss curves are evaluated after every epoch on a fixed 100k training subsample
and on the held-out set (these evaluation distances are not counted as
algorithmic distance computations). The final centroids are evaluated on the
full training set.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from hmeans import backend as B  # noqa: E402
from hmeans import data  # noqa: E402
from hmeans.baselines import Lloyd, SculleyMiniBatch  # noqa: E402
from hmeans.emeans_ref import EMeansRef  # noqa: E402

RESULTS = os.path.join(ROOT, "experiments", "results")


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "nogit"


def build(args, device):
    kw = dict(k=args.k, seed=args.seed, device=device)
    if args.algo == "emeans":
        return EMeansRef(nepochs=args.epochs, batch_law=args.batch_law, batch_size=args.batch_size, alpha0=args.alpha0, **kw)
    if args.algo == "emeans-official":
        return "official"
    if args.algo == "lloyd":
        return Lloyd(n_iter=int(args.epochs), init=args.init, init_sample=args.init_sample, init_trials=args.init_trials, **kw)
    if args.algo == "sculley":
        return SculleyMiniBatch(nepochs=args.epochs, batch_size=args.batch_size or 4096, reassignment_ratio=args.reassign, **kw)
    if args.algo == "hmeans":
        from hmeans.hmeans import HMeansCore as HMeans

        opts = json.loads(args.hopts) if args.hopts else {}
        opts = {"batch_law": args.batch_law, "batch_size": args.batch_size, "alpha0": args.alpha0, **opts}
        return HMeans(nepochs=args.epochs, **opts, **kw)
    raise ValueError(args.algo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["emeans", "emeans-official", "lloyd", "sculley", "hmeans"])
    p.add_argument("--dataset", required=True)
    p.add_argument("--n", type=int, default=None, help="training rows to use (default all)")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--epochs", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--init", default="random")
    p.add_argument("--init-sample", type=int, default=None)
    p.add_argument("--init-trials", type=int, default=None, help="greedy k-means++ local trials (default 2+log k)")
    p.add_argument("--batch-law", default="auto")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--alpha0", type=float, default=None)
    p.add_argument("--reassign", type=float, default=0.01)
    p.add_argument("--hopts", default=None, help="JSON dict of HMeans options")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--no-curve", action="store_true")
    p.add_argument("--eval-sample", type=int, default=100_000)
    args = p.parse_args()

    device = B.pick_device(args.device)
    torch.manual_seed(args.seed)
    t_load = time.perf_counter()
    xtr_np, xte_np = data.load(args.dataset, n_train=args.n) if not args.dataset.startswith("syn:") else data.load(args.dataset)
    if args.dataset.startswith("syn:") and args.n:
        xtr_np = xtr_np[: args.n]
    n, d = xtr_np.shape
    xtr = B.to_device(xtr_np, device)
    xte = B.to_device(xte_np, device)
    # evaluation subsample from its own RNG stream (seed 0 would coincide with the random init rows)
    sub = torch.from_numpy(np.random.RandomState(987654321).permutation(n)[: min(args.eval_sample, n)]).to(device)
    xsub = xtr[sub]
    t_load = time.perf_counter() - t_load

    curve = []
    curve_eval_time = [0.0]

    def callback(epoch_idx, centroids):
        if args.no_curve or centroids is None:
            return
        B.synchronize(device)
        t = time.perf_counter()
        tr = B.exact_mse(xsub, centroids)
        te = B.exact_mse(xte, centroids)
        B.synchronize(device)
        curve_eval_time[0] += time.perf_counter() - t
        curve.append(dict(epoch_idx=epoch_idx, train_sub_mse=tr, test_mse=te, ops=B.OPS.pairs))

    B.OPS.reset()
    model = build(args, device)
    extra = {}
    with B.Timer(device) as timer:
        if model == "official":
            import faiss  # noqa: F401
            from faiss.contrib.e_means import NumpyEmeans

            em = NumpyEmeans(d, args.k, nepochs=args.epochs, seed=args.seed, batch_law=args.batch_law,
                             batch_size=args.batch_size, alpha0=args.alpha0)
            cents, _ = em.train(xtr_np)
            centroids = B.to_device(cents, device)
            extra = dict(recipe={k: v for k, v in em.recipe.items() if k in ("batch_size", "alpha0", "batch_law", "total_steps")}, n_splits=em.n_splits)
            B.OPS.add(n * args.k * args.epochs, "assign")
            history = [dict(epoch=i + 1, moving_obj=s["obj"], time=s["time"]) for i, s in enumerate(em.iteration_stats)]
        else:
            model.fit(xtr, callback=callback)
            centroids = model.centroids_
            history = model.history_
            extra = dict(recipe=getattr(model, "recipe_", None), n_splits=getattr(model, "n_splits_", None),
                         init_time=getattr(model, "init_time_", None), extra=getattr(model, "extra_", None))
    fit_time = timer.t - curve_eval_time[0]

    with B.Timer(device) as t_eval:
        train_mse = B.exact_mse(xtr, centroids)
        test_mse = B.exact_mse(xte, centroids)
        labels, _ = B.assign_exact(xtr, centroids, count=False)
        counts = torch.bincount(labels, minlength=args.k).cpu().double()
        imbalance = float((counts * counts).sum().item() * args.k / (n * n))
        n_empty = int((counts == 0).sum().item())

    rec = dict(
        time=dt.datetime.now().isoformat(timespec="seconds"),
        commit=git_hash(),
        tag=args.tag,
        algo=args.algo,
        dataset=args.dataset,
        n=n,
        n_requested=args.n,
        d=d,
        k=args.k,
        epochs=args.epochs,
        seed=args.seed,
        device=str(device),
        init=args.init,
        init_sample=args.init_sample,
        init_trials=args.init_trials,
        batch_law=args.batch_law,
        batch_size=args.batch_size,
        alpha0=args.alpha0,
        hopts=json.loads(args.hopts) if args.hopts else None,
        reassign=args.reassign if args.algo == "sculley" else None,
        train_mse=train_mse,
        test_mse=test_mse,
        imbalance=imbalance,
        n_empty=n_empty,
        fit_time=fit_time,
        eval_time=t_eval.t,
        load_time=t_load,
        ops=B.OPS.snapshot(),
        ops_per_epoch_equiv=B.OPS.pairs / (n * args.k),
        curve=curve,
        history=history,
        **extra,
    )
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, args.out + ".jsonl"), "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(json.dumps({k: rec[k] for k in ("algo", "dataset", "n", "d", "k", "epochs", "seed", "train_mse", "test_mse", "fit_time", "ops_per_epoch_equiv", "n_empty")}, default=float))


if __name__ == "__main__":
    main()
