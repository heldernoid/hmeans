"""Export the numbers the site shows (site/src/data/results.json) from experiments/results/*.jsonl.

Everything the site renders as a table or chart comes from this file; nothing is typed by hand.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import agg, hversion, load, method_name, short_ds  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "site", "src", "data", "results.json")


def stats(e):
    return dict(mean=float(e["train"].mean()), std=float(e["train"].std()), test=float(e["test"].mean()), test_std=float(e["test"].std()),
                time=float(e["time"].mean()), passes=float(e["ops"].mean()), n=int(e["n"]))


def main():
    auto = load("auto")
    rows = (load("main") + load("v2") + load("v3") + load("kmeanspp") + load("kmeanspp_full") + load("deep") + load("deep_v3")
            + [r for r in auto if r["dataset"] != "sift" or r["k"] == 16384])
    a = agg(rows)
    configs = sorted({(r["dataset"], r["k"]) for r in rows}, key=lambda x: (x[0], x[1]))
    grid = []
    for ds, k in configs:
        e5 = a.get((ds, k, "e-means T5")); h5 = a.get((ds, k, "h-means T5"))
        if e5 is None or h5 is None:
            continue
        methods = {}
        for (d2, k2, m), e in a.items():
            if (d2, k2) == (ds, k):
                methods[m] = stats(e)
        e5q = a.get((ds, k, "e-means T5 quality"))
        best = "e-means T5 quality" if (e5q is not None and e5q["train"].mean() < e5["train"].mean()) else "e-means T5"
        be = a[(ds, k, best)]
        grid.append(dict(dataset=ds, name=short_ds(ds), k=k, n=int(e5["n_points"]), d=int(e5["d"]), nk=e5["n_points"] / k, best_emeans=best,
                         vs_best=100 * (h5["train"].mean() / be["train"].mean() - 1), vs_best_std=100 * h5["train"].std() / be["train"].mean(),
                         vs_default=100 * (h5["train"].mean() / e5["train"].mean() - 1),
                         test_vs_best=100 * (h5["test"].mean() / be["test"].mean() - 1),
                         seeds_won=int((h5["train"] < be["train"].mean()).sum()), seeds=int(h5["n"]), methods=methods))
    # SIFT k sweep (final recipe rows from scaling_v3, baselines from scaling)
    sw = agg(load("scaling") + load("scaling_v3") + [r for r in auto if r["dataset"] == "sift" and r["k"] != 16384])
    sweep = []
    for k in sorted({k for (_, k, _) in sw}):
        e = sw.get(("sift", k, "e-means T5")); q = sw.get(("sift", k, "e-means T5 quality")); h = sw.get(("sift", k, "h-means T5"))
        if not (e and q and h):
            continue
        sweep.append(dict(k=k, nk=e["n_points"] / k, emeans=stats(e), emeans_q=stats(q), hmeans=stats(h),
                          vs_best=100 * (h["train"].mean() / min(e["train"].mean(), q["train"].mean()) - 1)))
    # held-out curves (seed 0) for a few configurations
    curves = {}
    for ds, k in [("sift", 65536), ("nytimes", 32768), ("glove", 131072), ("syn:gmm-zipf:500000:96:2048", 4096)]:
        cs = {}
        for r in rows:
            if (r["dataset"], r["k"]) != (ds, k) or r["seed"] != 0 or not r.get("curve"):
                continue
            m = method_name(r)
            if " v1 " in m or "(EMA" in m or m.startswith("k-means++"):
                continue
            cs[m] = dict(x=[c["ops"] / (r["n"] * r["k"]) for c in r["curve"]] + [r["ops_per_epoch_equiv"]], y=[c["test_mse"] for c in r["curve"]] + [r["test_mse"]])
        if cs:
            curves[f"{short_ds(ds)} k={k:,}"] = cs
    # ablations (final recipe if available)
    ab_rows = load("ablation_v3") or load("ablation")
    ab = agg(ab_rows, key=lambda r: (r["dataset"], r["k"], r.get("tag")))
    ablation = {}
    for (ds, k, tag), e in ab.items():
        ref = ab.get((ds, k, "h-means (full)"))
        if ref is None:
            continue
        ablation.setdefault(f"{short_ds(ds)} k={k:,}", []).append(dict(variant=tag, mean=float(e["train"].mean()), vs_full=100 * (e["train"].mean() / ref["train"].mean() - 1), time=float(e["time"].mean())))
    for v in ablation.values():
        v.sort(key=lambda x: x["mean"])
    cpu = agg(load("cpu") + load("cpu_v3"), key=lambda r: (r["dataset"], r["k"], (r.get("tag") or "").replace("h-means", "h-means" + hversion(r)) if r["algo"] == "hmeans" else r.get("tag")))
    cpu_rows = [dict(dataset=short_ds(ds), k=k, method=tag, mean=float(e["train"].mean()), time=float(e["time"].mean()), passes=float(e["ops"].mean())) for (ds, k, tag), e in sorted(cpu.items()) if "(EMA" not in tag]
    out = dict(grid=grid, sweep=sweep, curves=curves, ablation=ablation, cpu=cpu_rows, ablation_recipe="final" if load("ablation_v3") else "v2")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT, "grid", len(grid), "sweep", len(sweep), "curves", list(curves), "ablation", list(ablation), "cpu", len(cpu_rows))


if __name__ == "__main__":
    main()
