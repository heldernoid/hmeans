"""Regenerate every table and figure in REPORT.md from experiments/results/*.jsonl.

  python experiments/analyze.py            # writes experiments/figures/*.png and experiments/tables/*.md

Nothing here runs clustering; it only aggregates logged runs (mean and standard
deviation over seeds).
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")
FIG = os.path.join(ROOT, "experiments", "figures")
TAB = os.path.join(ROOT, "experiments", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# dataviz palette (brand-neutral, light/dark safe)
COLORS = {
    "lloyd": "#7f7f7f",
    "kmeans++": "#4d4d4d",
    "sculley": "#b8860b",
    "e-means": "#1f77b4",
    "e-means quality": "#6baed6",
    "h-means": "#d62728",
    "h-means quality": "#ff9896",
}


def load(name: str) -> list[dict]:
    path = os.path.join(RESULTS, name + ".jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def short_ds(ds: str) -> str:
    return {"syn:gmm-zipf:500000:96:2048": "GMM-Zipf (d=96)", "syn:gmm:500000:96:4096": "GMM (d=96)",
            "sift": "SIFT-128", "glove": "GloVe-100", "nytimes": "NYTimes-256", "fashion-mnist": "Fashion-MNIST-784",
            "deep": "Deep-96 (10M)"}.get(ds, ds)


def auto_ema_frac(n: int, k: int) -> float:
    r = n / max(k, 1)
    return 0.4 if r < 24 else 0.8 if r < 64 else 1.0


def hrecipe(r: dict) -> tuple[bool, float]:
    """(hartigan_ema, ema_frac) of a logged h-means run, resolving the defaults of its time."""
    h = r.get("hopts") or {}
    if "hartigan_ema" in h:
        he = bool(h["hartigan_ema"])
    else:
        he = not (r.get("time", "") < "2026-11-16T20")  # default flipped to True that evening
    return he, float(h.get("ema_frac", 0.4))


def hversion(r: dict) -> str:
    """'' for the final recipe (Hartigan-EMA with the adaptive split: ema_frac 0.4 below 24 points per
    cluster, 0.8 below 64, 1.0 above), ' v1' for no Hartigan-EMA, ' (EMA f)' for a fixed split that
    differs from the adaptive choice for this (n, k)."""
    he, ef = hrecipe(r)
    if not he:
        return " v1"
    if ef == auto_ema_frac(r["n"], r["k"]):
        return ""
    return f" (EMA {ef:g})"


def method_name(r: dict) -> str:
    t = r.get("tag") or ""
    if r["algo"] == "lloyd":
        if r.get("init", "random") == "kmeans++":
            return "k-means++ (full data) Lloyd@25" if not r.get("init_sample") else f"k-means++ ({r['init_sample'] // 1000}k sample) Lloyd@25"
        return "Lloyd@25 (random init)"
    if r["algo"] == "sculley":
        return "Sculley mini-batch T5"
    if r["algo"] in ("emeans", "emeans-official"):
        law = " quality" if r.get("batch_law") == "quality" else ""
        off = " (official faiss)" if r["algo"] == "emeans-official" else ""
        return f"e-means T{int(r['epochs'])}{law}{off}"
    if r["algo"] == "hmeans":
        law = " quality" if r.get("batch_law") == "quality" else ""
        return f"h-means{hversion(r)} T{int(r['epochs'])}{law}"
    return t


def agg(rows: list[dict], key=lambda r: (r["dataset"], r["k"], method_name(r))) -> dict:
    g = defaultdict(list)
    for r in rows:
        g[key(r)].append(r)
    out = {}
    for kk, rs in g.items():
        out[kk] = dict(
            n=len(rs),
            train=np.array([r["train_mse"] for r in rs]),
            test=np.array([r["test_mse"] for r in rs]),
            time=np.array([r["fit_time"] for r in rs]),
            ops=np.array([r["ops_per_epoch_equiv"] for r in rs]),
            macs=np.array([r["ops"].get("macs", float("nan")) for r in rs]),
            n_points=rs[0]["n"], d=rs[0]["d"],
            rows=rs,
        )
    return out


def fmt(a: np.ndarray, rel: float | None = None, digits=4) -> str:
    m, s = a.mean(), a.std()
    if rel is not None:
        return f"{100 * (m / rel - 1):+.2f}% (±{100 * s / rel:.2f})"
    return f"{m:.{digits}g} ± {s:.{digits - 2}g}" if len(a) > 1 else f"{m:.{digits}g}"


def main_table(rows: list[dict]) -> str:
    """Per (dataset, k): train/test MSE relative to e-means T5, time, passes, for every method."""
    a = agg(rows)
    configs = sorted({(r["dataset"], r["k"]) for r in rows}, key=lambda x: (x[0], x[1]))
    order = ["Lloyd@25 (random init)", "k-means++ (50k sample) Lloyd@25", "k-means++ (full data) Lloyd@25", "Sculley mini-batch T5", "e-means T3", "e-means T5",
             "e-means T5 quality", "h-means v1 T3", "h-means v1 T5", "h-means v1 T5 quality",
             "h-means (EMA 0.4) T3", "h-means (EMA 0.4) T5", "h-means (EMA 0.4) T5 quality",
             "h-means (EMA 0.8) T3", "h-means (EMA 0.8) T5", "h-means (EMA 0.8) T5 quality",
             "h-means (EMA 1) T3", "h-means (EMA 1) T5", "h-means (EMA 1) T5 quality",
             "h-means T3", "h-means T5", "h-means T5 quality"]
    lines = []
    for ds, k in configs:
        ref = a.get((ds, k, "e-means T5"))
        if ref is None:
            continue
        n = ref["n_points"]
        lines.append(f"\n### {short_ds(ds)}, n = {n:,}, k = {k:,} (n/k = {n / k:.0f})\n")
        lines.append("| method | seeds | train MSE | vs e-means T5 | test MSE | vs e-means T5 | passes (n·k dist.) | time (s) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m in order:
            if (ds, k, m) not in a:
                continue
            e = a[(ds, k, m)]
            lines.append(f"| {m} | {e['n']} | {fmt(e['train'], digits=6)} | {fmt(e['train'], ref['train'].mean())} | "
                         f"{fmt(e['test'], digits=6)} | {fmt(e['test'], ref['test'].mean())} | {e['ops'].mean():.2f} | {e['time'].mean():.1f} ± {e['time'].std():.1f} |")
    return "\n".join(lines)


def win_summary(rows: list[dict]) -> str:
    """h-means T5 vs e-means T5 (both laws) and h-means T3 vs e-means T5: relative train MSE by n/k."""
    a = agg(rows)
    configs = sorted({(r["dataset"], r["k"]) for r in rows}, key=lambda x: (x[0], x[1]))
    lines = ["| dataset | k | n/k | h-means T5 vs e-means T5 | h-means T5 vs e-means T5 quality | h-means T3 vs e-means T5 | h-means T5 vs Lloyd@25 | seeds with h-means T5 < best e-means T5 | held-out: h-means T5 vs best e-means T5 | time (s): e-means T5 / h-means T5 / h-means T3 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, k in configs:
        h5 = a.get((ds, k, "h-means T5"))
        e5 = a.get((ds, k, "e-means T5"))
        if h5 is None or e5 is None:
            continue
        e5q = a.get((ds, k, "e-means T5 quality"))
        h3 = a.get((ds, k, "h-means T3"))
        ll = a.get((ds, k, "Lloyd@25 (random init)"))
        n = h5["n_points"]
        best_e = min(e5["train"].mean(), e5q["train"].mean() if e5q else np.inf)
        wins = int((h5["train"] < best_e).sum())
        best_e_test = e5["test"].mean() if (e5q is None or e5["train"].mean() <= e5q["train"].mean()) else e5q["test"].mean()
        lines.append(f"| {short_ds(ds)} | {k:,} | {n / k:.0f} | {fmt(h5['train'], e5['train'].mean())} | "
                     f"{fmt(h5['train'], e5q['train'].mean()) if e5q else '-'} | {fmt(h3['train'], e5['train'].mean()) if h3 else '-'} | "
                     f"{fmt(h5['train'], ll['train'].mean()) if ll else '-'} | {wins}/{h5['n']} | {fmt(h5['test'], best_e_test)} | "
                     f"{e5['time'].mean():.1f} / {h5['time'].mean():.1f} / {h3['time'].mean() if h3 else float('nan'):.1f} |")
    return "\n".join(lines)


def ablation_table(rows: list[dict]) -> str:
    a = agg(rows, key=lambda r: (r["dataset"], r["k"], r.get("tag")))
    configs = sorted({(r["dataset"], r["k"]) for r in rows})
    lines = []
    for ds, k in configs:
        ref = a.get((ds, k, "h-means (full)"))
        if ref is None:
            continue
        lines.append(f"\n### Ablation: {short_ds(ds)}, k = {k:,} (5 passes, 5 seeds)\n")
        lines.append("| variant | train MSE | vs full h-means | test MSE | vs full | time (s) |")
        lines.append("|---|---|---|---|---|---|")
        for (d2, k2, tag), e in sorted(a.items(), key=lambda kv: kv[1]["train"].mean()):
            if d2 != ds or k2 != k:
                continue
            lines.append(f"| {tag} | {fmt(e['train'], digits=6)} | {fmt(e['train'], ref['train'].mean())} | {fmt(e['test'], digits=6)} | {fmt(e['test'], ref['test'].mean())} | {e['time'].mean():.1f} |")
    return "\n".join(lines)


def cpu_table(rows: list[dict]) -> str:
    a = agg(rows, key=lambda r: (r["dataset"], r["k"], (r.get("tag") or "").replace("h-means", "h-means" + hversion(r)) if r["algo"] == "hmeans" else r.get("tag")))
    lines = ["| dataset | k | method | train MSE | time (s) | passes |", "|---|---|---|---|---|---|"]
    for (ds, k, tag), e in sorted(a.items()):
        lines.append(f"| {short_ds(ds)} | {k:,} | {tag} | {fmt(e['train'], digits=6)} | {e['time'].mean():.1f} ± {e['time'].std():.1f} | {e['ops'].mean():.2f} |")
    return "\n".join(lines)


def scaling_table(rows: list[dict]) -> str:
    a = agg(rows)
    ks = sorted({r["k"] for r in rows})
    lines = ["| k | n/k | e-means T5 | e-means T5 quality | h-means T5 | h-means vs e-means T5 | h-means vs e-means T5 quality |", "|---|---|---|---|---|---|---|"]
    for k in ks:
        e = a.get(("sift", k, "e-means T5"))
        q = a.get(("sift", k, "e-means T5 quality"))
        h = a.get(("sift", k, "h-means T5"))
        if not (e and h):
            continue
        n = e["n_points"]
        lines.append(f"| {k:,} | {n / k:.0f} | {fmt(e['train'], digits=6)} | {fmt(q['train'], digits=6) if q else '-'} | {fmt(h['train'], digits=6)} | "
                     f"{fmt(h['train'], e['train'].mean())} | {fmt(h['train'], q['train'].mean()) if q else '-'} |")
    return "\n".join(lines)


def plot_curves(rows: list[dict]):
    """Held-out MSE after every pass (seed 0), final h-means recipe and the baselines only. (The
    100k-subsample train curve of the earliest runs overlapped the random-init rows and is not
    comparable across runs, so only the held-out curve is plotted.)"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = sorted({(r["dataset"], r["k"]) for r in rows if r.get("curve")})
    for ds, k in configs:
        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=130)
        for r in rows:
            if (r["dataset"], r["k"]) != (ds, k) or not r.get("curve") or r["seed"] != 0:
                continue
            m = method_name(r)
            if " v1 " in m or "(EMA" in m:
                continue
            base = "h-means" if r["algo"] == "hmeans" else ("e-means" if r["algo"] == "emeans" else ("sculley" if r["algo"] == "sculley" else "lloyd"))
            if r.get("batch_law") == "quality":
                base += " quality"
            xs = [c["ops"] / (r["n"] * r["k"]) for c in r["curve"]] + [r["ops_per_epoch_equiv"]]
            ys = [c["test_mse"] for c in r["curve"]] + [r["test_mse"]]
            ax.plot(xs, ys, marker="o", ms=3, lw=1.3, color=COLORS.get(base, "#333"), label=m, ls="--" if "T3" in m else "-")
        ax.set_xlabel("passes over the data (n·k distance computations)")
        ax.set_ylabel("held-out MSE")
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        finals = [np.array(l.get_ydata())[-1] for l in ax.get_lines() if len(l.get_ydata())]
        if finals:
            ax.set_ylim(min(finals) * 0.995, max(finals) * 1.006)
        ax.legend(fontsize=7)
        n = next(r["n"] for r in rows if (r["dataset"], r["k"]) == (ds, k))
        ax.set_title(f"{short_ds(ds)}, n = {n:,}, k = {k:,} (seed 0)", fontsize=10)
        fig.tight_layout()
        fname = f"curve_{ds.replace(':', '_')}_k{k}.png"
        fig.savefig(os.path.join(FIG, fname))
        plt.close(fig)


def plot_wins(rows: list[dict]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = agg(rows)
    pts = []
    for (ds, k, m), e in a.items():
        if m != "h-means T5":
            continue
        e5 = a.get((ds, k, "e-means T5"))
        q5 = a.get((ds, k, "e-means T5 quality"))
        if e5 is None:
            continue
        best = min(e5["train"].mean(), q5["train"].mean() if q5 else np.inf)
        pts.append((e["n_points"] / k, 100 * (e["train"].mean() / best - 1), 100 * e["train"].std() / best, short_ds(ds)))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    dss = sorted({p[3] for p in pts})
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]
    for i, ds in enumerate(dss):
        sel = [p for p in pts if p[3] == ds]
        ax.errorbar([p[0] for p in sel], [p[1] for p in sel], yerr=[p[2] for p in sel], fmt="o", color=palette[i % len(palette)], label=ds, capsize=3)
    ax.axhline(0, color="#888", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("n / k (points per cluster)")
    ax.set_ylabel("h-means T5 train MSE vs best e-means T5 (%)")
    ax.set_title("h-means vs e-means at 5 passes (negative = h-means better)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "wins_vs_nk.png"))
    plt.close(fig)


def main():
    auto = load("auto")
    main_rows = load("main") + load("v2") + load("v3") + [r for r in auto if r["dataset"] not in ("deep", "sift") or (r["dataset"], r["k"]) == ("sift", 16384)]
    kpp = load("kmeanspp") + load("kmeanspp_full")
    abl = load("ablation_v3") or load("ablation")
    cpu = load("cpu") + load("cpu_v3")
    scal = load("scaling") + load("scaling_v3") + [r for r in auto if r["dataset"] == "sift" and r["k"] != 16384]
    deep = load("deep") + load("deep_v3") + [r for r in auto if r["dataset"] == "deep"]
    with open(os.path.join(TAB, "main.md"), "w") as f:
        f.write("# Main results (mean ± std over seeds)\n")
        f.write(main_table(main_rows + kpp + deep))
    with open(os.path.join(TAB, "wins.md"), "w") as f:
        f.write("# h-means vs e-means summary\n\n")
        f.write(win_summary(main_rows + deep))
    with open(os.path.join(TAB, "ablation.md"), "w") as f:
        f.write("# Ablations (relative to the %s recipe)\n" % ("final" if load("ablation_v3") else "v2 (ema_frac 0.8)"))
        f.write(ablation_table(abl))
    if load("ablation_v3") and load("ablation"):
        with open(os.path.join(TAB, "ablation_v2.md"), "w") as f:
            f.write("# Ablations relative to the v2 recipe (ema_frac 0.8), run before v3 was chosen\n")
            f.write(ablation_table(load("ablation")))
    with open(os.path.join(TAB, "cpu.md"), "w") as f:
        f.write("# CPU wall-clock (torch CPU backend, 3 seeds)\n\n")
        f.write(cpu_table(cpu))
    with open(os.path.join(TAB, "scaling.md"), "w") as f:
        f.write("# SIFT: sweep over k at fixed n (3 seeds, T = 5)\n\n")
        f.write(scaling_table(scal))
    plot_curves(main_rows + deep)
    plot_wins(main_rows + deep + scal)
    print("tables:", sorted(os.listdir(TAB)))
    print("figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
