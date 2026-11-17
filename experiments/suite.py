"""Run a named experiment plan (a list of run.py invocations) with resume.

  python experiments/suite.py main      # main grid (accelerator)
  python experiments/suite.py ablation  # h-means ablations
  python experiments/suite.py cpu       # CPU wall-clock comparisons + k-means++ baselines
  python experiments/suite.py deep      # Deep-image 10M

Every run appends one line to experiments/results/<plan>.jsonl. A run whose
(algo, dataset, n, k, epochs, seed, device, init, batch_law, hopts, tag) already
appears in that file is skipped, so an interrupted plan can be resumed with the
same command. Nothing is ever overwritten.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")
PY = sys.executable

SEEDS5 = [0, 1, 2, 3, 4]
SEEDS3 = [0, 1, 2]

# (dataset, n or None, [k values])
MAIN_DATASETS = [
    ("sift", None, [16384, 65536, 131072]),
    ("glove", None, [16384, 65536, 131072]),
    ("nytimes", None, [4096, 16384, 32768]),
    ("fashion-mnist", None, [512, 2048]),
    ("syn:gmm-zipf:500000:96:2048", None, [4096, 16384]),
    ("syn:gmm:500000:96:4096", None, [8192, 32768]),
]


def run_key(cfg: dict) -> tuple:
    return (cfg["algo"], cfg["dataset"], cfg.get("n"), cfg["k"], float(cfg["epochs"]), cfg["seed"], cfg["device"],
            cfg.get("init", "random"), cfg.get("batch_law", "auto"), json.dumps(cfg.get("hopts"), sort_keys=True) if cfg.get("hopts") else None,
            cfg.get("tag", ""))


def done_keys(out: str) -> set:
    path = os.path.join(RESULTS, out + ".jsonl")
    keys = set()
    if not os.path.exists(path):
        return keys
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        keys.add((r["algo"], r["dataset"], r.get("n_requested"), r["k"], float(r["epochs"]), r["seed"], r["device"],
                  r.get("init", "random"), r.get("batch_law", "auto"), json.dumps(r.get("hopts"), sort_keys=True) if r.get("hopts") else None,
                  r.get("tag", "")))
    return keys


def cmd(cfg: dict, out: str) -> list[str]:
    c = [PY, os.path.join(ROOT, "experiments", "run.py"), "--algo", cfg["algo"], "--dataset", cfg["dataset"],
         "--k", str(cfg["k"]), "--epochs", str(cfg["epochs"]), "--seed", str(cfg["seed"]), "--device", cfg["device"],
         "--out", out, "--tag", cfg.get("tag", "")]
    if cfg.get("n"):
        c += ["--n", str(cfg["n"])]
    if cfg.get("init"):
        c += ["--init", cfg["init"]]
    if cfg.get("init_sample"):
        c += ["--init-sample", str(cfg["init_sample"])]
    if cfg.get("init_trials"):
        c += ["--init-trials", str(cfg["init_trials"])]
    if cfg.get("batch_law"):
        c += ["--batch-law", cfg["batch_law"]]
    if cfg.get("batch_size"):
        c += ["--batch-size", str(cfg["batch_size"])]
    if cfg.get("reassign") is not None:
        c += ["--reassign", str(cfg["reassign"])]
    if cfg.get("hopts"):
        c += ["--hopts", json.dumps(cfg["hopts"])]
    if cfg.get("no_curve"):
        c += ["--no-curve"]
    return c


def plan_main(device: str) -> list[dict]:
    runs = []
    for ds, n, ks in MAIN_DATASETS:
        for k in ks:
            base = dict(dataset=ds, n=n, k=k, device=device)
            for s in SEEDS3:
                runs.append(dict(base, algo="lloyd", epochs=25, seed=s, tag="lloyd@25 random"))
                runs.append(dict(base, algo="sculley", epochs=5, seed=s, batch_size=4096, reassign=0.01, tag="sculley B4096"))
            for s in SEEDS5:
                runs.append(dict(base, algo="emeans", epochs=3, seed=s, tag="e-means T3"))
                runs.append(dict(base, algo="emeans", epochs=5, seed=s, tag="e-means T5"))
                runs.append(dict(base, algo="emeans", epochs=5, seed=s, batch_law="quality", tag="e-means T5 quality"))
                runs.append(dict(base, algo="hmeans", epochs=3, seed=s, tag="h-means T3"))
                runs.append(dict(base, algo="hmeans", epochs=5, seed=s, tag="h-means T5"))
                runs.append(dict(base, algo="hmeans", epochs=5, seed=s, batch_law="quality", tag="h-means T5 quality"))
    return runs


V2 = {"hartigan_ema": True, "ema_frac": 0.8}  # second recipe (chosen after seeing n/k~60 results, dropped after the full grid)


V3 = {"hartigan_ema": True, "ema_frac": 0.4}  # FINAL recipe (= HMeans class defaults): Hartigan-corrected EMA, 2 EMA + 3 Hartigan passes at T = 5
V1 = {"hartigan_ema": False, "ema_frac": 0.4}  # first recipe (class defaults before 2026-11-16 evening)


def plan_v3(device: str) -> list[dict]:
    runs = []
    for ds, n, ks in MAIN_DATASETS:
        for k in ks:
            base = dict(dataset=ds, n=n, k=k, device=device, algo="hmeans", hopts=V3)
            for s in SEEDS5:
                runs.append(dict(base, epochs=3, seed=s, tag="h-means T3"))
                runs.append(dict(base, epochs=5, seed=s, tag="h-means T5"))
                runs.append(dict(base, epochs=5, seed=s, batch_law="quality", tag="h-means T5 quality"))
    return runs


def plan_v2(device: str) -> list[dict]:
    runs = []
    for ds, n, ks in MAIN_DATASETS:
        for k in ks:
            base = dict(dataset=ds, n=n, k=k, device=device, algo="hmeans", hopts=V2)
            for s in SEEDS5:
                runs.append(dict(base, epochs=3, seed=s, tag="h-means T3"))
                runs.append(dict(base, epochs=5, seed=s, tag="h-means T5"))
                runs.append(dict(base, epochs=5, seed=s, batch_law="quality", tag="h-means T5 quality"))
    return runs


def plan_ablation(device: str) -> list[dict]:
    runs = []
    configs = [("sift", 65536), ("sift", 16384), ("syn:gmm-zipf:500000:96:2048", 4096), ("glove", 65536)]
    variants = {
        "h-means (full)": dict(V2),
        "- warm start": dict(V2, warm_start=None),
        "- SSE allocation (uniform children)": dict(V2, warm_start={"factor": 16, "epochs": 0.5, "local_iters": 2, "alloc": "uniform", "init_mass": 1.0}),
        "- SSE donors (mass donors)": dict(V2, donor="usage"),
        "- Hartigan in EMA phase": dict(V2, hartigan_ema=False),
        "- Hartigan tail (EMA only, same passes)": dict(V2, tail=None),
        "- Hartigan rule in tail (Lloyd-rule tail)": dict(V2, tail="incremental"),
        "ema_frac 0.4 (v1 split)": dict(V2, ema_frac=0.4),
        "ema_frac 1.0 (no tail, Hartigan-EMA only)": dict(V2, tail=None, ema_frac=1.0),
        "e-means (all off)": {"warm_start": None, "donor": "usage", "tail": None},
        "e-means + Hartigan-EMA only": {"warm_start": None, "donor": "usage", "tail": None, "hartigan_ema": True},
    }
    for ds, k in configs:
        for name, h in variants.items():
            for s in SEEDS5:
                runs.append(dict(dataset=ds, k=k, device=device, algo="hmeans", epochs=5, seed=s, hopts=h, tag=name, no_curve=True))
    return runs


def plan_cpu() -> list[dict]:
    runs = []
    for ds, ks in [("sift", [16384, 65536]), ("glove", [16384, 65536])]:
        for k in ks:
            base = dict(dataset=ds, k=k, device="cpu", no_curve=True)
            for s in SEEDS3:
                runs.append(dict(base, algo="emeans", epochs=5, seed=s, tag="e-means T5 (torch cpu)"))
                runs.append(dict(base, algo="hmeans", epochs=5, seed=s, hopts=V2, tag="h-means T5 (torch cpu)"))
                runs.append(dict(base, algo="hmeans", epochs=3, seed=s, hopts=V2, tag="h-means T3 (torch cpu)"))
                runs.append(dict(base, algo="emeans-official", epochs=5, seed=s, tag="e-means official (faiss cpu) T5"))
                if k == 16384:
                    runs.append(dict(base, algo="lloyd", epochs=25, seed=s, tag="lloyd@25 (torch cpu)"))
    return runs


def plan_kmeanspp(device: str) -> list[dict]:
    runs = []
    for ds in ["sift", "glove", "nytimes"]:
        k = 4096 if ds == "nytimes" else 16384
        for s in SEEDS3:
            runs.append(dict(dataset=ds, k=k, device=device, algo="lloyd", epochs=25, seed=s, init="kmeans++", init_sample=50000,
                             init_trials=3, tag="lloyd@25 greedy k-means++ (50k sample, 3 trials)", no_curve=True))
    return runs


def plan_deep(device: str) -> list[dict]:
    runs = []
    for k in [65536]:
        base = dict(dataset="deep", k=k, device=device)
        for s in [0, 1]:
            runs.append(dict(base, algo="emeans", epochs=5, seed=s, tag="e-means T5"))
            runs.append(dict(base, algo="hmeans", epochs=5, seed=s, hopts=V2, tag="h-means T5"))
            runs.append(dict(base, algo="emeans", epochs=3, seed=s, tag="e-means T3"))
            runs.append(dict(base, algo="hmeans", epochs=3, seed=s, hopts=V2, tag="h-means T3"))
        runs.append(dict(base, algo="lloyd", epochs=25, seed=0, tag="lloyd@25 random"))
    return runs


def plan_scaling(device: str) -> list[dict]:
    """n/k sweep on SIFT at fixed n: where does h-means start to win?"""
    runs = []
    for k in [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]:
        for s in SEEDS3:
            runs.append(dict(dataset="sift", k=k, device=device, algo="emeans", epochs=5, seed=s, tag="e-means T5", no_curve=True))
            runs.append(dict(dataset="sift", k=k, device=device, algo="emeans", epochs=5, seed=s, batch_law="quality", tag="e-means T5 quality", no_curve=True))
            runs.append(dict(dataset="sift", k=k, device=device, algo="hmeans", epochs=5, seed=s, hopts=V2, tag="h-means T5", no_curve=True))
    return runs


def plan_scaling_v3(device: str) -> list[dict]:
    return [dict(r, hopts=V3) for r in plan_scaling(device) if r["algo"] == "hmeans"]


def plan_deep_v3(device: str) -> list[dict]:
    return [dict(r, hopts=V3) for r in plan_deep(device) if r["algo"] == "hmeans"]


def plan_cpu_v3() -> list[dict]:
    return [dict(r, hopts=V3) for r in plan_cpu() if r["algo"] == "hmeans"]


def plan_ablation_v3(device: str) -> list[dict]:
    """Ablations rebased on the final recipe (V3). Variants that do not touch ema_frac differ from
    the V2-based plan only through the 0.4 split."""
    runs = []
    configs = [("sift", 65536), ("sift", 16384), ("syn:gmm-zipf:500000:96:2048", 4096), ("glove", 65536)]
    variants = {
        "h-means (full)": dict(V3),
        "- warm start": dict(V3, warm_start=None),
        "- SSE allocation (uniform children)": dict(V3, warm_start={"factor": 16, "epochs": 0.5, "local_iters": 2, "alloc": "uniform", "init_mass": 1.0}),
        "- SSE donors (mass donors)": dict(V3, donor="usage"),
        "- Hartigan in EMA phase": dict(V3, hartigan_ema=False),
        "- Hartigan tail (EMA only, same passes)": dict(V3, tail=None),
        "- Hartigan rule in tail (Lloyd-rule tail)": dict(V3, tail="incremental"),
        "ema_frac 0.8 (v2 split)": dict(V3, ema_frac=0.8),
        "ema_frac 1.0 (no tail, Hartigan-EMA only)": dict(V3, tail=None, ema_frac=1.0),
        "e-means (all off)": {"warm_start": None, "donor": "usage", "tail": None, "hartigan_ema": False},
        "e-means + Hartigan-EMA only": {"warm_start": None, "donor": "usage", "tail": None, "hartigan_ema": True},
    }
    for ds, k in configs:
        for name, h in variants.items():
            for s in SEEDS5:
                runs.append(dict(dataset=ds, k=k, device=device, algo="hmeans", epochs=5, seed=s, hopts=h, tag=name, no_curve=True))
    return runs


DATASET_N = {"sift": 980000, "glove": 1163514, "nytimes": 228969, "fashion-mnist": 40000, "syn:gmm-zipf:500000:96:2048": 500000, "syn:gmm:500000:96:4096": 500000, "deep": 9970000}
AUTO1 = {"hartigan_ema": True, "ema_frac": 1.0}  # the n/k >= 64 branch of ema_frac="auto" (warm start + Hartigan-EMA, no exact tail)


def plan_auto(device: str) -> list[dict]:
    """h-means rows still missing for the adaptive split: ema_frac 1.0 on the n/k >= 64 configurations
    (main grid, SIFT sweep, Deep-10M). The 0.4 and 0.8 branches are the v3 and v2 plans."""
    runs = []
    for ds, n, ks in MAIN_DATASETS:
        for k in ks:
            if (n or DATASET_N[ds]) / k < 64:
                continue
            base = dict(dataset=ds, n=n, k=k, device=device, algo="hmeans", hopts=AUTO1)
            for s in SEEDS5:
                runs.append(dict(base, epochs=3, seed=s, tag="h-means T3"))
                runs.append(dict(base, epochs=5, seed=s, tag="h-means T5"))
                runs.append(dict(base, epochs=5, seed=s, batch_law="quality", tag="h-means T5 quality"))
    for k in [2048, 4096, 8192]:
        for s in SEEDS3:
            runs.append(dict(dataset="sift", k=k, device=device, algo="hmeans", epochs=5, seed=s, hopts=AUTO1, tag="h-means T5", no_curve=True))
    for s in [0, 1]:
        runs.append(dict(dataset="deep", k=65536, device=device, algo="hmeans", epochs=5, seed=s, hopts=AUTO1, tag="h-means T5"))
        runs.append(dict(dataset="deep", k=65536, device=device, algo="hmeans", epochs=3, seed=s, hopts=AUTO1, tag="h-means T3"))
    return runs


PLANS = {
    "main": lambda dev: plan_main(dev),
    "v2": lambda dev: plan_v2(dev),
    "v3": lambda dev: plan_v3(dev),
    "ablation": lambda dev: plan_ablation(dev),
    "cpu": lambda dev: plan_cpu(),
    "kmeanspp": lambda dev: plan_kmeanspp(dev),
    "deep": lambda dev: plan_deep(dev),
    "scaling": lambda dev: plan_scaling(dev),
    "scaling_v3": lambda dev: plan_scaling_v3(dev),
    "deep_v3": lambda dev: plan_deep_v3(dev),
    "cpu_v3": lambda dev: plan_cpu_v3(),
    "ablation_v3": lambda dev: plan_ablation_v3(dev),
    "auto": lambda dev: plan_auto(dev),
}


def main():
    plan = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else ("cpu" if plan.startswith("cpu") else "mps")
    out = plan
    runs = PLANS[plan](device)
    done = done_keys(out)
    todo = [r for r in runs if run_key(r) not in done]
    print(f"plan {plan}: {len(runs)} runs, {len(done)} done, {len(todo)} to do", flush=True)
    t0 = time.time()
    for i, cfg in enumerate(todo):
        t = time.time()
        proc = subprocess.run(cmd(cfg, out), capture_output=True, text=True)
        tail = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        status = tail[-1][:160] if tail else ("ERROR: " + proc.stderr.strip().splitlines()[-1][:200] if proc.stderr.strip() else "no output")
        print(f"[{i + 1}/{len(todo)} {time.time() - t:6.1f}s total {(time.time() - t0) / 60:6.1f}m] {cfg['tag']} {cfg['dataset']} k={cfg['k']} seed={cfg['seed']}: {status}", flush=True)


if __name__ == "__main__":
    main()
