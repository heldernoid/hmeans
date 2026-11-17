# NOTES: h-means investigation log

Working notes, in chronological order. Numbers here come from runs executed on
this machine (Apple M3 Ultra, 32 CPU cores, 80-core GPU used through PyTorch
MPS, 512 GB RAM, macOS; faiss-cpu 1.16.0, torch 2.15, numpy 2.5). Every number
in `REPORT.md` is regenerated from `experiments/results/*.jsonl`.

## 0. Environment and sources

* **e-means paper**: arXiv 2611.09501 (Bessa, Mazaré, Szilvasy, Touvron,
  Picard, Douze, Jégou; Meta FAIR, Nov 2026). Read the HTML and PDF (text
  extracted with pdftotext). Code: `faiss/contrib/e_means` shipped in faiss-cpu
  1.16.0 (`NumpyEmeans` on CPU through `faiss.knn`; `Emeans` torch class with
  Triton/CuTe kernels for CUDA only). Project page e-means.github.io.
* **o-means** (Bajaj, Nupur, Garg, "o-means: index the centroids, don't scan
  them"): **not found**. No arXiv entry, repository or project page could be
  located (web search, 2026-11-16). The only trace is the `keo` package's README,
  which cites it as an unreleased paper known from a project page and
  reconstructs it as "e-means + two-level IVF over the centroids (cells of ~128
  centroids, nlist = ceil(k/128) rounded to 16, nprobe = 16 for d < 512, 12
  otherwise, 24 if k > 16384), refreshed every 5 batches in the first epoch and
  rebuilt every half epoch afterwards, with donor->child split pointers".
  Treated here as **unconfirmed**; what I compare against is that reconstruction
  (`keo.algos.omeans`) and, equivalently, e-means's own IVF assignment
  (Appendix M of the paper: 1024 meta-centroids, 4 probed, no gain below 1B
  points on GPU).
* **keo** 0.1.3 (PyPI, github.com/rajveer43/keo, MIT): installs fine. It is a
  benchmark harness on GloVe-100 with reference PyTorch implementations of
  k-means / e-means / o-means and adapters to the official faiss code. Its README
  states "No benchmark conclusions are claimed yet" (phases 7-10, the actual
  benchmark grid, are "not started"), so it provides no numbers to compare
  against. I reuse its o-means description (above) but not its code.
* Related: "Lloyd's k-means is Frank-Wolfe in disguise" (arXiv 2607.25190) and
  power k-means were read for ideas only (see idea log).
* **Datasets** (ANN-benchmarks HDF5): SIFT-128 (1M, image descriptors),
  GloVe-100 (1.18M word embeddings, L2-normalised), NYTimes-256 (290k text
  embeddings, L2-normalised), Fashion-MNIST-784 (60k images), Deep-image-96
  (10M GoogLeNet embeddings, normalised; download in progress). Synthetic
  mixtures as in the paper's Appendix D.2 (Gaussian mixture with sigma_w^2 = 1/3,
  sigma_mu^2 = 2/3, optional Zipf-imbalanced weights, uniform, isotropic
  Gaussian, stretched Gaussian). 20k rows are held out from each training set.
* **Software pitfall**: faiss-cpu and torch each bundle their own `libomp.dylib`;
  importing both in one process aborts ("OMP: Error #15") or segfaults with
  `KMP_DUPLICATE_LIB_OK`. Fix used: symlink `faiss/.dylibs/libomp.dylib` to
  torch's copy (`setup.sh`). After that faiss.knn, NumpyEmeans, torch CPU/MPS
  and scikit-learn coexist and give identical results.

## 1. Phase 1: understanding and reproduction

### What e-means is (from code + paper, verified)
Per cluster an EMA mass `N_j` and position `c_j = S_j / N_j`. Per batch:
`N_j <- (1-a) N_j + a n_j`, `c_j <- c_j + (a n_j / N_j_new)(mean_j - c_j)`.
Step `a` follows a cosine schedule from `a0` to `0.1 a0` over all updates.
Batch size `B` and `a0` come from power laws in (k, n, T); the shipped
coefficients differ from the paper's Table 2 (the code says so). Revival every 5
updates: clusters with `N_j < 0.1 mean(N)` are rebuilt by splitting donors
sampled proportionally to `N_j` (perturbation 1e-4 along alternating
coordinates). Finalisation: centroid = plain mean of the points assigned during
the last pass. Initial mass 2^-16 so the random init carries no weight.

Important consequence of the laws at the paper's n/k ~ 60 (SIFT 1M, k = 16384,
T = 5): B = 65536, a0 = 0.345, so each update sees ~4 points per cluster and
moves the centroid ~35% toward their mean. e-means is "online k-means with
heavy momentum", and at T = 25 the law gives a0 = 1, B = 262144, i.e. Lloyd
steps on random quarters of the data.

### Reproduction (see `experiments/results/repro_*.jsonl`)
* My torch reimplementation (`hmeans/emeans_ref.py`, same recipe constants,
  same RNG streams for init/permutations/splits) matches the official
  `NumpyEmeans` to within seed noise: GMM n = 200k, d = 64, k = 1024, T = 3:
  official 23.931 / 23.971 (seeds 0/1), ref 23.931 / 24.018 (MPS), 23.976 /
  24.024 (CPU). Split counts agree (253/252/255). Residual differences come from
  float32 argmin ties in different BLAS. My torch Lloyd equals faiss Lloyd to 6
  digits from the same init (24.824890 vs 24.824825).
* SIFT 1M, k = 16384, seed 0 (MPS): Lloyd@25 random init train MSE 38238,
  test 41028; e-means T = 3: 38113 / 40940; T = 5: 37829 / 40806. e-means beats
  Lloyd@25 in 3 epochs by 0.33% train / 0.22% test. The paper's Table 5 reports
  -0.35% test for e-means@3 vs Flash-KMeans@25 on SIFT 1M; consistent.
* Synthetic GMM (n = 500k, d = 96, 4096 components, k = 8192): e-means T = 3
  is 3.6% below Lloyd@25. Reproduced: the mini-batch noise finds much better
  optima than Lloyd on well-separated mixtures.
* **Discrepancy noted**: the paper's "~10x fewer epochs" is relative to
  Lloyd@25, but e-means itself keeps improving well past 5 epochs: SIFT T = 10
  gives 37599 (-0.6%) and T = 25 gives 37427 (-1.06% vs T = 5). "Converged in 1-3
  epochs" means "matches Lloyd@25", not "converged".
* **Hyper-parameter sensitivity**: the shipped `throughput` batch law is a
  hardware knee, not a quality optimum. On SIFT the `quality` law (B 8x smaller)
  gives 38012 vs 38113 at T = 3 (-0.27%) and 37757 vs 37829 at T = 5 (-0.19%)
  at ~2x the wall-clock on MPS.

### Where e-means is weak (measured)
1. **Distance computations**: every pass is a full n x k scan; T = 5 costs 5
   n k distances. All of e-means's gains over Lloyd are in epochs, none in the
   per-epoch cost (the paper's IVF only pays off at 1B+ points on GPU).
2. **The convergence tail is not the problem**: 20 exact Lloyd iterations after
   e-means T = 5 on SIFT gain only 0.1% (37829 -> 37789). The remaining gap to
   longer runs (1%) is in the stochastic optimisation itself, i.e. in the number
   of useful updates, not in polishing.
3. **Random init is deliberately erased**: the first update moves each centroid
   ~100% to its first batch mean (mass 2^-16), so good seeds are wasted; the
   paper found k-means++ init helps only 0.1-0.2% at great cost.
4. **Revival donors are chosen by mass**, not by loss; a cheap SSE-proportional
   donor rule helps slightly (below).
5. On the paper's own account: laws fitted on d >= 96 embeddings, unreliable at
   low d and tiny n/k; test MSE can rise in late epochs at high d (overfitting).

## 2. Phase 2: idea log (each entry: idea, quick test, outcome)

Test bed unless stated: GMM n = 500k, d = 96, 4096 components, k = 8192, MPS,
3 seeds, train MSE; e-means T = 2 / 3 / 5 / 10 = 31.239 / 31.090 / 31.037 / 31.006.

* **Growing batch size** (B/8 -> B during the first pass, step law per batch):
  31.84 at T = 2 vs 31.24. Worse; dropped.
* **SSE-proportional revival donors**: 31.151 at T = 2 (-0.3%), 31.076 at T = 3
  (-0.05%). Free; kept as `donor="sse"`. On SIFT T = 3: -0.02% (noise level).
* **Hierarchical warm start** (k/16 coarse e-means for 0.5 pass, split each
  coarse centroid into 16 children by 2 local Lloyd iterations, cost ~0.1 pass):
  31.14 at T = 2 (-0.3%), 31.07-31.09 at T = 3; SIFT T = 3: 38047 vs 38113
  (-0.17%). Same gain as the paper reports for k-means++ init at 1/1000 of the
  cost. Giving the children initial EMA mass did not matter. Kept.
* **Exact assignment with per-point cached candidates + movement bounds**
  (Hamerly-style, adapted to moving centroids via cumulative path lengths and a
  "hot" set of the most-moved centroids): provably exact (losses identical to
  e-means) but certifies only 0.1% of points during training and ~50% during
  polishing: with ~4 points per cluster per batch the centroids move more per
  step than typical margins. Dropped for training; polishing gain too small to
  matter (tail is 0.1%).
* **Approximate cached candidates (stale per-point lists)**: 2.0 pass-equivalents
  for any T, but train MSE +0.5% (31.19-31.21) while test MSE improved
  (32.21 vs 32.29). Same effect the paper reports for IVF assignment: staleness
  acts as a regulariser and hurts the training objective. Dropped.
* **Graph assignment** (k-NN graph over centroids, 1-2 greedy hops from the
  point's previous label, NN-descent refresh each step, exact rebuild every 5
  steps, fallback to a full scan when d2 > 1.5 x cluster mean d2):
  on the GMM: miss rate 0.02%, 31.076 at T = 3 (2.2 pass-eq) and 31.040 at T = 5
  (2.4 pass-eq) vs e-means 31.090 (3) and 31.037 (5): same loss, half the
  distance computations. **On SIFT it fails**: 1-hop hit rate from the exact
  graph is only 72% (m = 8) to 89% (m = 64) between passes 1 and 2 and the
  misses are near-ties (median 1.05x the true distance) that no radius test
  separates; miss rate stayed ~21% and train MSE was 1.1% worse. Real
  embeddings have many near-equidistant centroids; clean mixtures do not.
* **IVF first pass** (children of the 4-8 nearest warm-start parents): 7-12%
  misses that then persist in the graph phase; worse loss. Dropped.
* **Lloyd polishing after e-means**: -0.05 to -0.16% at T = 2-3, -0.1% max on
  SIFT after 20 iterations. Cheap only with the bounded assigner. Minor.

(continued below as experiments progress)

### Idea log, continued (2026-11-16 afternoon)

* **Incremental exact-mean tail** (switch from EMA to block-coordinate descent with exact
  sums over all points' current labels for the last passes): SIFT k = 16384, 3 EMA + 2
  incremental = 38003 vs e-means T = 5 = 37829. Worse. Dropped.
* **Hartigan tail** (same, with the n/(n-1), n/(n+1) correction; first version let points
  join empty clusters for free and collapsed - fixed): 3 + 2 = 37834, 4 + 1 = 37812,
  5 + 5 = 37606 vs e-means 37829 (T = 5) and 37599 (T = 10). Equal, not better, at
  n/k = 60. Kept only as an option to test at small n/k.
* **PCA filter-and-refine assignment** (lower bounds in d' PCA dimensions, exact re-rank
  of the 32-64 smallest, certified fallback): exact (100% label agreement) but SIFT needs
  d' = 48-64 of 128 for a < 10% fallback rate, so MACs only halve, and the gather-based
  re-rank is 6x slower than the plain matmul on MPS and 1.5x slower on CPU. Dropped.
  (This is the pruning family the paper's SuperKMeans baseline belongs to; they also
  found it does not pay on GPUs.)
* **Batch-size sweep (e-means, SIFT, T = 5, seed 0)**: the shipped throughput law is a
  hardware knee fitted on H100s. Here smaller batches are nearly free and clearly better:
  k = 16384: B = 262144 / 65536 (law) / 16384 / 4096 / 1024 -> 38037 / 37829 / 37761 /
  37752 / 37766, MPS time 2.2 / 2.5 / 3.3 / 6.6 / 19.6 s;
  k = 65536: 29870 / 29606 / 29482 / 29405 / 29371, time 14.3 / 14.6 / 15.9 / 20.3 / 37.3 s.
  So at large k the `quality` law (8x smaller) gives -0.4 to -1.0% for +10% time.
  Consequence for fairness: every h-means claim below is checked against e-means with
  the quality law too.
* **Warm-start allocation matters**: splitting every coarse cluster into the same
  number of children breaks on imbalanced data (Zipf GMM: 32.96 vs e-means 31.87).
  Allocating children proportionally to the coarse cluster's SSE (largest-remainder
  apportionment, >= 1 and <= points) fixes it and more: 30.99 (-2.8% vs e-means T = 3,
  -2.1% vs T = 5, -4.9% vs Lloyd@25). Count-proportional: 32.27. On SIFT k = 131072 the
  SSE allocation gives 23767 vs 24268 for e-means at T = 3 (-2.1%) and vs 23927 at T = 5.
  Insensitive to its other knobs (0.5 vs 1 coarse pass, 2 vs 5 local iterations, one or
  two levels, factor 16 vs 64: all within 0.05%); a schedule offset or a large initial
  mass (4x) hurt.

### Large-k results that define h-means (seed 0, MPS, SIFT 1M; see experiments/results/largek_explore.jsonl)

Quality batch law for both methods. "T3+hart2" = 3 EMA passes then 2 exact Hartigan passes
(5 passes of n·k distance computations in total, plus ~0.1 pass of warm start).

| k (n/k) | e-means T3 | e-means T5 | h warm+sse T3 | h warm+sse T5 | h T3 + hart2 | h T2 + hart1 (3 passes) |
|---|---|---|---|---|---|---|
| 65,536 (15) | 29850 | 29482 | 29509 | 29250 | **28866** | 29186 |
| 131,072 (7.5) | 24042 | 23688 | 23498 | 23236 | **22416** | - |

* The Hartigan tail is the big lever at small n/k: -2.1% (k = 65k) and -5.4% (k = 131k) vs
  e-means at the same 5 passes; held-out MSE improves too (36657 vs 36778; 34271 vs 34578).
* Split of the pass budget barely matters: at k = 65k, 1+4 / 2+3 / 3+2 give 28831 / 28830 / 28866.
* The Hartigan correction, not the exact means, is what matters: a Lloyd-rule incremental tail
  gives 29463 (+2.1% vs Hartigan). Without warm start the Hartigan tail gives 29086 (+0.8%).
* Implementation: Hartigan's join costs g_j ||x - c_j||^2 with g_j = N_j/(N_j+1) are one matmul of
  [||x||^2, x, 1] against [g_j, -2 g_j c_j, g_j ||c_j||^2]; the stay cost N_a/(N_a-1) ||x - c_a||^2
  is scattered into the current cluster's column. Same cost as an exact pass (d+2 instead of d).
  Points cannot join empty clusters through the rule (that collapsed everything in a first
  version); empties are re-seeded every 5 updates at the farthest point of a mass-sampled donor.
* GMM-Zipf (k = 4096): h-means T5 = 30.86 vs e-means T5 = 31.66 (throughput) / 32.00 (quality law),
  Lloyd@25 = 32.57.
* At n/k = 60 (SIFT k = 16384) the Hartigan tail is neutral (37834 vs 37829) and only the warm
  start remains (-0.3%). h-means is a large-k / small-clusters method.
* CPU (torch, 24 threads): e-means ref T3 15.4 s, T5 23.0 s; h-means T3 18.2 s; official faiss
  NumpyEmeans T3 60.8 s (faiss.knn runs at ~0.3 TFLOP/s on this machine, torch matmul at ~1).

## 3. Phase 3: main suite (v1 recipe), 2026-11-16 16:00

540 runs, zero failures (`experiments/results/main.jsonl`; `python experiments/analyze.py`).
h-means v1 = warm start + SSE donors + EMA for 40% of the passes + Hartigan tail. Train MSE of
h-means at 5 passes relative to e-means at 5 passes with the better of its two batch laws, mean of
5 seeds (h-means wins on every configuration and every seed):

| dataset | k | n/k | h-means T5 vs best e-means T5 | Lloyd@25 vs e-means T5 | Sculley vs e-means T5 |
|---|---|---|---|---|---|
| Fashion-MNIST | 512 / 2048 | 78 / 19 | -0.23% / -1.89% | +0.95% / +2.51% | +2.6% / +5.3% |
| GloVe-100 | 16k / 65k / 131k | 71 / 17 / 8 | -0.12% / -1.97% / -4.57% | +0.9% / +2.2% / +2.1% | +9% / +12% / +13% |
| NYTimes-256 | 4k / 16k / 32k | 55 / 13 / 6 | -0.22% / -2.88% / -6.96% | +0.7% / +0.3% / -1.4% | +13% / +14% / +13% |
| SIFT-128 | 16k / 65k / 131k | 59 / 14 / 7 | +0.00% / -2.10% / -5.52% | +1.1% / +2.9% / +4.4% | +5% / +7% / +11% |
| GMM-Zipf | 4k / 16k | 122 / 30 | -2.44% / -0.97% | +3.0% / +1.2% | +19% / +15% |
| GMM | 8k / 32k | 61 / 15 | -0.23% / -1.79% | +0.8% / +1.8% | +1.4% / +7% |

* The gain is a function of n/k: ~0.2% at 60 points per cluster, ~2% at 15, 5-7% at 6-8.
* At NYTimes k = 32768 (n/k = 6, below the paper's fitted range) Lloyd@25 already beats e-means
  by 1.4%; h-means beats both by a wide margin.
* **2-D playground** (demo/playground.html, JS port cross-checked against the Python reference on
  the same distribution within 4%): 3 Gaussians n = 105k, k = 5000: Lloyd 0.001028, e-means 25
  passes 0.000922, h-means 5 passes 0.000627, 25 passes 0.000598 (Python). The warm start alone
  reaches e-means's 25-pass loss after ~0.26 pass-equivalents. Low-d data is the regime the
  e-means authors themselves flag as weak (revival switched off at d <= 2, laws not fitted).
* **v2 recipe** (Hartigan-corrected assignment also during the EMA passes, with the EMA mass times
  n/B as the cluster-size estimate; ema_frac 0.8): side test on CPU, SIFT, 2 seeds, quality law,
  5 passes: k = 16384: 37631 vs 37764 (v1) and 37757 (e-means quality); k = 65536: 28691 vs
  28859 (v1) and 29482 (e-means quality). Adopted as the final recipe; v2 runs, ablations, the
  k-sweep, Deep-10M and a full-data greedy k-means++ baseline are queued.

## 4. Recipe decision (v1 / v2 / v3), 2026-11-16 evening

Three recipes on the full 15-configuration grid, 5 seeds each (`main.jsonl` = v1, `v2.jsonl`,
`v3.jsonl`). Train MSE of h-means T5 vs the better e-means T5 law:

| config (n/k) | v1 | v2 (Hartigan-EMA, ema_frac 0.8) | v3 (Hartigan-EMA, ema_frac 0.4) |
|---|---|---|---|
| Fashion 512 (78) / 2048 (20) | -0.16 / -2.00 | -0.46 / -2.18 | -0.19 / -2.19 |
| GloVe 16k (71) / 65k (18) / 131k (9) | -0.04 / -1.87 / -4.35 | -0.16 / -1.42 / -2.06 | -0.04 / -2.02 / -4.44 |
| NYTimes 4k (56) / 16k (14) / 32k (7) | -0.22 / -2.95 / -6.81 | -0.46 / -2.14 / -3.95 | -0.32 / -3.15 / -6.70 |
| SIFT 16k (60) / 65k (15) / 131k (7) | +0.08 / -1.90 / -5.37 | -0.24 / -2.34 / -4.14 | +0.01 / -2.26 / -5.97 |
| GMM-Zipf 4k (122) / 16k (31) | -1.89 / -0.97 | -2.19 / -0.89 | -1.94 / -1.01 |
| GMM 8k (61) / 32k (15) | -0.23 / -1.57 | -0.18 / -1.20 | -0.23 / -1.62 |

* v2 was adopted prematurely from a 2-seed CPU side test at n/k = 15-60; on the full grid it
  gives up half of v1's gain at n/k <= 9 (more EMA passes, fewer exact Hartigan passes).
* v3 = Hartigan correction in the EMA passes + the v1 split (2 EMA + 3 Hartigan passes at
  T = 5). Best or within 0.2% of the best recipe on every configuration. **Final recipe**;
  `HMeans` defaults changed to `hartigan_ema=True` (commit after this entry).
* Held-out (v3): better than best e-means on 9/15, within 0.05% on 4, worse on GloVe 16k
  (+0.26%) and GMM 32k (+0.70%). The balanced synthetic mixture at n/k = 15 is where exact
  passes overfit the training rows.
* The k-sweep, Deep-10M, CPU and ablation plans had been run with v2; their h-means rows were
  re-run with v3 (`scaling_v3.jsonl`, `deep_v3.jsonl`, `cpu_v3.jsonl`, `ablation_v3.jsonl`).
  The v2-based ablation is kept in `experiments/tables/ablation_v2.md`.
* Full-data greedy k-means++ (SIFT k = 16384, 1 seed): seeding 419 s, Lloyd@25 after it reaches
  38107 train (e-means T5 37830, h-means T5 37764). The 50k-sample version reaches 38328.

### Final-recipe re-runs (v3), 2026-11-16 20:00-21:00

* SIFT k-sweep (`scaling_v3.jsonl`): +0.25% vs e-means quality at n/k 120-480 (v2 was +0.07%),
  +0.02% at 60, -0.69% at 30, -2.28% at 15, -5.98% at 7, -14.73% at 4 (v2: -7.38%). v3 buys
  twice the small-n/k gain for a quarter percent at large n/k. Wall-clock equal at k >= 65k,
  3-4x slower at k <= 8k (fixed costs: warm start, Hartigan bookkeeping, sub-3 s problems).
* Deep-10M k = 65536 (`deep_v3.jsonl`): tie with e-means T5 (+0.00% train, -0.02% held-out,
  152 s vs 151 s). v2 was -0.23% / 141 s. The 10M-row, n/k = 152 regime is not where h-means helps.
* Decision stands: v3 is the recipe for small clusters (the mission's "beat e-means" regime); an
  adaptive split (more EMA passes when n/k is large, i.e. v2's 0.8 above n/k ~ 100) would recover
  the large-n/k quarter percent and is the obvious next step; not done, not claimed.

## 5. Library, NumPy backend, adaptive pass split (2026-11-16 evening)

* `hmeans` is now a pip-installable package (`pip install .` = NumPy only; `.[torch]` adds the torch
  backend for CPU/MPS/CUDA); `HMeans`/`EMeans` are scikit-learn style estimators; version from git
  tags (setuptools-scm). `HMeansNumpy` mirrors `HMeansCore` step for step (same seeds and batch
  orders): with `hartigan_ema=False` the two backends give identical centroids (checked: e-means
  path and warm start and tail match to 1e-5); with Hartigan-corrected EMA assignment the argmin
  tie-breaking differs in float order and the runs diverge chaotically on near-degenerate data
  (k = 256 equal-size separable mixture: 48.5 vs 52.9 for the same seed) while agreeing to 0.01% on
  ordinary data (same mixture, k = 2048). NumPy CPU is faster than torch CPU on small problems
  (less per-op overhead), similar at k >= 2048.
* **Exact tail at large n/k hurts on separable data.** Toy: 256 equal Gaussians in 32-d, n = 60k,
  k = 256 (n/k = 234), 3 seeds, quality law: e-means 50.9; h-means ema_frac 0.4 / 0.8 / 1.0 =
  43.0 / 42.0 / 41.0; warm start + plain EMA 41.1; Lloyd@25 61.2. The warm start is the whole gain
  there and the exact passes (2 EMA + 3 exact) lose it: 3 exact passes cannot move a centroid
  across clusters, 3 more EMA passes can. With k = 2048 (n/k = 29) the order flips back: 0.4 =
  0.8 = 27.02 < 1.0 = 27.14 = e-means 27.15.
* Therefore `ema_frac="auto"` = 0.4 for n/k < 24, 0.8 for n/k < 64, 1.0 above, from: the main grid
  (v3 better than v2 at n/k <= 18, equal at 20, v2 better at 30-60), the ablation rows "ema_frac
  1.0" (= v2 within 0.01% at n/k = 60 and 122), the SIFT sweep (v2 +0.07% vs v3 +0.25% at n/k >=
  120) and the toy above. The 1.0 branch is being measured on the three n/k >= 64 grid configs,
  the sweep at k <= 8192 and Deep-10M (`auto.jsonl`); the thresholds were chosen after seeing the
  grid, which stays a stated limitation.
* A 6000-point, 64-component, 16-d toy disagrees (0.4 best, 1.0 worse by 9%, Hartigan-EMA itself
  hurts there, 3 seeds, huge seed variance): n = 6000 is far outside the regime the laws were
  fitted for; not used for the rule, recorded here.

### Adaptive split measured (`auto.jsonl`, 58 runs, 2026-11-16 21:30)

EMA fraction 1.0 on the n/k >= 64 configurations, vs best e-means T5: Fashion 512 (78) -0.35%
(fixed 0.4: -0.19%), GloVe 16k (71) -0.03% (-0.04%), GMM-Zipf 4k (122) -3.71% (-1.94%); SIFT sweep
k = 2048 / 4096 / 8192 (n/k 479 / 239 / 120): +0.03 / +0.02 / -0.03% (fixed 0.4: +0.25% each);
Deep-10M k = 65536 (152): -0.20% in 135.5 s vs 151 s (fixed 0.4: +0.00% in 152 s; 0.8: -0.23% in
141 s). With the rule applied to the logged runs, h-means is below the best e-means on all 15 main
configurations and Deep-10M, every seed; held-out lower on 11/15. The rule is a heuristic in n/k
only; thresholds chosen on the reported grid (stated in REPORT.md §5).
