#!/bin/bash
# Reproducible environment for the h-means experiments (macOS / Apple Silicon tested;
# Linux works the same minus the libomp symlink).
set -e
cd "$(dirname "$0")"
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install numpy scipy scikit-learn matplotlib pandas torch faiss-cpu psutil tqdm h5py
# faiss-cpu and torch each ship a libomp.dylib; loading both aborts the process on macOS.
# Point faiss at torch's copy (both are LLVM OpenMP 5.0).
S=$(python -c "import site; print(site.getsitepackages()[0])")
if [ -f "$S/faiss/.dylibs/libomp.dylib" ] && [ -f "$S/torch/lib/libomp.dylib" ]; then
  mv -n "$S/faiss/.dylibs/libomp.dylib" "$S/faiss/.dylibs/libomp.dylib.orig" || true
  ln -sf "$S/torch/lib/libomp.dylib" "$S/faiss/.dylibs/libomp.dylib"
fi
# datasets (ANN-benchmarks)
mkdir -p data
for f in sift-128-euclidean fashion-mnist-784-euclidean glove-100-angular nytimes-256-angular; do
  [ -f data/$f.hdf5 ] || curl -L -o data/$f.hdf5 https://ann-benchmarks.com/$f.hdf5
done
echo "optional (3.6 GB): curl -L -o data/deep-image-96-angular.hdf5 https://ann-benchmarks.com/deep-image-96-angular.hdf5"
python -c "import torch, faiss, numpy; from faiss.contrib.e_means import NumpyEmeans; print('ok: torch', torch.__version__, 'faiss', faiss.__version__, 'mps', torch.backends.mps.is_available())"
