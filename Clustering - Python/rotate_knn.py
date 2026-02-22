# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 15:24:11 2025

@author: RBU-Kilosort2
"""

# null_knn_pipeline.py
# Build null datasets by per-neuron circular shifts along the bins axis (MATLAB circshift equivalent),
# then construct FAISS kNN affinity graphs via spectral_affinity_faiss, saving one HDF5 per run.

import os
import time
import tempfile
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from scipy.sparse import csr_matrix

# Import your existing FAISS-based builder
from spectral_clustering_faiss import spectral_affinity_faiss

# ---------------------------
# Module-level shared state for workers (memmap & config)
# ---------------------------
_SHARED = {}

def _pool_init_shared(
    xx_npy_path: str,
    shape_T: int,
    shape_N: int,
    shape_B: int,
    save_dir: str,
    # spectral_affinity_faiss controls:
    knn_values: List[int],
    correlation_metric: bool,
    normalize_data: bool,
    knngraphtype: str,
    dendroreorder: bool,
    verbose: bool,
    faiss_threads: Optional[int],
):
    """Initializer for worker processes. Mounts memmap and stores config."""
    # Load source data as read-only memmap to avoid copying
    _SHARED["XX"] = np.load(xx_npy_path, mmap_mode="r")
    _SHARED["T"] = int(shape_T)
    _SHARED["N"] = int(shape_N)
    _SHARED["B"] = int(shape_B)
    _SHARED["save_dir"] = str(save_dir)

    # spectral_affinity_faiss params
    _SHARED["knn_values"] = list(knn_values)
    _SHARED["correlation_metric"] = bool(correlation_metric)
    _SHARED["normalize_data"] = bool(normalize_data)
    _SHARED["knngraphtype"] = str(knngraphtype)
    _SHARED["dendroreorder"] = bool(dendroreorder)
    _SHARED["verbose"] = bool(verbose)

    # Optionally limit FAISS OMP threads inside each worker
    if faiss_threads is not None:
        try:
            import faiss  # local import to avoid top-level dependency in parent
            faiss.omp_set_num_threads(int(faiss_threads))
        except Exception:
            # If FAISS not available or threads cannot be set, silently continue
            pass


def _save_knn_graphs_h5(
    filepath: Path,
    knn_graphs: Dict[int, csr_matrix],
    params: Dict,
    run_index: int,
    integer_offset_row: np.ndarray,
):
    """Save CSR graphs + params for one run to an HDF5 file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        # ---- Meta / params ----
        meta = f.create_group("meta")
        # Preserve scalar-like attributes
        meta.attrs["run_index"] = int(run_index)
        meta.attrs["save_timestamp"] = float(time.time())

        # Store params (robust to lists/numpy scalars)
        pgrp = meta.create_group("params")
        for k, v in params.items():
            try:
                if isinstance(v, (str, bytes)):
                    pgrp.attrs[k] = v
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    pgrp.attrs[k] = v
                elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in v):
                    pgrp.create_dataset(k, data=np.array(v))
                elif isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                    # pack list of strings as a single comma-joined attribute
                    pgrp.attrs[k] = ",".join(v)
                elif isinstance(v, np.ndarray):
                    pgrp.create_dataset(k, data=v)
                else:
                    # Fallback to string attr
                    pgrp.attrs[k] = str(v)
            except Exception:
                # Last resort
                pgrp.attrs[k] = str(v)

        # Also record the integer_offset row used in this null
        pgrp.create_dataset("integer_offset_row", data=integer_offset_row.astype(np.int64))

        # ---- Graphs ----
        ggrp = f.create_group("knn_graphs")
        for k, W in knn_graphs.items():
            kgrp = ggrp.create_group(f"k{k}")
            # Store CSR triplets
            kgrp.create_dataset("data", data=W.data.astype(np.float32))
            kgrp.create_dataset("indices", data=W.indices.astype(np.int32))
            kgrp.create_dataset("indptr", data=W.indptr.astype(np.int32))
            kgrp.attrs["shape"] = W.shape


def _roll_bins_per_neuron(xx_src: np.memmap, offsets: np.ndarray) -> np.ndarray:
    """
    Apply per-neuron circular shifts along the bins axis (axis=2).
    This matches MATLAB circshift(A, K, 3) semantics using numpy.roll.

    offsets: shape (N,), can be negative/positive; values are modulo B.
    """
    T, N, B = _SHARED["T"], _SHARED["N"], _SHARED["B"]
    x_null = np.empty((T, N, B), dtype=xx_src.dtype)

    # Ensure 1D int array length N
    offs = np.asarray(offsets, dtype=np.int64).reshape(-1)
    if offs.shape[0] != N:
        raise ValueError(f"offsets length {offs.shape[0]} != N={N}")

    # circshift equivalence: positive values roll toward higher indices (wrap-around)
    # identical across trials (T), different per neuron (N), along bins axis (B)
    for n in range(N):
        shift = int(offs[n]) % B  # modulo handles large |shift|
        x_null[:, n, :] = np.roll(xx_src[:, n, :], shift=shift, axis=1)
    return x_null


def _worker_make_null_and_knn(args: Tuple[int, np.ndarray, Optional[str]]) -> Tuple[int, str]:
    """
    For one run: build null by per-neuron circshift, call spectral_affinity_faiss, save HDF5.
    Returns (run_index, saved_filepath_str).
    """
    run_index, offsets_row, file_prefix = args

    # 1) Build null dataset via per-neuron circular shifts
    XX_src = _SHARED["XX"]
    Xnull = _roll_bins_per_neuron(XX_src, offsets_row)

    # 2) Build knn graphs from the null dataset (ignore X return)
    knn_graphs, params, _ = spectral_affinity_faiss(
        Xnull,
        knn_values=_SHARED["knn_values"],
        correlation_metric=_SHARED["correlation_metric"],
        normalize_data=_SHARED["normalize_data"],
        knngraphtype=_SHARED["knngraphtype"],
        dendroreorder=_SHARED["dendroreorder"],
        verbose=_SHARED["verbose"],
    )

    # Append the offsets used for this run into params
    params_out = dict(params)
    params_out["integer_offset_note"] = "Per-neuron circular shift (MATLAB circshift equivalent) along bins axis"
    # (the exact vector is stored in HDF5 separately; also record B for clarity)
    params_out["bins_dimension_B"] = int(_SHARED["B"])

    # 3) Save to HDF5 (separate file per run)
    save_dir = Path(_SHARED["save_dir"])
    prefix = file_prefix or "null_knn"
    outpath = save_dir / f"{prefix}_run_{run_index:05d}.h5"
    _save_knn_graphs_h5(outpath, knn_graphs, params_out, run_index, offsets_row)

    return run_index, str(outpath)


def build_null_knn_graphs_parallel(
    XX: np.ndarray,
    integer_offset: np.ndarray,   # shape (R, N) – offsets in bin units (can be ±)
    save_dir: str,
    *,
    # Parallel controls
    n_processes: Optional[int] = None,
    faiss_threads: Optional[int] = None,  # per-process FAISS OMP threads (None = leave default)
    progress_every: int = 5,

    # spectral_affinity_faiss controls (defaults match your function except knn_values):
    knn_values: List[int] = (12, 24, 48),
    correlation_metric: bool = True,
    normalize_data: bool = True,
    knngraphtype: str = "complete",  # {'complete', 'mutual'}
    dendroreorder: bool = False,
    verbose: bool = True,

    file_prefix: Optional[str] = "null_knn",
) -> Dict[int, str]:
    """
    Parallel pipeline:
      For each row in integer_offset (R x N), build a null dataset by per-neuron circshift
      along the bins axis (axis=2), construct FAISS kNN graphs via spectral_affinity_faiss,
      and save {knn_graphs, params} to a separate HDF5 file.

    Parameters
    ----------
    XX : np.ndarray, shape (T, N, B)
        Original population data (trials, neurons, bins).
    integer_offset : np.ndarray, shape (R, N)
        Per-run, per-neuron integer circular shifts in bin units. Positive values move toward
        higher bin indices (wrap-around), matching MATLAB circshift(A, K, 3) with K along dim 3.
    save_dir : str
        Directory to write one HDF5 output per run.
    n_processes : int or None
        Number of worker processes. If None, uses min(16, max(1, mp.cpu_count()-1)).
    faiss_threads : int or None
        If provided, set FAISS OMP threads per worker (helps avoid oversubscription).
    progress_every : int
        Print progress every this many completed runs.
    knn_values, correlation_metric, normalize_data, knngraphtype, dendroreorder, verbose
        Passed through to spectral_affinity_faiss. Defaults mirror that function,
        except knn_values defaults to (12,24,48) here.
    file_prefix : str
        File name prefix for per-run HDF5 files.

    Returns
    -------
    Dict[int, str]
        Mapping from run_index -> path to the saved HDF5 for that run.
    """
    assert XX.ndim == 3, "XX must be (T, N, B)"
    T, N, B = XX.shape

    integer_offset = np.asarray(integer_offset)
    assert integer_offset.ndim == 2 and integer_offset.shape[1] == N, \
        f"integer_offset must be (R, N); got {integer_offset.shape}, N={N}"
    R = integer_offset.shape[0]

    save_dir = str(save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Default process count (match leiden pattern)
    if n_processes is None:
        n_processes = min(16, max(1, mp.cpu_count() - 1))
    if n_processes <= 0:
        raise ValueError("n_processes must be positive")

    # Persist XX to a temporary .npy so workers can attach via memmap (Windows-friendly)
    tmpdir = tempfile.mkdtemp(prefix="null_knn_shared_")
    xx_npy_path = os.path.join(tmpdir, "XX.npy")
    # Save as .npy (read-only in workers)
    np.save(xx_npy_path, XX, allow_pickle=False)

    # Build worker args
    worker_args = [(run_idx, integer_offset[run_idx, :].astype(np.int64), file_prefix) for run_idx in range(R)]

    # Shared initializer config
    initargs = (
        xx_npy_path, T, N, B, save_dir,
        list(knn_values), bool(correlation_metric), bool(normalize_data),
        str(knngraphtype), bool(dendroreorder), bool(verbose),
        faiss_threads,
    )

    saved_paths: dict = {}
    start = time.time()
    try:
        if verbose:
            print(f"[null_knn] Starting R={R} runs with n_processes={n_processes}")
            print(f"[null_knn] Data shape: T={T}, N={N}, B={B}; save_dir='{save_dir}'")

        with Pool(
            processes=n_processes,
            initializer=_pool_init_shared,
            initargs=initargs,
        ) as pool:
            completed = 0
            for run_index, outpath in pool.imap_unordered(_worker_make_null_and_knn, worker_args):
                saved_paths[run_index] = outpath
                completed += 1
                if verbose and (completed % progress_every == 0 or completed == R):
                    elapsed = time.time() - start
                    rate = completed / max(elapsed, 1e-9)
                    print(f"[null_knn] Progress: {completed}/{R} "
                          f"({100.0 * completed / R:.1f}%) | Rate: {rate:.2f} runs/s")

    finally:
        # Clean up temp directory with memmap source
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    if verbose:
        print(f"[null_knn] Completed R={R} runs in {time.time() - start:.1f}s")
    return saved_paths
