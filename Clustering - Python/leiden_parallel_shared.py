# -*- coding: utf-8 -*-
"""
Leiden Consistency Analysis - Parallel Implementation (Shared/Memmap version)
Optimized to avoid pickling large edge lists and to support incremental HDF5 checkpointing.

Author: Jon Rueckemann
"""

import warnings
import time
import os
import shutil
import tempfile
from typing import Dict, Optional, Tuple, List, Literal
from multiprocessing import Pool, Value, Lock
import multiprocessing as mp

import numpy as np
import scipy.sparse as sp
import igraph as ig
import leidenalg
import h5py


# ---------------------------
# Globals for progress display
# ---------------------------
_completed_runs = None
_total_runs = None
_start_time = None
_progress_lock = None
_h5_lock = None

# ---------------------------
# Globals populated by pool initializer
# ---------------------------
_SHARED: Dict[str, object] = {}


def leiden_consistency_analysis(
    affinity_matrix: sp.csr_matrix,
    resolution: float,
    *,
    quality_function: Literal["rb", "modularity", "cpm"] = "rb",
    n_runs: int = 1000,
    n_processes: Optional[int] = None,
    base_seed: int = 42,
    savepath: Optional[str] = None,
    checkpoint: bool = True,
    progress_every: int = 50,
    progress_every_seconds: Optional[float] = None,
    add_edges_chunk_size: int = 200_000,
    verbose: bool = True,
    parallel: Optional[bool] = None,   
) -> Dict[str, np.ndarray]:
    """
    Run the Leiden algorithm many times on the same graph/resolution to study consistency.
    """
    _validate_affinity_matrix(affinity_matrix)

    if n_processes is None:
        n_processes = min(16, max(1, mp.cpu_count() - 1))
    if n_processes <= 0:
        raise ValueError("n_processes must be positive")

    n_nodes = affinity_matrix.shape[0]

    if verbose:
        print("Leiden Consistency Analysis (memmap/shared)")
        print(f"Graph: {n_nodes:,} nodes, {affinity_matrix.nnz:,} nonzeros")
        print(f"Quality: {quality_function} | Resolution: {resolution}")
        print(f"Runs: {n_runs:,} | Processes: {n_processes} | Base seed: {base_seed}")
        print("-" * 60)

    # Prepare COO upper triangle arrays (unchanged)
    rows, cols, wts = _prepare_matrix_arrays(affinity_matrix)

    # Distribute runs across processes (used by both modes for seeding order)
    run_batches = _distribute_runs(n_runs, max(1, n_processes), base_seed)

    # Init progress globals (unchanged)
    _completed_runs = Value("i", 0)
    _total_runs = Value("i", n_runs)
    _progress_lock = Lock()
    _h5_lock = Lock()
    _start_time = time.time()

    # Pre-allocate return arrays in parent (unchanged)
    labels_all = np.zeros((n_runs, n_nodes), dtype=np.int32)
    quality_all = np.zeros(n_runs, dtype=np.float64)
    ncom_all = np.zeros(n_runs, dtype=np.int32)
    seeds_all = np.zeros(n_runs, dtype=np.int32)

    # HDF5 checkpoint pre-allocation (unchanged)
    if savepath is not None and os.path.isdir(savepath):
        # If a directory was passed, construct a default filename (unchanged)
        filetoken = "leiden_parallel.h5"
        savepath = os.path.join(savepath, filetoken)
    h5 = None
    if savepath is not None and checkpoint:
        if verbose:
            print(f"Pre-allocating HDF5 datasets at {savepath} ...")
        h5 = h5py.File(savepath, "w")
        _init_h5(h5, n_runs, n_nodes, resolution, base_seed, quality_function, normalized=True)

    # ---------- NEW: SERIAL FAST-PATH (no multiprocessing) ----------
    # Trigger rules:
    #   - parallel is explicitly False, OR
    #   - n_processes == 1 (caller requests serial)
    if (parallel is False) or (n_processes == 1):
        # Build igraph once from local arrays (mirror _build_graph_from_shared, but no memmap)
        g = ig.Graph(n=int(n_nodes), directed=False)
        m = rows.shape[0]
        if m != cols.shape[0] or m != wts.shape[0]:
            raise ValueError("rows, cols, weights must have the same length")
        edge_start = 0
        while edge_start < m:
            edge_end = min(edge_start + add_edges_chunk_size, m)
            r_chunk = rows[edge_start:edge_end]
            c_chunk = cols[edge_start:edge_end]
            wt_chunk = wts[edge_start:edge_end].astype(float).tolist()
            e_chunk = list(zip(r_chunk.tolist(), c_chunk.tolist()))
            g.add_edges(e_chunk, attributes={"weight": wt_chunk})
            edge_start = edge_end
        total_weight = float(np.sum(wts))

        # Serial loop over all seeds in deterministic order
        completed = 0
        last_print = 0.0
        for run_idx in range(n_runs):
            seed = base_seed + run_idx
            opt = leidenalg.Optimiser()
            opt.set_rng_seed(int(seed))

            partition = _make_partition(g, str(quality_function), float(resolution))
            opt.optimise_partition(partition, n_iterations=-1)

            labels_all[run_idx] = np.asarray(partition.membership, dtype=np.int32)
            raw_q = float(partition.quality())
            quality_all[run_idx] = raw_q / (total_weight * 2) if total_weight > 0 else raw_q
            ncom_all[run_idx] = int(len(set(partition.membership)))
            seeds_all[run_idx] = int(seed)

            # Checkpoint: write this run immediately (uses same writer)
            if h5 is not None:
                _write_h5_chunk(
                    h5,
                    np.array([run_idx], dtype=np.int32),
                    {
                        "labels": labels_all[run_idx:run_idx+1],
                        "quality": quality_all[run_idx:run_idx+1],
                        "n_communities": ncom_all[run_idx:run_idx+1],
                        "seeds": seeds_all[run_idx:run_idx+1],
                        "run_indices": np.array([run_idx], dtype=np.int32),
                    },
                )

            # Parent progress (same cadence as parallel path)
            if verbose:
                completed += 1
                now = time.time()
                count_trigger = (completed % progress_every == 0) or (completed == n_runs)
                time_trigger = (progress_every_seconds is not None) and (now - last_print >= progress_every_seconds)
                if count_trigger or time_trigger:
                    elapsed = now - _start_time
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    remaining = (n_runs - completed)
                    eta = remaining / rate if rate > 0 else float("inf")
                    print(
                        f"Progress: {completed}/{n_runs} ({100*completed/n_runs:.1f}%) | "
                        f"Rate: {rate:.2f} runs/s | ETA: {eta/60:.1f} min"
                    )
                    last_print = now

        # Finalize HDF5 (unchanged)
        if savepath is not None:
            if h5 is None:  # no checkpointing: write once now
                if verbose:
                    print(f"Saving results to {savepath}")
                with h5py.File(savepath, "w") as f:
                    _init_h5(f, n_runs, n_nodes, resolution, base_seed, quality_function, normalized=True)
                    f["results/labels"][...] = labels_all
                    f["results/quality"][...] = quality_all
                    f["results/n_communities"][...] = ncom_all
                    f["results/seeds"][...] = seeds_all
            else:
                if verbose:
                    print("Finalizing HDF5 file ...")
                h5.flush()
                h5.close()

        if verbose:
            elapsed = time.time() - _start_time
            print(f"\nCompleted in {elapsed:.1f}s")
            print(
                f"Communities: {float(ncom_all.mean()):.2f} ± {float(ncom_all.std()):.2f} "
                f"(min={ncom_all.min()}, max={ncom_all.max()})"
            )

        return {
            "labels": labels_all,
            "quality": quality_all,
            "n_communities": ncom_all,
            "seeds": seeds_all,
        }
   

    #PARALLELIZATION CODE ONLY EXECUTES IF SERIAL DOES NOT (due to return)
    # Use a temp directory to store .npy for memory-mapped arrays (shared, read-only)
    tmpdir = tempfile.mkdtemp(prefix="leiden_shared_")
    try:
        rows_path = os.path.join(tmpdir, "rows.npy")
        cols_path = os.path.join(tmpdir, "cols.npy")
        wts_path  = os.path.join(tmpdir, "weights.npy")

        # Save as .npy once; workers will mmap read-only
        np.save(rows_path, rows, allow_pickle=False)
        np.save(cols_path, cols, allow_pickle=False)
        np.save(wts_path, wts, allow_pickle=False)

        # (… the rest of your original Pool-based implementation, unchanged …)
        # Distribute runs across processes
        run_batches = _distribute_runs(n_runs, n_processes, base_seed)

        if verbose:
            print("Starting parallel Leiden optimizations...")

        worker_args = [(batch, idx) for idx, batch in enumerate(run_batches)]
        with Pool(
            processes=n_processes,
            initializer=_pool_init_mmap,
            initargs=(rows_path, cols_path, wts_path, n_nodes, add_edges_chunk_size,
                      quality_function, resolution, progress_every, progress_every_seconds, verbose),
        ) as pool:
            completed = 0
            last_print = 0.0
            for wres in pool.imap_unordered(leiden_worker, worker_args):
                ridx = wres["run_indices"]
                labels_all[ridx] = wres["labels"]
                quality_all[ridx] = wres["quality"]
                ncom_all[ridx] = wres["n_communities"]
                seeds_all[ridx] = wres["seeds"]

                if h5 is not None:
                    with _h5_lock:
                        _write_h5_chunk(h5, ridx, wres)

                if verbose:
                    completed += len(ridx)
                    now = time.time()
                    count_trigger = (completed % progress_every == 0) or (completed == n_runs)
                    time_trigger = (progress_every_seconds is not None) and (now - last_print >= progress_every_seconds)
                    if count_trigger or time_trigger:
                        elapsed = now - _start_time
                        rate = completed / elapsed if elapsed > 0 else 0.0
                        remaining = (n_runs - completed)
                        eta = remaining / rate if rate > 0 else float("inf")
                        print(
                            f"Progress: {completed}/{n_runs} ({100*completed/n_runs:.1f}%) | "
                            f"Rate: {rate:.2f} runs/s | ETA: {eta/60:.1f} min"
                        )
                        last_print = now

        # Finalize HDF5 (unchanged)
        if savepath is not None:
            if h5 is None:
                if verbose:
                    print(f"Saving results to {savepath}")
                with h5py.File(savepath, "w") as f:
                    _init_h5(f, n_runs, n_nodes, resolution, base_seed, quality_function, normalized=True)
                    f["results/labels"][...] = labels_all
                    f["results/quality"][...] = quality_all
                    f["results/n_communities"][...] = ncom_all
                    f["results/seeds"][...] = seeds_all
            else:
                if verbose:
                    print("Finalizing HDF5 file ...")
                h5.flush()
                h5.close()

        if verbose:
            elapsed = time.time() - _start_time
            print(f"\nCompleted in {elapsed:.1f}s")
            print(
                f"Communities: {float(ncom_all.mean()):.2f} ± {float(ncom_all.std()):.2f} "
                f"(min={ncom_all.min()}, max={ncom_all.max()})"
            )

        return {
            "labels": labels_all,
            "quality": quality_all,
            "n_communities": ncom_all,
            "seeds": seeds_all,
        }

    finally:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


# ---------------------------
# Worker path
# ---------------------------

def _pool_init_mmap(
    rows_path: str,
    cols_path: str,
    wts_path: str,
    n_nodes: int,
    add_edges_chunk_size: int,
    quality_function: str,
    resolution: float,
    progress_every: int,
    progress_every_seconds: Optional[float],
    verbose: bool,
):
    """Initializer: memory-map arrays and stash config in a module-global dict."""
    _SHARED["rows"] = np.load(rows_path, mmap_mode="r")
    _SHARED["cols"] = np.load(cols_path, mmap_mode="r")
    _SHARED["wts"] = np.load(wts_path, mmap_mode="r")
    _SHARED["n_nodes"] = int(n_nodes)
    _SHARED["add_edges_chunk_size"] = int(add_edges_chunk_size)
    _SHARED["quality_function"] = str(quality_function)
    _SHARED["resolution"] = float(resolution)
    _SHARED["progress_every"] = int(progress_every)
    _SHARED["progress_every_seconds"] = float(progress_every_seconds) if progress_every_seconds else None
    _SHARED["verbose"] = bool(verbose)

    # Reset last progress time for time-based cadence
    _SHARED["last_progress_print"] = 0.0


def leiden_worker(args: Tuple[List[Tuple[int, int]], int]) -> Dict[str, np.ndarray]:
    """
    Worker function: receives (run_batch, worker_id). Builds its own igraph once,
    then runs Leiden repeatedly with different seeds.
    """
    run_batch, worker_id = args

    # Build graph once per worker from shared memmapped arrays
    graph, total_weight = _build_graph_from_shared()

    # Set up outputs
    batch_size = len(run_batch)
    n_nodes = graph.vcount()
    labels = np.zeros((batch_size, n_nodes), dtype=np.int32)
    quality = np.zeros(batch_size, dtype=np.float64)
    n_communities = np.zeros(batch_size, dtype=np.int32)
    seeds = np.zeros(batch_size, dtype=np.int32)

    # Partition factory
    qfunc = _SHARED["quality_function"]
    resolution = _SHARED["resolution"]

    # Process each run
    for i, (run_idx, seed) in enumerate(run_batch):
        opt = leidenalg.Optimiser()
        opt.set_rng_seed(int(seed))

        partition = _make_partition(graph, qfunc, resolution)
        opt.optimise_partition(partition, n_iterations=-1)

        labels[i] = np.asarray(partition.membership, dtype=np.int32)
        raw_q = float(partition.quality())
        quality[i] = raw_q / (total_weight * 2) if total_weight > 0 else raw_q
        n_communities[i] = int(len(set(partition.membership)))
        seeds[i] = int(seed)

    return {
        "labels": labels,
        "quality": quality,
        "n_communities": n_communities,
        "seeds": seeds,
        "run_indices": np.array([r for (r, _) in run_batch], dtype=np.int32),
    }



def _make_partition(graph: ig.Graph, qfunc: str, resolution: float):
    if qfunc == "rb":
        return leidenalg.RBConfigurationVertexPartition(
            graph, weights="weight", resolution_parameter=float(resolution)
        )
    elif qfunc == "modularity":
        return leidenalg.ModularityVertexPartition(graph, weights="weight")
    elif qfunc == "cpm":
        return leidenalg.CPMVertexPartition(
            graph, weights="weight", resolution_parameter=float(resolution)
        )
    else:
        raise ValueError(f"Unknown quality_function: {qfunc}")


def _build_graph_from_shared() -> Tuple[ig.Graph, float]:
    """Build igraph from shared memmapped arrays. Add edges in chunks with per-call attributes."""
    rows = _SHARED["rows"]
    cols = _SHARED["cols"]
    wts  = _SHARED["wts"]
    n_nodes = _SHARED["n_nodes"]
    chunk   = _SHARED["add_edges_chunk_size"]

    g = ig.Graph(n=int(n_nodes), directed=False)
    m = rows.shape[0]
    if m != cols.shape[0] or m != wts.shape[0]:
        raise ValueError("rows, cols, weights must have the same length")

    edge_start = 0
    while edge_start < m:
        edge_end = min(edge_start + chunk, m)
        r_chunk = rows[edge_start:edge_end]
        c_chunk = cols[edge_start:edge_end]
        wt_chunk = wts[edge_start:edge_end].astype(float).tolist()

        e_chunk = list(zip(r_chunk.tolist(), c_chunk.tolist()))
        # Bind weights atomically to the edges being added in this call:
        g.add_edges(e_chunk, attributes={"weight": wt_chunk})

        edge_start = edge_end

    total_weight = float(np.sum(wts))
    return g, total_weight




# ---------------------------
# Utilities
# ---------------------------

def _validate_affinity_matrix(A: sp.csr_matrix) -> None:
    """Validate affinity matrix properties."""
    if not isinstance(A, sp.csr_matrix):
        raise ValueError(f"Matrix must be csr_matrix, got {type(A)}")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got {A.shape}")
    if A.data.size and np.any(A.data < 0):
        raise ValueError("Matrix contains negative weights")
    if np.count_nonzero(A.diagonal()) > 0:
        raise ValueError("Matrix diagonal must be all zeros")
    # Check symmetry
    diff = (A - A.T).tocoo()
    if diff.nnz and (np.abs(diff.data).max() > 1e-10):
        raise ValueError("Matrix is not symmetric within tolerance")


def _prepare_matrix_arrays(A: sp.csr_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rows, cols, weights) of the strict upper triangle in COO format."""
    A_triu = sp.triu(A, k=1).tocoo()
    rows = A_triu.row.astype(np.int32, copy=False)
    cols = A_triu.col.astype(np.int32, copy=False)
    wts = A_triu.data.astype(np.float64, copy=False)
    return rows, cols, wts


def _distribute_runs(n_runs: int, n_processes: int, base_seed: int) -> List[List[Tuple[int, int]]]:
    """Distribute runs across processes with simple load balancing."""
    runs_per_process = n_runs // n_processes
    extra_runs = n_runs % n_processes
    batches: List[List[Tuple[int, int]]] = []
    run_idx = 0
    for proc_id in range(n_processes):
        batch_size = runs_per_process + (1 if proc_id < extra_runs else 0)
        batch: List[Tuple[int, int]] = []
        for _ in range(batch_size):
            seed = base_seed + run_idx
            batch.append((run_idx, seed))
            run_idx += 1
        batches.append(batch)
    return batches


def _update_progress_worker():
    """Thread-safe progress update with count/time cadence."""
    if not _SHARED.get("verbose", False):
        return

    now = time.time()
    with _progress_lock:
        _completed_runs.value += 1
        completed = _completed_runs.value
        total = _total_runs.value

        # count-based cadence
        count_trigger = (completed % _SHARED["progress_every"] == 0) or (completed == total)

        # time-based cadence
        time_trigger = False
        interval = _SHARED.get("progress_every_seconds")
        if interval is not None:
            last = _SHARED.get("last_progress_print", 0.0)
            if now - last >= interval:
                time_trigger = True

        if count_trigger or time_trigger:
            elapsed = now - _start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            if completed < total:
                eta = (total - completed) / rate if rate > 0 else float("inf")
                print(
                    f"Progress: {completed}/{total} ({100*completed/total:.1f}%) | "
                    f"Rate: {rate:.1f} runs/s | ETA: {eta/60:.1f} min"
                )
            else:
                print(f"Completed: {completed}/{total} runs in {elapsed:.1f}s")
            _SHARED["last_progress_print"] = now


def _init_h5(
    f: h5py.File,
    n_runs: int,
    n_nodes: int,
    resolution: float,
    base_seed: int,
    qfunc: str,
    normalized: bool,
):
    """Initialize HDF5 structure and metadata."""
    meta = f.create_group("meta")
    meta.attrs["resolution"] = float(resolution)
    meta.attrs["n_runs"] = int(n_runs)
    meta.attrs["base_seed"] = int(base_seed)
    meta.attrs["quality_function"] = str(qfunc)
    meta.attrs["quality_normalization"] = "divide_by_total_edge_weight" if normalized else "none"
    meta.attrs["save_timestamp"] = time.time()

    data = f.create_group("results")
    data.create_dataset("labels", shape=(n_runs, n_nodes), dtype="i4", compression="gzip")
    data.create_dataset("quality", shape=(n_runs,), dtype="f8")
    data.create_dataset("n_communities", shape=(n_runs,), dtype="i4")
    data.create_dataset("seeds", shape=(n_runs,), dtype="i4")
    f.flush()


def _write_h5_chunk(f: h5py.File, run_indices: np.ndarray, worker_result: Dict[str, np.ndarray]) -> None:
    """Write a worker batch into preallocated HDF5 datasets at the correct rows."""
    # Expect contiguous blocks per worker, but handle arbitrary order safely
    idx = np.asarray(run_indices, dtype=np.int64)
    f["results/labels"][idx, :] = worker_result["labels"]
    f["results/quality"][idx] = worker_result["quality"]
    f["results/n_communities"][idx] = worker_result["n_communities"]
    f["results/seeds"][idx] = worker_result["seeds"]
    f.flush()


def leiden_consistency_analysis_tuple(*args, **kwargs):
    """
    Convenience wrapper to match code expecting a 4-tuple:
        L, Q, K, S = leiden_consistency_analysis_tuple(...)
    """
    res = leiden_consistency_analysis(*args, **kwargs)
    return res["labels"], res["quality"], res["n_communities"], res["seeds"]


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    print("Creating example sparse matrix...")
    n = 10_000  # smaller demo size
    density = 12 / n  # mimic k=12 average degree (symmetric)
    rng = np.random.default_rng(42)
    A = sp.random(n, n, density=density, format="csr", random_state=42)
    A = (A + A.T) * 0.5
    A.setdiag(0)
    A.eliminate_zeros()

    print("Running leiden consistency analysis...")
    results = leiden_consistency_analysis(
        affinity_matrix=A,
        resolution=1.0,
        quality_function="rb",
        n_runs=50,
        n_processes=min(16, max(1, mp.cpu_count() - 1)),
        savepath="leiden_consistency_demo_shared.h5",
        checkpoint=True,
        progress_every=10,
        progress_every_seconds=5.0,
        add_edges_chunk_size=200_000,
        verbose=True,
    )

    print(f"\nResults shape: {results['labels'].shape}")
    print(f"Quality range: {results['quality'].min():.6f} - {results['quality'].max():.6f}")
    print(
        f"Community count: {results['n_communities'].min()} - {results['n_communities'].max()} "
        f"(mean={float(results['n_communities'].mean()):.2f})"
    )
