"""
folder_leiden_parallel_v2.py

Implements four performance-focused changes to the folder-level Leiden runner:
  1) Load-on-worker (avoid pickling large CSR matrices).
  2) Strict CSR triplets with compact dtypes (int32 indices/indptr, float32 data).
  3) Per-file processing that reuses the loaded CSR across all resolutions for that file.
  4) Dtype discipline throughout (no accidental float64/int64 bloat).

Also integrates safety against corruption and resource churn:
  • Atomic writes (write to *.part then os.replace -> final).
  • Single-writer policy per output file; no shared HDF5 handles across workers.
  • Clean, bounded file descriptor usage (open-read-close; open-write-close cycles).
  • Optional pool maxtasksperchild to mitigate native leaks.
  • Explicit seeds stored in /source; reproducible seed mapping.
  • "complete" marker toggled at the very end of a successful write.

Requirements:
  - h5py, numpy, scipy (sparse), and your existing `leiden_parallel_shared.leiden_consistency_analysis`.
  - Input files must contain CSR triplets under /knn_graphs/k{K}/ {indptr, indices, data} and either a
    dataset `shape` (2,) or attribute `shape` or `n` (square matrix assumed).

Usage example (Python):

from folder_leiden_parallel_v2 import run_leiden_over_null_folder

run_leiden_over_null_folder(
    affinity_dir=r"D:/null_affinity",
    save_dir=r"D:/leiden_out",
    k_select=24,                         # int or "k24"
    resolutions=[0.2, 0.4, 0.6, 0.8, 1.0],
    n_runs=100,
    n_processes=16,                      # Kilosort2: 5950X + Titan RTX (set BLAS threads = 1)
    quality_function="rb",
    base_seed=42,
    overwrite=False,
    progress_every=50,
    verbose=True,
    pool_maxtasksperchild=1,
)

This will emit one result file per (input file × resolution), e.g.:
  leiden_<src-stem>__k24__qrb__res_0.60__runs100.h5

Each result has `/source` with: source path, filename, k_select, n_processes, quality_function,
resolutions, n_runs, base_seed, seeds (explicit array), timing, and a `complete=True` marker.
"""

from __future__ import annotations

import os
import gc
import time
import math
import uuid
import errno
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from scipy.sparse import csr_matrix

# Your existing parallel Leiden runner (assumed available in PYTHONPATH)
from leiden_parallel_shared import leiden_consistency_analysis


# -----------------------------
# Utilities: CSR loading & dtypes
# -----------------------------

def _ensure_int32(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    return arr.astype(np.int32, copy=False)


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    return arr.astype(np.float32, copy=False)


def load_csr_from_h5(
    h5_path: Union[str, Path],
    k_select: Union[int, str],
    *,
    rdcc_nbytes: int = 32 * 1024 * 1024,
    rdcc_nslots: int = 200003,
    rdcc_w0: float = 0.75,
) -> csr_matrix:
    """
    Load a CSR matrix from an HDF5 at /knn_graphs/k{K}/ with datasets:
      - indptr  (1D, int32)
      - indices (1D, int32)
      - data    (1D, float32)
      - shape   (2,) dataset or attribute (or fall back to square: n x n using attr 'n')

    Uses h5py chunk cache tuning to reduce read I/O overhead.
    """
    h5_path = Path(h5_path)
    k_key = f"k{int(k_select)}" if not str(k_select).startswith("k") else str(k_select)

    with h5py.File(
        h5_path,
        "r",
        rdcc_nbytes=int(rdcc_nbytes),
        rdcc_nslots=int(rdcc_nslots),
        rdcc_w0=float(rdcc_w0),
        libver="latest",
    ) as f:
        if "knn_graphs" not in f:
            raise KeyError(f"Missing /knn_graphs in {h5_path}")
        root = f["knn_graphs"]
        if k_key not in root:
            available = ", ".join(sorted(root.keys()))
            raise KeyError(f"Missing {k_key} in /knn_graphs. Available: {available}")
        g = root[k_key]

        if not all(name in g for name in ("indptr", "indices", "data")):
            raise KeyError(
                f"Group /knn_graphs/{k_key} must contain CSR triplets indptr/indices/data"
            )
        # Read with dtype-cast at IO time to avoid extra copies
        indptr = g["indptr"].astype("i4")[...]
        indices = g["indices"].astype("i4")[...]
        data = g["data"].astype("f4")[...]

        if "shape" in g:
            shape = tuple(int(x) for x in g["shape"][...].ravel())
        else:
            if "shape" in g.attrs:
                shp = g.attrs["shape"]
                shape = tuple(int(x) for x in np.array(shp).ravel())
            elif "n" in g.attrs:
                n = int(g.attrs["n"])  # type: ignore[arg-type]
                shape = (n, n)
            else:
                raise KeyError(
                    f"/knn_graphs/{k_key} missing shape (dataset or attr) and attr 'n'"
                )

        W = csr_matrix((data, indices, indptr), shape=shape)
        # Ensure disciplined dtypes (SciPy expects int32 for index arrays)
        if W.indices.dtype != np.int32:
            W.indices = W.indices.astype(np.int32, copy=False)
        if W.indptr.dtype != np.int32:
            W.indptr = W.indptr.astype(np.int32, copy=False)
        if W.data.dtype != np.float32:
            W.data = W.data.astype(np.float32, copy=False)
        return W


# -----------------------------
# HDF5 write helpers (atomic, single-writer)
# -----------------------------

def _atomic_target_path(final_path: Path) -> Path:
    return final_path.with_suffix(final_path.suffix + f".part-{uuid.uuid4().hex}")


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except Exception:
        pass


def _copy_input_meta_to_output(src_h5: Path, dst_h5: Path) -> None:
    with h5py.File(src_h5, "r") as src, h5py.File(dst_h5, "a") as dst:
        _copy_input_meta_into_open_file(src, dst, mode="attrs")


def _copy_input_meta_into_open_file(src: h5py.File, dst: h5py.File, *, mode: str = "attrs") -> None:
    """Copy /meta from src into dst:/source/original_meta. mode: 'none'|'attrs'|'full'"""
    if mode == "none":
        return
    root = dst.require_group("source")
    om = root.require_group("original_meta")
    if "meta" not in src:
        return
    sm = src["meta"]
    # Copy top-level attrs
    for k, v in sm.attrs.items():
        try:
            om.attrs[k] = v
        except Exception:
            om.attrs[k] = str(v)
    if mode == "attrs":
        return

    def _copy_group(sg: h5py.Group, dg: h5py.Group) -> None:
        for k, v in sg.attrs.items():
            try:
                dg.attrs[k] = v
            except Exception:
                dg.attrs[k] = str(v)
        for name, item in sg.items():
            if isinstance(item, h5py.Dataset):
                if name in dg:
                    del dg[name]
                dg.create_dataset(name, data=item[...])
            elif isinstance(item, h5py.Group):
                ng = dg.require_group(name)
                _copy_group(item, ng)

    _copy_group(sm, om)


def _append_source_info(
    dst_h5: Path,
    *,
    source_path: Path,
    filename: str,
    k_select: Union[int, str],
    n_processes: int,
    quality_function: str,
    resolutions: Sequence[float],
    n_runs: int,
    base_seed: int,
    file_elapsed_sec: float,
    cumulative_elapsed_sec: float,
    mark_complete: bool = True,
) -> None:
    seeds = (int(base_seed) + np.arange(int(n_runs), dtype=np.int64)).astype(np.int64)
    with h5py.File(dst_h5, "a", libver="latest") as f:
        _append_source_info_open(
            f,
            source_path=source_path,
            filename=filename,
            k_select=k_select,
            n_processes=n_processes,
            quality_function=quality_function,
            resolutions=resolutions,
            n_runs=n_runs,
            base_seed=base_seed,
            seeds=seeds,
            file_elapsed_sec=file_elapsed_sec,
            cumulative_elapsed_sec=cumulative_elapsed_sec,
            mark_complete=mark_complete,
        )


def _append_source_info_open(
    f: h5py.File,
    *,
    source_path: Path,
    filename: str,
    k_select: Union[int, str],
    n_processes: int,
    quality_function: str,
    resolutions: Sequence[float],
    n_runs: int,
    base_seed: int,
    seeds: np.ndarray,
    file_elapsed_sec: float,
    cumulative_elapsed_sec: float,
    mark_complete: bool = True,
) -> None:
    src = f.require_group("source")
    src.attrs["source_path"] = str(source_path)
    src.attrs["source_filename"] = str(filename)
    src.attrs["k_select"] = str(k_select)
    src.attrs["n_processes"] = int(n_processes)
    src.attrs["quality_function"] = str(quality_function)

    if "resolutions" in src:
        del src["resolutions"]
    src.create_dataset("resolutions", data=np.asarray(list(resolutions), dtype=np.float64))

    src.attrs["n_runs"] = int(n_runs)
    src.attrs["base_seed"] = int(base_seed)
    if "seeds" in src:
        del src["seeds"]
    src.create_dataset("seeds", data=np.asarray(seeds, dtype=np.int64), dtype="i8")

    timing = src.require_group("timing")
    timing.attrs["file_elapsed_seconds"] = float(file_elapsed_sec)
    timing.attrs["cumulative_elapsed_seconds"] = float(cumulative_elapsed_sec)

    if mark_complete:
        src.attrs["complete"] = True


# -----------------------------
# Core worker logic (per-file)
# -----------------------------

@dataclass
class WorkerConfig:
    save_dir: Path
    k_select: Union[int, str]
    resolutions: Sequence[float]
    n_runs: int
    n_processes: int
    quality_function: str
    base_seed: int
    overwrite: bool
    progress_every: int
    verbose: bool
    # Performance & I/O tuning
    seeds: np.ndarray
    copy_meta_mode: str  # 'none' | 'attrs' | 'full'
    h5_rdcc_nbytes: int
    h5_rdcc_nslots: int
    h5_rdcc_w0: float


def _result_name(stem: str, k_key: str, qfunc: str, res: float, n_runs: int) -> str:
    return f"leiden_{stem}__{k_key}__q{qfunc}__res_{res:.2f}__runs{int(n_runs)}.h5"


def _process_one_file(path: Path, cfg: WorkerConfig) -> Tuple[Path, int, int, float]:
    """
    Process a single input file: load CSR once -> run Leiden for each resolution.

    Returns (path, n_done, n_skipped, elapsed_seconds)
    """
    t0 = time.time()

    # Load CSR once per file with tuned HDF5 read cache
    try:
        W = load_csr_from_h5(
            path,
            cfg.k_select,
            rdcc_nbytes=cfg.h5_rdcc_nbytes,
            rdcc_nslots=cfg.h5_rdcc_nslots,
            rdcc_w0=cfg.h5_rdcc_w0,
        )
    except Exception as e:
        if cfg.verbose:
            print(f"[SKIP: load error] {path.name}: {e}")
        return (path, 0, 0, 0.0)

    k_key = f"k{int(cfg.k_select)}" if not str(cfg.k_select).startswith("k") else str(cfg.k_select)

    n_done = 0
    n_skip = 0
    for i, res in enumerate(cfg.resolutions):
        out_name = _result_name(path.stem, k_key, cfg.quality_function, float(res), cfg.n_runs)
        out_final = cfg.save_dir / out_name

        if out_final.exists() and not cfg.overwrite:
            if cfg.verbose:
                print(f"  - exists, skipping: {out_final.name}")
            n_skip += 1
            continue

        tmp_path = _atomic_target_path(out_final)

        try:
            leiden_consistency_analysis(
                affinity_matrix=W,
                resolution=float(res),
                quality_function=str(cfg.quality_function),
                n_runs=int(cfg.n_runs),
                n_processes=int(cfg.n_processes),
                base_seed=int(cfg.base_seed),
                savepath=str(tmp_path),
                checkpoint=True,
                progress_every=int(cfg.progress_every),
                progress_every_seconds=None,
                verbose=bool(cfg.verbose),
                parallel=False,
            )
        except Exception as e:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            if cfg.verbose:
                print(f"  - ERROR Leiden res={res} on {path.name}: {e}")
            continue

        try:
            # Single open for all metadata writes → less overhead
            with h5py.File(tmp_path, "a", libver="latest") as dst, h5py.File(
                path,
                "r",
                rdcc_nbytes=cfg.h5_rdcc_nbytes,
                rdcc_nslots=cfg.h5_rdcc_nslots,
                rdcc_w0=cfg.h5_rdcc_w0,
                libver="latest",
            ) as src:
                _copy_input_meta_into_open_file(src, dst, mode=cfg.copy_meta_mode)
                elapsed_file = time.time() - t0
                _append_source_info_open(
                    dst,
                    source_path=path.resolve(),
                    filename=path.name,
                    k_select=cfg.k_select,
                    n_processes=cfg.n_processes,
                    quality_function=cfg.quality_function,
                    resolutions=[float(res)],
                    n_runs=cfg.n_runs,
                    base_seed=cfg.base_seed,
                    seeds=cfg.seeds,
                    file_elapsed_sec=elapsed_file,
                    cumulative_elapsed_sec=elapsed_file,
                    mark_complete=True,
                )
                dst.flush()
                try:
                    os.fsync(dst.fid.get_vfd_handle())  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception as e:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            if cfg.verbose:
                print(f"  - ERROR annotating temp result for res={res}: {e}")
            continue

        try:
            os.replace(str(tmp_path), str(out_final))
            _fsync_dir(out_final.parent)
        except Exception as e:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            if cfg.verbose:
                print(f"  - ERROR finalizing {out_final.name}: {e}")
            continue

        n_done += 1
        if (i + 1) % 3 == 0:
            gc.collect()

    del W
    gc.collect()

    return (path, n_done, n_skip, time.time() - t0)


# -----------------------------
# Public API
# -----------------------------

def run_leiden_over_null_folder(
    *,
    affinity_dir: Union[str, Path],
    save_dir: Union[str, Path],
    k_select: Union[int, str],
    resolutions: Sequence[float],
    n_runs: int = 100,
    n_processes: int = 1,
    quality_function: str = "rb",
    base_seed: int = 42,
    overwrite: bool = False,
    progress_every: int = 50,
    verbose: bool = True,
    pool_maxtasksperchild: Optional[int] = 1,  
    copy_meta_mode: str = "attrs",  # 'none'|'attrs'|'full'
    h5_rdcc_nbytes: int = 32 * 1024 * 1024,  # 32 MB read cache per open file
    h5_rdcc_nslots: int = 200003,
    h5_rdcc_w0: float = 0.75,
) -> None:
    """
    Iterate over *.h5 files in `affinity_dir`, and for each file:
      - Load the selected k's CSR once (on the worker),
      - Run Leiden for each resolution (serial within the worker),
      - Write each result atomically and enrich with /source metadata.

    Concurrency model: parallel **per file**. This avoids concurrent writers to the
    same HDF5 and keeps HDF5 handles process-local.
    """
    affinity_dir = Path(affinity_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in affinity_dir.glob("*.h5*") if p.is_file()])
    if verbose:
        print(f"Found {len(files)} input file(s) in {affinity_dir}")

    # Limit hidden native threading
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    start_all = time.time()

    # Precompute seeds once
    seeds = (int(base_seed) + np.arange(int(n_runs), dtype=np.int64)).astype(np.int64)

    cfg = WorkerConfig(
        save_dir=save_dir,
        k_select=k_select,
        resolutions=list(resolutions),
        n_runs=int(n_runs),
        n_processes=int(n_processes),
        quality_function=str(quality_function),
        base_seed=int(base_seed),
        overwrite=bool(overwrite),
        progress_every=int(progress_every),
        verbose=bool(verbose),
        seeds=seeds,
        copy_meta_mode=str(copy_meta_mode),
        h5_rdcc_nbytes=int(h5_rdcc_nbytes),
        h5_rdcc_nslots=int(h5_rdcc_nslots),
        h5_rdcc_w0=float(h5_rdcc_w0),
    )

    if len(files) == 0:
        if verbose:
            print("No input files found; exiting.")
        return

    from multiprocessing import Pool

    total_done = 0
    total_skip = 0

    pool_kwargs = {}
    if pool_maxtasksperchild is not None:
        pool_kwargs["maxtasksperchild"] = int(pool_maxtasksperchild)

    with Pool(processes=min(len(files), os.cpu_count() or 1), **pool_kwargs) as pool:
        results = [pool.apply_async(_process_one_file, (fpath, cfg)) for fpath in files]

        for r in results:
            try:
                path, n_done, n_skip, elapsed = r.get()
                total_done += n_done
                total_skip += n_skip
                if verbose:
                    print(f"Processed {path.name}: wrote {n_done}, skipped {n_skip}, time {elapsed:.1f}s")
            except Exception as e:
                if verbose:
                    print(f"[Worker failure] {e}")

    total_elapsed = time.time() - start_all

    if verbose:
        hh = int(total_elapsed // 3600)
        rem = total_elapsed - 3600 * hh
        mm = int(rem // 60)
        ss = rem - 60 * mm
        print(
            f"All done. Outputs: wrote={total_done}, skipped={total_skip} | total time: {hh:02d}:{mm:02d}:{ss:05.2f}"
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run Leiden across a folder of null-affinity HDF5 files")
    ap.add_argument("affinity_dir", type=str, help="Folder containing input HDF5 affinity files")
    ap.add_argument("save_dir", type=str, help="Folder to store Leiden results")
    ap.add_argument("k_select", type=str, help="k (e.g., 24 or 'k24')")
    ap.add_argument("--resolutions", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--n_runs", type=int, default=100)
    ap.add_argument("--n_processes", type=int, default=8)
    ap.add_argument("--quality_function", type=str, default="rb")
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress_every", type=int, default=50)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--pool_maxtasksperchild", type=int, default=1)

    args = ap.parse_args()

    run_leiden_over_null_folder(
        affinity_dir=args.affinity_dir,
        save_dir=args.save_dir,
        k_select=args.k_select,
        resolutions=args.resolutions,
        n_runs=args.n_runs,
        n_processes=args.n_processes,
        quality_function=args.quality_function,
        base_seed=args.base_seed,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
        verbose=not args.quiet,
        pool_maxtasksperchild=args.pool_maxtasksperchild,
    )
