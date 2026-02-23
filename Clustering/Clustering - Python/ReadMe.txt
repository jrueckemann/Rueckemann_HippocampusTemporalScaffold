Leiden_PythonCode.py demonstrates the workflow for k-NN graph construction and Leiden community detection
used in "The Primate Hippocampus Constructs a Temporal Scaffold Anchored to Behavioral Events".


Data loading helper
- load_mat.py
    Lightweight helper for loading MATLAB v7.3 (HDF5) data files into
    Python for downstream analysis.

kNN graph construction utilities
- rotate_knn.py
    Construction of kNN affinity graphs from trialwise population data,
    including support for independent per-neuron circular rotations used
    to generate null distributions.

- spectral_clustering_faiss.py
    FAISS-based kNN search and affinity matrix construction used by the
    clustering pipeline.

Core clustering and graph utilities
- folder_leiden_parallel_files.py
    Wrapper for applying Leiden clustering across a folder of saved kNN
    affinity graphs, writing results to HDF5 files.

- leiden_parallel_shared.py
    Core implementation of repeated Leiden clustering on a single
    affinity graph, including seed control and result aggregation.