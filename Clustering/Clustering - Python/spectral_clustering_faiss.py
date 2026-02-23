# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 02:21:13 2025

@author: Jon Rueckemann
"""

import numpy as np
import faiss
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh
import h5py
import os
from scipy.linalg import orthogonal_procrustes, qr




#Note: steps are parcellated between separate functions to avoid RAM overload

#Calculate affinity matrix via FAISS
def spectral_affinity_faiss(
        XX,
        knn_values=[5, 10, 20, 50, 100], 
        correlation_metric=True,
        normalize_data=True,
        knngraphtype='complete',
        dendroreorder=False,
        verbose=True        
        ):
    """
    Construct sparse k-nearest-neighbor (k-NN) affinity graphs using FAISS 
    with cosine similarity from 3D neural population data.

    Parameters
    ----------
    XX : np.ndarray, shape (T, N, B)
        Input data tensor where:
            - T = number of time bins per trial
            - N = number of neurons (features)
            - B = number of trials
        The data will be reshaped to (T * B, N) for k-NN construction.

    knn_values : list of int, optional
        List of k values for which to construct k-NN graphs.

    correlation_metric : bool, optional
        If True, center each row of the input by subtracting its mean
        (equivalent to cosine of centered data, approximating correlation).

    normalize_data : bool, optional
        If True, normalize each row of the reshaped data to unit L2 norm
        (required for cosine similarity to behave properly).
        
    knngraphtype : {'complete', 'mutual'}, optional
    Method for symmetrizing the k-NN graph:
        - 'complete': keep edge if it exists in either direction (element-wise max).
        - 'mutual'  : keep edge only if it exists in both directions (element-wise min).
    
    dendroreorder : bool, optional
        If True, reorder the matrix so that it is ordered by time bin then trial,
        rather than the default of time bin is nested within trial

    verbose : bool, optional
        If True, print progress messages during graph construction.

    Returns
    -------
    knn_graphs : dict of {int: scipy.sparse.csr_matrix}
        Dictionary mapping each value of k to a symmetrized sparse
        affinity matrix of shape (T * B, T * B), weighted by cosine similarity.

    params : dict
        Metadata dictionary recording:
            - Original input shape
            - List of k-NN values
            - Whether normalization or centering was applied
            - Graph symmetrization type used
    """
    
    assert XX.ndim == 3, "Input data must be of shape (T, N, B)"
    T, N, B = XX.shape
    
    # Validate graph symmetrization type
    valid_graph_types = ['complete', 'mutual']
    if knngraphtype not in valid_graph_types:
        raise ValueError(f"Invalid 'knngraphtype': {knngraphtype}. Must be one of {valid_graph_types}.")


    X = XX.transpose(0, 2, 1).reshape(-1, N)  # (T*B, N)
    
    if dendroreorder: #Reorder so trials are nested within bins
        idx = np.arange(B * T).reshape(B, T).T.reshape(-1)
        X = X[idx,:]
    
    if correlation_metric:
        X -= X.mean(axis=1, keepdims=True)
    if normalize_data:
        X = normalize(X, norm='l2', axis=1)

    X_f32 = X.astype(np.float32)
    n_samples = X.shape[0]

    index = faiss.IndexFlatIP(N)
    index.add(X_f32)


    # Precompute knn graphs
    knn_graphs = {}
    for k_nn in knn_values:
        if verbose:
            print(f"Creating knn affinity graph for k={k_nn} ...")

        _, indices = index.search(X_f32, k_nn + 1)
        indices = indices[:, 1:]

        rows = np.repeat(np.arange(n_samples), k_nn)
        cols = indices.flatten()
        sims = np.sum(X_f32[rows] * X_f32[cols], axis=1)

        W = csr_matrix((sims, (rows, cols)), shape=(n_samples, n_samples))
        # Create symmetry
        if knngraphtype == 'complete':
            W = W.maximum(W.T)
        elif knngraphtype == 'mutual':
            W = W.minimum(W.T)
            
        knn_graphs[k_nn] = W
                    
    # Pack processing parameters for metadata
    params = {
        'input_shape': [T, N, B],
        'knn_values': knn_values,
        'normalize_data': normalize_data,
        'correlation_metric': correlation_metric,
        'knngraphtype': knngraphtype,
    }
    
    return knn_graphs, params, X



    
def spectral_laplacian(
        knn_graphs,
        params,
        laplace_method='symmetric',
        verbose=True
        ):
    
    """
    Compute graph Laplacians for multiple k-NN graphs using either symmetric or random walk normalization.

    Parameters:
    - knn_graphs: dict of {k_nn: affinity sparse matrix}
    - params: dict, input parameters (will be updated with laplace_method)
    - laplace_method: 'symmetric' or 'random_walk'
    - verbose: print progress if True
    
    Returns:
    - L_dict: dict of {k_nn: Laplacian sparse matrix}
    - params_out: updated parameter dictionary
    """
    
    if laplace_method not in ['symmetric', 'random_walk']:
        raise ValueError(f"Invalid laplace_method: {laplace_method}")

    L_dict={}
    for k_nn in knn_graphs:
        if verbose:
            print(f"Creating affinity graph Laplacian for k={k_nn} ...")
            
        W = knn_graphs[k_nn]            
        d = np.array(W.sum(axis=1)).flatten() #degree matrix (total connectivity of each point)
        
        if np.any(d == 0):
            print(f"Isolated points in affinity graph k={k_nn} ")
            #raise ValueError(f"Isolated points in affinity graph k={k_nn} ")
            
         
        n = W.shape[0]
        I = identity(n, format='csr')
        if laplace_method == 'symmetric':
            D_inv_sqrt = diags(1.0 / np.sqrt(d + 1e-10))
            L_dict[k_nn] = I - D_inv_sqrt @ W @ D_inv_sqrt
        elif laplace_method == 'random_walk':
            D_inv = diags(1.0 / (d + 1e-10))
            L_dict[k_nn] = I - D_inv @ W
            
    # Update params
    params_out=dict(params)
    params_out['laplace_method'] = laplace_method
    
    return L_dict, params_out
        
        



def spectral_embeddings(
    L_dict,
    params,
    eigvec_counts=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    save_path=None,
    verbose=True
):
    """
    Compute spectral embeddings (eigenvectors of Laplacians).

    Parameters
    ----------
    L_dict : dict
        Dictionary of Laplacian matrices (keyed by k_nn).
    params : dict
        Metadata including 'laplace_method' and input shape.
    eigvec_counts : list of int
        Numbers of eigenvectors to extract for each Laplacian.
    save_path : str or None
        If given, save results to this HDF5 path.
    verbose : bool
        Print status updates.

    Returns
    -------
    embeddings : dict
        Dictionary of spectral embeddings, keys like 'k10_n5'.
    params : dict
        Updated params including 'embedding_keys' and 'n_eigvecs_list'.
    """
    
    # If user supplies a folder, append a default prefix
    if save_path and os.path.isdir(save_path):
        save_path = os.path.join(save_path, "spectralembedding")

    laplace_method = params['laplace_method']
    embeddings = {}
    embedding_keys = []
    n_eigvecs_used = []
    
    # Validate eigenvector count
    all_sample_sizes = [L.shape[0] for L in L_dict.values()]
    min_samples = min(all_sample_sizes)
    invalid_counts = [k for k in eigvec_counts if k >= min_samples]
    if invalid_counts:
        raise ValueError(
            f"Each n_eigvecs must be less than the number of samples (min = {min_samples}). "
            f"Invalid entries: {invalid_counts}"
        )

    for k_nn, L in L_dict.items():

        for k_use in eigvec_counts:

            if verbose:
                print(f"Computing embedding: kNN={k_nn}, n_eigvecs={k_use}")
            
            #Find the eigenvectors corresponding to the k smallest eigenvalues
            eigvals, eigvecs = eigsh(L, k=k_use, which='SM')
            if laplace_method == 'symmetric':
                embedding = normalize(eigvecs, norm='l2', axis=1)
            else:
                embedding = eigvecs

            key = f'k{k_nn}_n{k_use}'
            embeddings[key] = embedding
            embedding_keys.append(key)
            n_eigvecs_used.append(k_use)

    # Update params
    params_out=dict(params)
    params_out['embedding_keys'] = embedding_keys
    params_out['n_eigvecs_list'] = n_eigvecs_used

    # Optional save
    if save_path:
        if verbose:
            print(f"Saving embeddings to {save_path}")
        with h5py.File(f"{save_path}.h5", 'w') as f:
            # Store metadata
            meta = f.create_group('meta')
            for k, v in params_out.items():
                try:
                    if isinstance(v, str):
                        meta.attrs[k] = v
                    elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                        meta.attrs[k] = ','.join(v)  # Join string lists into single attribute
                    elif isinstance(v, (list, np.ndarray)):
                        meta.create_dataset(k, data=np.array(v))
                    elif isinstance(v, (int, float, np.integer, np.floating)):
                        meta.create_dataset(k, data=v)
                    else:
                        meta.attrs[k] = str(v)
                except Exception as e:
                    if verbose:
                        print(f"Skipping param {k} due to: {e}")

            # Store embeddings
            grp = f.create_group('embedding')
            grp.attrs['laplace_method'] = laplace_method
            for key, mat in embeddings.items():
                grp.create_dataset(key, data=mat.astype(np.float32))

    return embeddings, params_out



def discrete_spectral_clustering(embeddings, 
                                 params,
                                 save_path=None,
                                 verbose=True):
    """
    Apply Yu & Shi (2003) discrete spectral clustering to all embeddings.

    Parameters
    ----------
    embeddings : dict
        Dictionary of spectral embeddings (output from spectral_embeddings()).
        Keys are strings like 'k10_n7'.

    params : dict
        Metadata dictionary from spectral_embeddings(), must include:
        - 'laplace_method': must be 'symmetric'
        - 'embedding_keys': list of keys in `embeddings`

    verbose : bool
        If True, print progress.

    Returns
    -------
    labels_dict : dict
        Dictionary mapping each embedding key to its discrete cluster labels.

    Raises
    ------
    ValueError
        If Laplacian is not symmetric.
    """
    # Check Laplacian method
    laplace_method = params['laplace_method']
    if laplace_method != 'symmetric':
        raise ValueError(
            f"'discrete' clustering requires symmetric Laplacian. "
            f"Got laplace_method = '{laplace_method}'."
        )
        
    # If user supplies a folder, append a default prefix
    if save_path and os.path.isdir(save_path):
        save_path = os.path.join(save_path, "discretespectrallabel")

    labels_dict = {}

    for key in params.get('embedding_keys', embeddings.keys()):
        U = embeddings[key]
        n_samples, n_clusters = U.shape

        # Normalize defensively
        U = U / np.linalg.norm(U, axis=1, keepdims=True)

        # Greedy orthogonal row initialization
        idx = [np.random.choice(n_samples)]
        for _ in range(1, n_clusters):
            dist = 1 - np.abs(U @ U[idx].T).max(axis=1)
            idx.append(np.argmax(dist))
        T = U[idx]
        T_full = np.tile(T, (n_samples // n_clusters + 1, 1))[:n_samples]

        # Orthogonal Procrustes alignment
        R, _ = orthogonal_procrustes(U, T_full)
        U_rot = U @ R
        labels = np.argmax(U_rot, axis=1)

        labels_dict[key] = labels

        if verbose:
            print(f"[Discrete Clustering] Key={key} → {n_clusters} clusters.")
            
    # Optional save
    if save_path:
        if verbose:
            print(f"Saving discrete labels to {save_path}")
        with h5py.File(f"{save_path}.h5", 'w') as f:
            # Store metadata
            meta = f.create_group('meta')
            for k, v in params.items():
                try:
                    if isinstance(v, str):
                        meta.attrs[k] = v
                    elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                        meta.attrs[k] = ','.join(v)  # Join string lists into single attribute
                    elif isinstance(v, (list, np.ndarray)):
                        meta.create_dataset(k, data=np.array(v))
                    elif isinstance(v, (int, float, np.integer, np.floating)):
                        meta.create_dataset(k, data=v)
                    else:
                        meta.attrs[k] = str(v)
                except Exception as e:
                    if verbose:
                        print(f"Skipping param {k} due to: {e}")

            # Store labels_dict
            grp = f.create_group('discretelabel')
            grp.attrs['laplace_method'] = laplace_method
            for key, mat in labels_dict.items():
                grp.create_dataset(key, data=mat.astype(np.float32))

    return labels_dict


def qr_spectral_clustering(embeddings, 
                           params, 
                           save_path=None,
                           verbose=True):
    """
    Apply cluster_qr spectral clustering to all embeddings.

    This method uses pivoted QR decomposition of U.T to select k landmark rows,
    and assigns each sample to the nearest landmark (no k-means, no tuning).

    Parameters
    ----------
    embeddings : dict
        Dictionary of spectral embeddings (output from spectral_embeddings()).
        Keys are strings like 'k10_n7'.

    params : dict
        Metadata dictionary from spectral_embeddings(), must include:
        - 'embedding_keys'

    verbose : bool
        If True, print progress.

    Returns
    -------
    labels_dict : dict
        Dictionary mapping each embedding key to its assigned cluster labels.
    """
    
    # If user supplies a folder, append a default prefix
    if save_path and os.path.isdir(save_path):
        save_path = os.path.join(save_path, "qrspectrallabel")
    
    labels_dict = {}

    for key in params.get('embedding_keys', embeddings.keys()):
        U = embeddings[key]  # shape (n_samples, k)
        n_samples, k = U.shape

        # Normalize rows defensively
        U = U / np.linalg.norm(U, axis=1, keepdims=True)

        # Pivoted QR decomposition of U.T → get landmark indices
        _, _, pivot_indices = qr(U.T, pivoting=True)
        landmark_rows = U[pivot_indices[:k], :]  # shape (k, k)

        # Assign each row to nearest landmark
        dists = np.linalg.norm(U[:, np.newaxis, :] - landmark_rows[np.newaxis, :, :], axis=2)
        labels = np.argmin(dists, axis=1)

        labels_dict[key] = labels

        if verbose:
            print(f"[cluster_qr] Key={key} → {k} clusters via pivoted QR.")
            
    # Optional save
    if save_path:
        if verbose:
            print(f"Saving qr labels to {save_path}")
        with h5py.File(f"{save_path}.h5", 'w') as f:
            # Store metadata
            meta = f.create_group('meta')
            for k, v in params.items():
                try:
                    if isinstance(v, str):
                        meta.attrs[k] = v
                    elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                        meta.attrs[k] = ','.join(v)  # Join string lists into single attribute
                    elif isinstance(v, (list, np.ndarray)):
                        meta.create_dataset(k, data=np.array(v))
                    elif isinstance(v, (int, float, np.integer, np.floating)):
                        meta.create_dataset(k, data=v)
                    else:
                        meta.attrs[k] = str(v)
                except Exception as e:
                    if verbose:
                        print(f"Skipping param {k} due to: {e}")

            # Store labels_dict
            grp = f.create_group('qrlabel')
            for key, mat in labels_dict.items():
                grp.create_dataset(key, data=mat.astype(np.float32))


    return labels_dict



#LCC (largest connected component) calculation to ensure appropriateness of knn
from typing import Dict
from scipy.sparse.csgraph import connected_components

def assess_lcc_coverage(knn_graphs: Dict[int, csr_matrix]) -> Dict[int, float]:
    """
    Assess the LCC (Largest Connected Component) fraction for each k-NN affinity matrix.

    Parameters
    ----------
    knn_graphs : dict of {int: csr_matrix}
        Dictionary mapping each k to a preprocessed, symmetrized sparse affinity matrix
        (shape: [n_samples, n_samples], weighted or unweighted).

    Returns
    -------
    lcc_fractions : dict of {int: float}
        Dictionary mapping each k to the fraction of nodes in the LCC.
        Values close to 1.0 indicate well-connected graphs.
    """
    lcc_fractions = {}

    for k, A in knn_graphs.items():
        n_nodes = A.shape[0]

        # Zero diagonal (if not already done)
        A.setdiag(0)
        A.eliminate_zeros()

        # Binarize the graph (LCC cares about connectivity, not weights)
        A_bool = A.copy()
        A_bool.data = np.ones_like(A_bool.data, dtype=np.uint8)

        # Find connected components
        n_components, labels = connected_components(A_bool, directed=False, return_labels=True)

        # Count size of largest component
        component_sizes = np.bincount(labels)
        lcc_size = component_sizes.max()
        lcc_fraction = lcc_size / n_nodes

        lcc_fractions[k] = lcc_fraction

    return lcc_fractions
