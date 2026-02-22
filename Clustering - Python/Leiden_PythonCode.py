############################################################
# k-NN affinity graph construction (true + null)
############################################################

## True: Determine k-NN affinity graphs for pseudopopulation data
# (Run in Spyder IDE v5.5.6 using Python v3.9.7 via Anaconda)

from load_mat import load_mat #supports HDF5; interactive
import numpy as np
from rotate_knn import build_null_knn_graphs_parallel
from scipy.io import loadmat as scipy_loadmat


# Import Matlab HDF5 file (v7.3) pseudopopulation ('PP') data matrix (trials X neurons X bins)
data = load_mat()
data = data['PP']

# Create k-NN affinity matrices for pooled pseudopopulation. Default k=[12 24 48]
outdir = 'C:/Users/MyComp/Desktop/knn_true_data/' #replace w/ local path
no_rotation = np.zeros((1, data.shape[1]), dtype=int) #no rotation; used in null distribution
outdict_True = build_null_knn_graphs_parallel(data,integer_offset=no_rotation,save_dir=outdir,n_processes=5) 




## Null: Determine k-NN affinity graphs for pseudopopulation data with each neuron independently rotated WRT predictor bin

# Import standard Matlab file (v7.0) of rotation ('RRR') matrix (iterations X neurons)
rotmat_path = 'C:/Users/MyComp/Desktop/rotmat_August27.mat' #replace w/ local path
rotmat = scipy_loadmat(rotmat_path)
rotmat = rotmat['RRR']


# Create k-NN affinity matrices for each row of the 'rotmat' rotation matrix to support null distribution
outdir = 'C:/Users/MyComp/Desktop/knn_null_rotations/' #replace w/ local path
outdict_Null0827 = build_null_knn_graphs_parallel(data,integer_offset=rotmat,save_dir=outdir,n_processes=5) 




############################################################
# Leiden clustering over saved k-NN affinity graphs
############################################################
# NOTE:
# This section assumes that the k-NN affinity graphs have already been generated and saved
# to disk by the code above. In practice, these steps were run in separate sessions.

# Code operates similarly whether on true or null data, only differing by the input path.


from folder_leiden_parallel_files import run_leiden_over_null_folder


knndir = 'C:/Users/MyComp/Desktop/knn_true_data/' #replace w/ local path (true or null folder)
savedir = 'C:/Users/MyComp/Desktop/LeidenCommunities/LeidenTrueFits/'

# Run Leiden Community Finding on each affinity matrix in the knndir folder
# kNN = 12, RB configuration (Leiden) resolutions = [0.2 0.6 1.0], Repetitions = 100
run_leiden_over_null_folder(affinity_dir=knndir, save_dir=savedir, k_select=12, resolutions=[0.2,0.6,1.0], n_runs=100, n_processes=1, overwrite=False, progress_every=10)

#Parallelizes across input files (not within the n_runs repeats for a given file)