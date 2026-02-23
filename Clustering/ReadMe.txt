This folder contains two complementary clustering pipelines used for population-level analyses in
“The Primate Hippocampus Constructs a Temporal Scaffold Anchored to Behavioral Events.”

The code is organized in two folders by clustering approach and programming language.

Clustering - Python: 
Python code implementing k-nearest-neighbor (kNN) graph construction and Leiden community detection, including null graph generation via per-neuron rotations.

Clustering - Matlab: 
Matlab code implementing spherical k-means on trialwise population activity, including clustering of null data after per-neuron rotations.


Sample data:
pseudotrials_example_L_R.mat - BxNxT matrix (predictor bins X neurons X trials).  500 pseudotrials generated from true left trials and 500 pseudotrials generated from true right trials for a total of 1000 trials in this example.  Analyses in the manuscript used 1000 pseudotrials per left and right trial type.

rotmat_example.mat -  RxN (iterations X neurons). Matrix of integer bin offsets used to independently rotate each neuron's data. Each row specifies an individual randomization.

