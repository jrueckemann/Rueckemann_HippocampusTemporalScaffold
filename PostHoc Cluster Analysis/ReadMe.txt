This folder contains MATLAB functions used for post hoc analyses of population clustering results, including analyses of cluster structure, boundary consistency, and relationships between clustering outputs and behavior, as described in "The Primate Hippocampus Constructs a Temporal Scaffold Anchored to Behavioral Events".


Comparison of clustering solution functions:
-mass_clustersimilarity_True.m 
    within- and cross-resolution similarity for TRUE runs.

-mass_clustersimilarity_NullRes.m
    TRUE vs NULL similarity using null runs stored in a .mat file.

- stattest_cross_resolution_distributions.m
    Nonparametric tatistical comparison of clustering metrics across 
    resolutions or conditions.


Boundary analysis functions:
- massboundarydetect.m
- massboundarydetect_kmeans.m
    Boundary detection applied across clustering solutions (Leiden
    or k-means variants).

- createboundarymats.m
    Aggregrates and reorders boundary matrices created in massboundarydetect*.

- trial2runboundary.m
    Collapses trial boundaries into run-level boundary vectors & peak metrics.

- analyzeBoundaryEpochSimilarity.m
    Relates detected clustering boundaries to task epochs and behavioral
    structure.


Supporting functions and utilities:
- load_parallel_leiden_hdf5.m
    Loads parallel-processed Leiden clustering results written in HDF5 format.

- trial_cluster_borders.m
    Builds cross-trial cluster template & detects per-trial borders near template borders.

- parameter_consistency_extended_V2.m 
    Within-resolution replicate consistency stats.

- parameter_consistency_extended_Xcomp.m
    Cross-resolution comparison stats.

- clusterlabelsimilarity_extended_new.m
    Cluster label alignment & similarity metrics from contingency table.




