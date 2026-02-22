This folder contains MATLAB functions used for population clustering analyses (viz spherical k-means),
as described in "The Primate Hippocampus Constructs a Temporal Scaffold Anchored to Behavioral Events".

Primary calling function
- multiclustermodels.m
    Wrapper for evaluating multiple clustering models and parameter
    settings (e.g., different numbers of clusters) on the same data.

Supporting clustering functions, alignment utility, and null creation
- runClusterModelComparison.m
    Compares clustering solutions across models or parameter settings
    using similarity and consistency metrics.

- clusterTrialwiseActivity.m
    Core function for clustering trialwise population activity.
    Operates on trial × neuron × bin population representations.

- align_clusters_by_temporal_order.m
    Aligns cluster labels across runs or conditions based on temporal
    ordering of cluster centroids.

- rotatetrialdata.m
    Circular rotation of trial data, used for null/control clustering
    analyses.