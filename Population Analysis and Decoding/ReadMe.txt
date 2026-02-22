This folder contains MATLAB functions used for population-level analyses
including creation of pseudopopulations and decoding of pseudopopulation activity described in
"The Primate Hippocampus Constructs a Temporal Scaffold Anchored to Behavioral Events".


Pseudopopulation construction and preprocessing
- trlratetransform.m
    Trialwise rate transformations and normalization.

- createpseudopopulation.m
    Construction of pseudopopulations from trialwise and unitwise data
    structures.

- rotatetrialdata.m
    Circular rotation of trial data (used for control / null analyses).



Primary decoding and population-level analysis functions
- CVdecoding.m
    Cross-validated Bayesian decoding pipeline.

- CVdecoding_2samp.m
    Cross-validated Bayesian decoding variant for two-sample comparisons.


Supporting decoding and analysis utilities
- bayesiandecoder_parfor.m
    Bayesian decoding implementation

- estimatebayesianparameter.m
    Helper function used by Bayesian decoding routines to estimate
    distributional parameters.

- corrdecoder.m
    Correlation-based decoding

- altCorrDecoder.m
    Comparable stand-alone correlation-based decoding

- epochdecoding.m
    Population-level decoding across task epochs.