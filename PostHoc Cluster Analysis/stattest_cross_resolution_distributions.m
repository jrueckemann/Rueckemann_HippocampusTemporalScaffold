function [stats, true_values, null_median_per_pair] = stattest_cross_resolution_distributions(trueComp, nullComp)
% STATTEST_CROSS_RESOLUTION_DISTRIBUTIONS
%   Compare cross-resolution clustering similarity between TRUE data and NULL rotations.
%
% Inputs
%   trueComp : Vector of cross-resolution similarities from true data comparisons
%              (length = number of cross-resolution pairs)
%   nullComp : Matrix where each row corresponds to a cross-resolution pair and
%              each column represents NULL similarities for that pair
%              (size: [n_pairs x n_null_per_pair])
%
% Outputs
%   stats : struct with fields:
%       .pooled : ranksum (Mann-Whitney) test between all TRUE and all NULL values
%           - p, zval, U, A (Vargha-Delaney), cliffs_delta, n_true, n_null, med_true, med_null
%       .pair_median : analysis of per-pair median NULL vs TRUE
%           - signrank test if paired structure exists, otherwise descriptive stats
%           - effect sizes and medians
%   true_values : Vector of true cross-resolution similarities (same as trueComp)
%   null_median_per_pair : Vector of median of nulls matched to each true
%   comparison
%
% Notes
%   - Designed for cross-resolution comparisons where structure is different from
%     within-resolution symmetric matrices
%   - Uses non-parametric tests for robustness
%
% Jon Rueckemann 2025

% Input validation
assert(isvector(trueComp), 'trueComp must be a vector of cross-resolution similarities');
assert(ismatrix(nullComp), 'nullComp must be a matrix');
assert(length(trueComp) == size(nullComp,1), ...
    'Number of true comparisons must match number of rows in nullComp');

trueComp = trueComp(:);  % Ensure column vector
n_pairs = length(trueComp);
n_null_per_pair = size(nullComp, 2);

% Flatten null data for pooled analysis
null_pooled = nullComp(:);
true_pooled = trueComp;

% Remove any NaN values
valid_true = ~isnan(true_pooled);
valid_null = ~isnan(null_pooled);
true_pooled = true_pooled(valid_true);
null_pooled = null_pooled(valid_null);

%% Pooled analysis: all true vs all null
[p_pooled, ~, stats_pooled] = ranksum(true_pooled, null_pooled);

% Calculate effect sizes
n1 = numel(true_pooled);
n2 = numel(null_pooled);
U1 = stats_pooled.ranksum - n1*(n1+1)/2;
A  = U1 / (n1*n2);                       % Vargha-Delaney A
delta_pooled = 2*A - 1;                  % Cliff's delta from U

stats.pooled = struct( ...
    'test',            'ranksum', ...
    'p',               p_pooled, ...
    'zval',            stats_pooled.zval, ...
    'U',               U1, ...
    'A_VarghaDelaney', A, ...
    'cliffs_delta',    delta_pooled, ...
    'n_true',          n1, ...
    'n_null',          n2, ...
    'med_true',        median(true_pooled), ...
    'med_null',        median(null_pooled) ...
    );

%% Per-pair analysis: compare true value vs median null for each pair
null_median_per_pair = median(nullComp, 2, 'omitnan');
valid_pairs = ~isnan(trueComp) & ~isnan(null_median_per_pair);
true_valid = trueComp(valid_pairs);
null_median_valid = null_median_per_pair(valid_pairs);

if sum(valid_pairs) > 0
    % Test whether true values are systematically different from null medians
    [p_pair, ~, stats_pair] = signrank(true_valid, null_median_valid);

    % Effect sizes for paired comparison
    rb = rank_biserial_paired(true_valid, null_median_valid);
    delta_pair = cliffs_delta_unpaired(true_valid, null_median_valid);

    stats.pair_median = struct( ...
        'test',                   'signrank (paired)', ...
        'p',                      p_pair, ...
        'zval',                   stats_pair.zval, ...
        'W_signedrank',           stats_pair.signedrank, ...
        'rank_biserial_paired',   rb, ...
        'cliffs_delta_unpaired',  delta_pair, ...
        'n_pairs',                sum(valid_pairs), ...
        'med_true',               median(true_valid), ...
        'med_null',               median(null_median_valid) ...
        );
else
    % No valid pairs for comparison
    stats.pair_median = struct( ...
        'test',                   'signrank (paired)', ...
        'p',                      NaN, ...
        'zval',                   NaN, ...
        'W_signedrank',           NaN, ...
        'rank_biserial_paired',   NaN, ...
        'cliffs_delta_unpaired',  NaN, ...
        'n_pairs',                0, ...
        'med_true',               NaN, ...
        'med_null',               NaN ...
        );
end

% Output the processed data
true_values = trueComp;
null_values = null_pooled;  % Only return valid null values

end

function delta = cliffs_delta_unpaired(x, y)
% Unpaired Cliff's delta using Mann-Whitney relation via ranksum.
x = x(:); y = y(:);
[~, ~, st] = ranksum(x, y);
n1 = numel(x); n2 = numel(y);
R1 = st.ranksum;
U1 = R1 - n1*(n1+1)/2;
A  = U1 / (n1*n2);
delta = 2*A - 1;
end

function rrb = rank_biserial_paired(x, y)
% Rank-biserial correlation for paired data (Wilcoxon signed-rank effect size).
d = x(:) - y(:);
mask = d ~= 0;                  % exclude ties (zero diffs)
d = d(mask);
n_eff = numel(d);
if n_eff == 0
    rrb = 0; return;
end
% Ranks of absolute differences (average ties)
r = tiedrank(abs(d));
Wplus = sum(r(d > 0));
Wminus = sum(r(d < 0));
denom = n_eff*(n_eff+1)/2;
rrb = (Wplus - Wminus) / denom;
end