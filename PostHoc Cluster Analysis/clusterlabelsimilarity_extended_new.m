function [results, info] = clusterlabelsimilarity_extended_new(L1,L2,mapFineToCoarse,skipRelabeling,opts)
% CLUSTERLABELSIMILARITY_EXTENDED  Compare two clusterings with optional
% bidirectional fine→coarse consolidation.
%   • Builds a dense contingency S once using ACCUMARRAY
%   • Optional Hungarian matching on Jaccard, then greedy-fill L2_unmatched
%   • Optional reverse: greedy-map any L1_unmatched to L2
%   • Metrics are computed on the contingency table after relabeling
%   • Optionally reports pre-equalization metrics from S in the same pass
%   • Lightweight mode skips heavy/rarely used outputs for large sweeps
%
% Assumptions
%   • L1, L2 are integer labels, already compact and consecutive 1..k
%   • k1, k2 ≤ ~15 -- dense k×k ops are faster/cleaner than sparse matrices
%
% Inputs
%   L1, L2           : Label arrays (any shape, vectorized internally).
%   mapFineToCoarse  : 'purity' (overlap count), 'jaccard' (Jaccard score),
%                       or 'none' (no hierarchical relabeling).
%                       Default: 'purity'
%   skipRelabeling   : Boolean, skip matching/greedy entirely;
%                       compute metrics without relabeling. Default: False
%   opts             : struct with fields (all optional):
%       .reverse_match  (logical, default=mapFineToCoarse dependent)
%           - enable reverse greedy (L1_unmatched → L2)
%       .report_both    (logical, default=false)
%           - compute metrics before and after equalization
%       .lightweight    (logical, default=true)
%           - omit heavy fields to speed large sweeps
%       .hungarian_eps  (double,  default=eps)
%           - threshold for matchpairs to reject zero-overlap
%
%
% Outputs
%   results    : struct with fields
%               (post-equalization by default;
%                   plus pre_* if report_both=true)
%       NMI, ARI, muJMAX, JROWMAX
%       (optional when ~lightweight): JMAT, p_L1_given_L2, coverage_mat,
%                                     H_per_parent, H_weighted_mu,
%                                     parent_purity
%       (if report_both): pre_NMI, pre_ARI, pre_muJMAX, pre_JROWMAX, ...
%
%   info       : struct with bookkeeping
%       k1_orig, k2_orig, k_equalized  
%       reverse_match_used, hungarian_pairs, L2_to_L1_map, 
%       L1_to_L2_map (reverse-stage only on unmatched),
%       L1_unmatched_ids, L2_unmatched_ids (from Hungarian stage)
%
% Notes
%   • Equalization explicitly accepts merges on both sides to align
%       boundaries and compare structure.
%   • When skipRelabeling=true, equalization is not attempted;
%       only pre-equalization metrics are returned.
%   • Greedy assignment:
%       score1 → overlap tie → score2 → overlap tie → smallest index
%
%
% Jon Rueckemann 2025


if nargin < 3 || isempty(mapFineToCoarse), mapFineToCoarse = 'purity'; end
if nargin < 4 || isempty(skipRelabeling), skipRelabeling = false; end

switch lower(mapFineToCoarse)
    case 'purity'
        puritymap=true;
        revmatch_val=true;
    case 'jaccard'
        puritymap=false;
        revmatch_val=true;
    case 'none'
        puritymap=false;
        revmatch_val=false;
        skipRelabeling=true;
    otherwise
        error('mapFinetoCoarse is not {''none'',''purity'',''jaccard''}')
end

if nargin < 5 || isempty(opts), opts = struct; end
if ~isfield(opts,'reverse_match'), opts.reverse_match = revmatch_val; end
if ~isfield(opts,'report_both'),   opts.report_both   = false; end
if ~isfield(opts,'lightweight'),   opts.lightweight   = true; end
if ~isfield(opts,'hungarian_eps'), opts.hungarian_eps = eps; end

% Flatten & validate
L1 = L1(:); L2 = L2(:);
assert(numel(L1)==numel(L2), 'Inputs must have the same number of elements.');
k1_tmp = max(L1); k2_tmp = max(L2);

%Calculate contingency matrix  (FLIP L1 and L2, if L2 has more clusters)
if k1_tmp<=k2_tmp
    k1 = k1_tmp;
    k2 = k2_tmp;
    S = accumarray([L1, L2], 1, [k1, k2]);
    k1_k2_flipped=false;
else
    k1 = k2_tmp;
    k2 = k1_tmp;
    S = accumarray([L2, L1], 1, [k1, k2]);
    k1_k2_flipped=true;
end
r = sum(S, 2);         % k1×1
c = sum(S, 1).';       % k2×1

%Compute Jaccard scores (k1×k2)
den = r + c.' - S;           % union counts
J = zeros(k1, k2);
mask = den > 0;
J(mask) = S(mask) ./ den(mask);
% (0/0 treated as 0)

%Pre-merging metrics
if opts.report_both || skipRelabeling
    if opts.lightweight
        [pre_NMI, pre_ARI, pre_muJMAX, pre_JROWMAX, pre_JMAT] = ...
            metrics_from_contingency(S);
        [pre_cov, pre_pL1gL2, pre_Hrow, pre_Hw, pre_parent_purity]=deal([]);
    else
        [pre_NMI, pre_ARI, pre_muJMAX, pre_JROWMAX, pre_JMAT, ...
            pre_cov, pre_pL1gL2, pre_Hrow, pre_Hw, pre_parent_purity] = ...
            metrics_from_contingency(S);
    end
end

% Early exit if skipping matching/equalization
if skipRelabeling
    results = pack_results(pre_NMI,pre_ARI,pre_muJMAX,pre_JROWMAX,...
        pre_JMAT,pre_pL1gL2,pre_cov,pre_Hrow,pre_Hw,pre_parent_purity,...
        opts.lightweight);
    info = struct('k1_orig',k1,'k2_orig',k2,'k_equalized',[], ...
        'k1_k2_flipped',k1_k2_flipped,'reverse_match_used',false, ...
        'hungarian_pairs',[],'L2_to_L1_map',[], 'L1_to_L2_map',[], ...
        'L1_unmatched_ids',[], 'L2_unmatched_ids',[]);
    return;
end

% Hungarian on Jaccard Index matrix
[cPairs,L1_unmatched,L2_unmatched]=matchpairs(J,opts.hungarian_eps,'max');
% Build L2→L1 & L1→L2 maps; NaN = unmatched
L2_to_L1 = nan(k2,1);
L2_to_L1(cPairs(:,2)) = cPairs(:,1);
L1_to_L2 = nan(k1,1);
L1_to_L2(cPairs(:,1)) = cPairs(:,2);

% Greedy match remaining L2 (columns) to L1 (rows)
if ~isempty(L2_unmatched)
    for idx = 1:numel(L2_unmatched)
        col_idx = L2_unmatched(idx);
        row_idx = greedy_pick_row_for_col(col_idx, S, J, puritymap);
        L2_to_L1(col_idx) = row_idx;
    end
end

%Optional reverse greedy match (L1_unmatched → L2)
if opts.reverse_match && ~isempty(L1_unmatched)
    reverse_used = true;
    for idx = 1:numel(L1_unmatched)
        row_idx = L1_unmatched(idx);
        col_idx = greedy_pick_col_for_row(row_idx, S, J, puritymap);
        L1_to_L2(row_idx) = col_idx;
    end
elseif ~isempty(L1_unmatched) && opts.report_both
    %Degenerate hierarchical mapping; early return
    results.pre = pack_results(pre_NMI,pre_ARI,pre_muJMAX,pre_JROWMAX,...
        pre_JMAT,pre_pL1gL2,pre_cov,pre_Hrow,pre_Hw,pre_parent_purity,...
        opts.lightweight);
    info = struct('k1_orig',k1,'k2_orig',k2,'k_equalized',[], ...
        'k1_k2_flipped',k1_k2_flipped,'reverse_match_used',false, ...
        'hungarian_pairs',[],'L2_to_L1_map',[], 'L1_to_L2_map',[], ...
        'L1_unmatched_ids',[], 'L2_unmatched_ids',[]);
    return;
else
    assert(isempty(L1_unmatched),...
        ['There is not a valid match for each cluster in L1.' ...
        'Hierarchical Hungarian mapping is degenerate.']);
    reverse_used = false;
end

%Restructure the contingency matrix after relabeling
assert(all(~isnan(L1_to_L2))&&all(~isnan(L2_to_L1)),...
    'Algorithmic failure; Invalid hierarchical mapping');
old_rowidx=cumsum(ones(size(S)),1);
old_colidx=cumsum(ones(size(S)),2);
new_rowidx = L1_to_L2(old_rowidx(:));        %L1 rows map via L1_to_L2
new_colidx = L2_to_L1(old_colidx(:));        %L2 cols map via L2_to_L1
new_S=accumarray([new_rowidx,new_colidx],S(:),[],@sum, 0);

%Remove unused rows/columns from restructured contingency table
dropcol=true(size(new_S,2),1);
dropcol(L2_to_L1)=false;
droprow=true(size(new_S,1),1);
droprow(L1_to_L2)=false;
new_S(:,dropcol)=[];
new_S(droprow,:)=[];
assert(size(new_S,1)==size(new_S,2),'new_S not square after equalization');
Kstar=size(new_S,2);

%Post-restructuring metrics
if opts.lightweight
    [NMI, ARI, muJMAX, JROWMAX, JMAT] = metrics_from_contingency(new_S);
    [pL1gL2, covMat, H_per_parent, H_weighted_mu, parent_purity] = ...
        deal([]);
else
    [NMI, ARI, muJMAX, JROWMAX, JMAT, ...
        covMat, pL1gL2, H_per_parent, H_weighted_mu, parent_purity] = ...
        metrics_from_contingency(new_S);
end

%Package outputs
if opts.report_both
    results = struct();
    % Post-equalization (canonical)
    post = pack_results(NMI, ARI, muJMAX, JROWMAX, JMAT, pL1gL2, covMat, H_per_parent, H_weighted_mu, parent_purity, opts.lightweight);
    % Pre-equalization (read-only)
    pre  = pack_results(pre_NMI, pre_ARI, pre_muJMAX, pre_JROWMAX, pre_JMAT, pre_pL1gL2, pre_cov, pre_Hrow, pre_Hw, pre_parent_purity, opts.lightweight);
    results.post = post;
    results.pre  = pre;
else
    results = pack_results(NMI, ARI, muJMAX, JROWMAX, JMAT, pL1gL2, covMat, H_per_parent, H_weighted_mu, parent_purity, opts.lightweight);
end

info = struct('k1_orig',k1,'k2_orig',k2,'k_equalized',Kstar, ...
    'k1_k2_flipped',k1_k2_flipped,...
    'reverse_match_used',reverse_used, ...
    'hungarian_pairs',cPairs, ...
    'L2_to_L1_map',L2_to_L1, ...
    'L1_to_L2_map',L1_to_L2, ...
    'L1_unmatched_ids',L1_unmatched, ...
    'L2_unmatched_ids',L2_unmatched);

end

%% Helper functions
function [NMI, ARI, muJMAX, JROWMAX, JMAT, covMat, pL1gL2, H_row, H_w, parent_purity] = metrics_from_contingency(S)
% Computes metrics from a dense contingency (k1×k2) without touching N-length labels
k1 = size(S,1); k2 = size(S,2);
r = sum(S,2); c = sum(S,1).'; N = sum(r);

% Jaccard matrix & row maxima
den = r + c.' - S; JMAT = zeros(k1,k2);
mask = den > 0; JMAT(mask) = S(mask) ./ den(mask);
JROWMAX = max(JMAT, [], 2);
muJMAX = mean(JROWMAX);

% Normalized Mutual Information
%(natural log; geometric mean normalization)
lnN = log(max(1,N));
% Guard zero rows/cols
rpos = r > 0; cpos = c > 0;
% Using sums of v*log v trick
v = S(:);
vpos = v(v>0);
sum_vlogv = sum(vpos .* log(vpos));
sum_rlogr = sum(r(rpos) .* log(r(rpos)));
sum_clogc = sum(c(cpos) .* log(c(cpos)));
MI_nat = (sum_vlogv - sum_rlogr - sum_clogc + N*lnN) / max(1,N);
Hx_nat = (N*lnN - sum_rlogr) / max(1,N);
Hy_nat = (N*lnN - sum_clogc) / max(1,N);
if Hx_nat <= 0 || Hy_nat <= 0
    % If either side is a single cluster, define NMI as 1 if identical partition of items, else 0
    % (Based on equality of S having only one nonzero row/col matching diagonally when k1==k2)
    % Here we approximate with agreement on dominant diagonal mass
    [~,i1] = max(r); [~,j1] = max(c);
    NMI = double(S(i1,j1) == N);
else
    NMI = max(0, min(1, MI_nat / sqrt(Hx_nat * Hy_nat)));
end

% ARI
% a = sum over cells of n_ij choose 2
a = sum( S(:) .* (S(:) - 1) ) / 2;
rb = sum( r .* (r - 1) ) / 2;
cb = sum( c .* (c - 1) ) / 2;
pairs_total = N * (N - 1) / 2;
expectedIndex = 0;
if pairs_total > 0
    expectedIndex = (rb * cb) / pairs_total;
end
maxIndex = 0.5 * (rb + cb);
denom = maxIndex - expectedIndex;
if denom <= 0
    ARI = 0;
else
    ARI = (a - expectedIndex) / denom;
end

% Conditional matrices & entropies
if nargout > 5
    % p(L1|L2) and coverage (row-normalized S)
    c_safe = c; c_safe(c_safe==0) = 1;
    pL1gL2 = S ./ c_safe.';   % k1×k2

    r_safe = r; r_safe(r_safe==0) = 1;
    covMat = S ./ r_safe;     % k1×k2

    % Row entropies H(L2|L1=i)
    H_row = zeros(k1,1);
    for i = 1:k1
        pi = covMat(i,:);
        pi = pi(pi>0);
        if ~isempty(pi)
            H_row(i) = -sum(pi .* log2(pi));
        end
    end
    if N > 0
        H_w = sum( (r / N) .* H_row );
    else
        H_w = 0;
    end
    parent_purity = max(covMat, [], 2);
else
    pL1gL2 = []; covMat = []; H_row = []; H_w = []; parent_purity = [];
end
end

function r_idx = greedy_pick_row_for_col(c_idx, S, J, puritymap)
% Choose L1 row for L2 column j
if puritymap %Use 'purity' to map
    scores1 = S(:,c_idx);
    scores2 = J(:,c_idx);
else %Use 'jaccard' to map
    scores1 = J(:,c_idx);
    scores2 = S(:,c_idx);
end

%Deterministically reassign labels
candidx = find(scores1 == max(scores1));  % Get actual indices
if numel(candidx) > 1
    scores2_subset = scores2(candidx);
    [~, best_idx] = max(scores2_subset);
    if sum(scores2_subset == scores2_subset(best_idx)) > 1
        % Handle ties in scores2
        tied_indices = candidx(scores2_subset == scores2_subset(best_idx));
        counts = S(tied_indices, c_idx);
        [~, final_idx] = max(counts);
        r_idx = tied_indices(final_idx);
    else
        r_idx = candidx(best_idx);
    end
else
    r_idx = candidx;
end
end

function c_idx = greedy_pick_col_for_row(r_idx, S, J, puritymap)
% Choose L2 column for L1 row i (reverse)
if puritymap %Use 'purity' to map
    scores1 = S(r_idx,:).';
    scores2 = J(r_idx,:).';
else %Use 'jaccard' to map
    scores1 = J(r_idx,:).';
    scores2 = S(r_idx,:).';
end

%Deterministically reassign labels
candidx = find(scores1 == max(scores1));  % Get actual indices
if numel(candidx) > 1
    scores2_subset = scores2(candidx);
    [~, best_idx] = max(scores2_subset);
    if sum(scores2_subset == scores2_subset(best_idx)) > 1
        % Handle ties in scores2
        tied_indices = candidx(scores2_subset == scores2_subset(best_idx));
        counts = S(r_idx,tied_indices);
        [~, final_idx] = max(counts);
        c_idx = tied_indices(final_idx);
    else
        c_idx = candidx(best_idx);
    end
else
    c_idx = candidx;
end
end


function out = pack_results(NMI, ARI, muJMAX, JROWMAX, JMAT, pL1gL2, covMat, H_row, H_w, parent_purity, lightweight)
if lightweight
    out = struct('NMI',NMI,'ARI',ARI,'muJMAX',muJMAX,'JROWMAX',JROWMAX);
else
    out = struct('NMI',NMI,'ARI',ARI,'muJMAX',muJMAX,'JROWMAX',JROWMAX, ...
        'JMAT',JMAT,'p_L1_given_L2',pL1gL2,'coverage_mat',covMat, ...
        'H_per_parent',H_row,'H_weighted_mu',H_w,'parent_purity',parent_purity);
end
end
