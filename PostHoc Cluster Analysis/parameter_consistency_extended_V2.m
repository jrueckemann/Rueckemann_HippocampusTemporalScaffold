function [stats,info,stats_pre] = parameter_consistency_extended_V2(labelings,mapFineToCoarse,skipRelabeling,report_both,compactoutput)
% PARAMETER_CONSISTENCY_EXTENDED  Quantify clustering consistency across replicates.
%
%NOTE: Code relabels clusters with consecutive integers, so any intrinsic
%value of cluster label (e.g. noise or behavioral relevance will be lost)
%
%
%   stats = parameter_consistency(labelings)
%
%   Computes pairwise similarity measures (ARI, NMI, Jaccard) between
%   cluster labelings obtained from replicate runs at the same parameter.
%
% Input
%   labelings:          Cell array {R x 1}, each element a vector of 
%                         cluster labels for one replicate. If a matrix is 
%                         passed, columns are treated as replicates.
%   mapFineToCoarse:    'none' (default), 'purity', or 'jaccard'
%   skipRelabeling    : Boolean; skip Hungarian matching. Desirable if
%                         not engaging greedy L2-->L1 relabeling AND order 
%                         in pairwise similarity values is not important,
%                         e.g. when only summary statistics are considered
%                         downstream.  If skipRelabeling is True, 
%                         no Hungarian matching will occur and greedy 
%                         cluster assignment is not needed. Default to
%                         'compactoutput' value, because Hungarian matching
%                         only affects specific pairwise values but not
%                         the summary statistics (without greedy remapping)
%   report_both:        Boolean; report the prelabeling comparisons as well
%   compactoutput:      Boolean; only save summary statistics
%
%
% Output
%   stats : struct with summary fields
%       .meanARI, .stdARI, .ARI
%       .meanNMI, .stdNMI, .NMI
%       .meanmuJaccard, .stdmuJaccard, .muJaccard
%       .JaccardMatrix       (cell array of pairwise Jaccard matrices)
%       .maxJaccardVector    (cell array of max Jaccard values per cluster)
%       .LabelOverlapMatrix  (cell array of overlap fractions)
%
% Notes
%   - Uses clusterlabelsimilarity_extended_new to compute similarity between pairs.
%   - Omit-NaN statistics are reported for robustness.
%
%Jon Rueckemann 2025



if nargin<2 || isempty(mapFineToCoarse)
    mapFineToCoarse = 'purity';
end
if nargin<3 || isempty(skipRelabeling)
    skipRelabeling = false;
end
if nargin<4 || isempty(report_both)
    report_both = false;
end
if nargin<5 || isempty(compactoutput)
    compactoutput = true;
end


if iscell(labelings)
    labelings=cellfun(@(x) x(:),labelings,'uni',0);
    labelings=cell2mat(labelings);
end
assert(all(isfinite(labelings)&isreal(labelings)&...
    ~isnan(labelings)&labelings>=0&labelings==fix(labelings),"all"),...
    ['Labeling has values that do not seem to correspond to standard '...
    'integer labeling, i.e. integers >=0'])
labelings=num2cell(labelings,1);


%Relabel each labeling with consecutive integers
[~,~,labelings]=cellfun(@unique,labelings,'uni',0);

R = numel(labelings);
n_combo = R*(R-1)/2;
[allARI,allNMI,allmeanJ] = deal(nan(n_combo,1));
[allARI_pre,allNMI_pre,allmeanJ_pre] = deal(nan(n_combo,1));
[allJmat,allJvec,alloverlap,info] = deal(cell(n_combo,1));


% Give each worker a single, shared copy of labelings
Lc = parallel.pool.Constant(labelings);

parfor idx = 1:n_combo
    [x, y] = ij_from_pairindex(idx, R);

    opts = struct;
    opts.report_both=report_both;
    opts.lightweight=compactoutput;

    % try
        [results, info{idx}] = ...
            clusterlabelsimilarity_extended_new(Lc.Value{x},Lc.Value{y},...
            mapFineToCoarse,skipRelabeling,opts);

        if report_both
            allARI_pre(idx) = results.pre.ARI;
            allNMI_pre(idx) = results.pre.NMI;
            allmeanJ_pre(idx) = results.pre.muJMAX;

            results=results.post;
        end
        

        allARI(idx) = results.ARI;
        allNMI(idx) = results.NMI;
        allmeanJ(idx) = results.muJMAX;
        if ~compactoutput
            allJmat{idx} = results.JMAT;
            allJvec{idx} = results.JROWMAX;
            alloverlap{idx} = results.coverage_mat;
        end
    % catch
    % end

end

info=cell2mat(info);

stats.meanARI = mean(allARI,'omitnan');
stats.stdARI  = std(allARI,'omitnan');
stats.ARI  = allARI;

stats.meanNMI = mean(allNMI,'omitnan');
stats.stdNMI  = std(allNMI,'omitnan');
stats.NMI  = allNMI;

stats.meanmuJaccard = mean(allmeanJ,'omitnan');
stats.stdmuJaccard  = std(allmeanJ,'omitnan');
stats.muJaccard  = allmeanJ;

stats.JaccardMatrix = allJmat;
stats.maxJaccardVector = allJvec;
stats.LabelOverlapMatrix = alloverlap;


%Pre-matching similarity statistics
if nargout==3
    stats_pre.meanARI = mean(allARI_pre,'omitnan');
    stats_pre.stdARI  = std(allARI_pre,'omitnan');
    stats_pre.ARI  = allARI_pre;

    stats_pre.meanNMI = mean(allNMI_pre,'omitnan');
    stats_pre.stdNMI  = std(allNMI_pre,'omitnan');
    stats_pre.NMI  = allNMI_pre;

    stats_pre.meanmuJaccard = mean(allmeanJ_pre,'omitnan');
    stats_pre.stdmuJaccard  = std(allmeanJ_pre,'omitnan');
    stats_pre.muJaccard  = allmeanJ_pre;
end
end

function [i,j] = ij_from_pairindex(idx, R)
% Map linear index idx=1..R*(R-1)/2 to the pair (i,j) with 1<=i<j<=R
% without constructing nchoosek or pairs matrix.

    idx = double(idx);
    R   = double(R);

    % cumulative pairs: c(i) = i*(2R - i - 1)/2 for i = 0..R-1
    % Invert approximately, guard against tiny negatives in the radicand.
    disc = (R - 0.5)^2 - 2*idx;
    disc = max(disc, 0);  % numerical safety
    i = floor( R - 0.5 - sqrt(disc) );

    % c(i) using i as defined above
    ci = (i)*(2*R - i - 1)/2;
    if ci >= idx
        i = i - 1;
        ci = (i)*(2*R - i - 1)/2;
    end

    pos = idx - ci;      % 1-based position within row i
    j = i + pos + 1;     % j index
    i = i + 1;           % convert row base to 1-based i
end