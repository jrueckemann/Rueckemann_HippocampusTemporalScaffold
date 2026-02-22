function [stats,info,stats_pre] = parameter_consistency_extended_Xcomp(labelingsA,labelingsB,mapFineToCoarse,skipRelabeling,report_both,compactoutput)
% PARAMETER_CONSISTENCY_EXTENDED_XCOMP  Quantify clustering consistency across replicates.
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



if nargin<3 || isempty(mapFineToCoarse)
    mapFineToCoarse = 'purity';
end
if nargin<4 || isempty(skipRelabeling)
    skipRelabeling = false;
end
if nargin<5 || isempty(report_both)
    report_both = false;
end
if nargin<6 || isempty(compactoutput)
    compactoutput = true;
end


%Input checking
if iscell(labelingsA)
    labelingsA=cellfun(@(x) x(:),labelingsA,'uni',0);
    labelingsA=cell2mat(labelingsA);
end
assert(all(isfinite(labelingsA)&isreal(labelingsA)&...
    ~isnan(labelingsA)&labelingsA>=0&labelingsA==fix(labelingsA),"all"),...
    ['LabelingA has values that do not seem to correspond to standard '...
    'integer labeling, i.e. integers >=0'])

if iscell(labelingsB)
    labelingsB=cellfun(@(x) x(:),labelingsB,'uni',0);
    labelingsB=cell2mat(labelingsB);
end
assert(all(isfinite(labelingsB)&isreal(labelingsB)&...
    ~isnan(labelingsB)&labelingsB>=0&labelingsB==fix(labelingsB),"all"),...
    ['LabelingB has values that do not seem to correspond to standard '...
    'integer labeling, i.e. integers >=0'])

assert(all(size(labelingsA)==size(labelingsB)),['"labelingsA" does '...
    'not have the same size as the elements of "labelingsB"']);
labelingsA=num2cell(labelingsA,1);
labelingsB=num2cell(labelingsB,1);



%Relabel each labeling with consecutive integers
[~,~,labelingsA]=cellfun(@unique,labelingsA,'uni',0);
[~,~,labelingsB]=cellfun(@unique,labelingsB,'uni',0);

n_combo = numel(labelingsA);
[allARI,allNMI,allmeanJ] = deal(nan(n_combo,1));
[allARI_pre,allNMI_pre,allmeanJ_pre] = deal(nan(n_combo,1));
[allJmat,allJvec,alloverlap,info] = deal(cell(n_combo,1));


% Give each worker a single, shared copy of labelings
Lc_A = parallel.pool.Constant(labelingsA);
Lc_B = parallel.pool.Constant(labelingsB);

parfor idx = 1:n_combo

    opts = struct;
    opts.report_both=report_both;
    opts.lightweight=compactoutput;

    % try
    [results, info{idx}] = clusterlabelsimilarity_extended_new( ...
        Lc_A.Value{idx},Lc_B.Value{idx},...
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