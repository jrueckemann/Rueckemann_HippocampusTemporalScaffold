function out = analyzeBoundaryEpochSimilarity(behaviorVec, resolutions, w, metric)
% ANALYZEBOUNDARYEPOCHSIMILARITY
% Compare boundary–epoch correspondence for True vs Null across resolutions.
%
% Inputs
%   behaviorVec : 1 x B logical/double (behavioral boundaries; 1=boundary)
%   resolutions : struct array with fields:
%                 - name           : char/string (e.g., 'low','mid','high')
%                 - trueBoundaries : R x B logical/double (runs x bins)
%                 - nullBoundaries : R x B logical/double (runs x bins)
%   w           : scalar tolerance in bins for ±W matching
%   metric      : 'f1' (default) or 'hitrate'
%   doPlot      : true/false (default=true) to render 3×2 violin plot
%
% Output (struct)
%   out(r) with fields per resolution:
%       .name
%       .trueScores, .nullScores        (R x 1)
%       .median_true, .iqr_true
%       .median_null, .iqr_null
%       .p_mannwhitney                  (two-sided)
%       .cliffs_delta                   (effect size)
%
%Jon Rueckemann 2025

if nargin < 4 || isempty(metric), metric = 'f1'; end


assert(isvector(behaviorVec), 'behaviorVec must be 1xB');
behaviorVec = logical(behaviorVec(:)'); % row logical
B = numel(behaviorVec);
behTol = toleranceMask(behaviorVec, w, B);

R = numel(resolutions);
out = struct('name', [], 'trueScores', [], 'trueHits', [], 'nullScores', [], ...
             'median_true', [], 'iqr_true', [], ...
             'median_null', [], 'iqr_null', [], ...
             'p_mannwhitney', [], 'cliffs_delta', []);
out = repmat(out, R, 1);

for i = 1:R
    name = string(resolutions(i).name);
    Tmat = resolutions(i).trueBoundaries;
    Nmat = resolutions(i).nullBoundaries;

    % Validate shapes
    assert(size(Tmat,2)==B && size(Nmat,2)==B, 'Boundary matrices must be R x B');
 %   assert(size(Tmat,1)==size(Nmat,1), 'True/Null must have same # runs per resolution');

    % Scores per run
    switch lower(metric)
        case 'f1'
            % figure; subplot(5,1,1:4); imagesc(Tmat); subplot(5,1,5); imagesc(behTol);
            [trueScores,trueHits] = f1_boundary_scores(Tmat, behTol, behaviorVec);
            nullScores = f1_boundary_scores(Nmat, behTol, behaviorVec);
        case 'hitrate'
            trueScores = hitrate_scores(Tmat, behaviorVec, w);
            nullScores = hitrate_scores(Nmat, behaviorVec, w);
            trueHits=nan(size(trueScores));            
        otherwise
            error('Unknown metric: %s (use ''f1'' or ''hitrate'')', metric);
    end

    % Summaries
    out(i).name = name;
    out(i).trueScores = trueScores(:);
    out(i).trueHits = trueHits(:);
    out(i).nullScores = nullScores(:);
    out(i).median_true = median(trueScores);
    out(i).iqr_true = iqr(trueScores);
    out(i).median_null = median(nullScores);
    out(i).iqr_null = iqr(nullScores);

    % Stats: Mann–Whitney U (ranksum) and Cliff's delta
    out(i).p_mannwhitney = ranksum(trueScores, nullScores, 'method','approximate','tail','both');
    out(i).cliffs_delta = cliffs_delta(trueScores, nullScores);
end

end


function behTol = toleranceMask(behaviorVec, w, B)
behTol = false(1,B);
if w==0
    behTol = behaviorVec;
else    
    for j = -w:w
        behTol = behTol | circshift(behaviorVec, j, 2);
    end
end
end

function [F1,TP] = f1_boundary_scores(boundaryMat, behTol, behaviorVec)
% Per-run F1 with ±W tolerance (matches against behTol; penalizes FP vs behTol, FN vs behaviorVec)
NR = size(boundaryMat,1);
[F1,TP] = deal(zeros(NR,1));
for r = 1:NR
    pred = logical(boundaryMat(r,:));
    TP(r) = sum(pred & behTol);
    FP = sum(pred & ~behTol);
    FN = sum(~pred & behaviorVec);
    if TP(r)==0
        F1(r) = 0;
    else
        prec = TP(r) / (TP(r)+FP);
        rec  = TP(r) / (TP(r)+FN);
        F1(r) = 2*(prec*rec) / (prec+rec);
    end
end
end

function HR = hitrate_scores(boundaryMat, behaviorVec, w)
% Per-run hit rate: fraction of behavior boundaries that have >=1 predicted boundary within ±W
[NR, B] = size(boundaryMat);
behIdx = find(behaviorVec);
nTrue = numel(behIdx);
HR = zeros(NR,1);
for r = 1:NR
    pred = find(boundaryMat(r,:));
    if isempty(pred) || nTrue==0
        HR(r) = 0;
    else
        hits = 0;
        for b = 1:nTrue
            lo = max(1, behIdx(b)-w);
            hi = min(B, behIdx(b)+w);
            if any(pred >= lo & pred <= hi)
                hits = hits + 1;
            end
        end
        HR(r) = hits / nTrue;
    end
end
end

function d = cliffs_delta(x, y)
% Cliff's delta: Pr(X>Y) - Pr(X<Y)
x = x(:); y = y(:);
nx = numel(x); ny = numel(y);

% Compute via pairwise sign using sorting indices
cnt_gt = 0; cnt_lt = 0;
for i = 1:nx
    cnt_gt = cnt_gt + sum(y < x(i));
    cnt_lt = cnt_lt + sum(y > x(i));
end
d = (cnt_gt - cnt_lt) / (nx * ny);
end