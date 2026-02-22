function [borderdist, bstruct, clustprob] = trial_cluster_borders(L, window, use_support, support_thresh, min_adj)
% TRIAL_CLUSTER_BORDERS
% Combines building a cross-trial clustering template and detecting borders in each trial.
%
% Inputs:
%   L              - [B x T] label matrix (bins x trials)
%   window         - scalar, how many bins to search around template boundary
%   use_support    - (bool) whether to allow minority-supported clusters
%   support_thresh - (0-1) fraction of trials needed to support minority cluster
%   min_adj        - (int) minimum adjacent bins required to accept minority cluster
%
% Outputs:
%   borderdist  - [NxB] border density across trials (N=# template borders)
%   bstruct:
%       .trial_borders      - [N x 3 x T]:
%               (:,1,:) = bin index of detected border (NaN if none)
%               (:,2,:) = 1 if per-trial transition matches CT(n,2)->CT(n,3)
%               (:,3,:) = 1 iff next template border is minority-supported 
%       .clustertemplate    - [B x 1] template cluster ID per bin
%       .template_borders   - [N x 4] Change Table (CT):
%               [last_bin_prev, prev_label, next_label, is_mode_template]
%   clustprob   - [B x k] probability of each cluster by bin
%
%
%Jon Rueckemann 2025

if nargin<2 || isempty(window)
    window = 3;
end
if nargin<3 || isempty(use_support)
    use_support = false;
end
if nargin<4 || isempty(support_thresh)
    support_thresh = .35;
end
if nargin<5 || isempty(min_adj)
    min_adj = 3;
end


[B, T] = size(L);
clustertemplate = mode(L, 2);
support_mask = false(B, 1);

% Optional minority-supported overrides
if use_support
    %Relabel in case L is not consecutive, positive integers
    [unqID,~,unqidx] = unique(L(:));
    unqidx=reshape(unqidx,size(L));

    %Find the ratio of cluster representation per bin
    clustratio=zeros(B,numel(unqID));
    for m=1:numel(unqID)
        clustratio(:,m)=mean(unqidx==m,2);
    end

    %Remove the default cluster (max ratio - equivalent to mode)
    [~,maxidx]=max(clustratio,[],2,'linear');
    clustratio(maxidx)=0;

    %Only retain the runner-up ratios for each bin
    [maxval,maxidx]=max(clustratio,[],2,'linear');
    clustratio_RU=zeros(size(clustratio));
    clustratio_RU(maxidx)=maxval;


    %Find secondary clusters above ratio threshold occupying adjacent bins
    %and insert them into the cluster template
    for m=1:numel(unqID)
        curidx=bwareaopen(clustratio_RU(:,m)>support_thresh,min_adj);
        clustertemplate(curidx)=unqID(m);
        support_mask(curidx)=true;
    end
end


% Probability of each cluster by bin
bidx=cumsum(ones(size(L)),1);
clustprob=accumarray([bidx(:) L(:)],1,...
    [size(L,1) max(L,[],'all')])./size(L,2);


% Identify borders in the template
% edges are START indices of runs; the transition is between edges(n+1)-1 and edges(n+1)
edges = find([true; diff(clustertemplate)~=0]); % run starts
N     = max(numel(edges) - 1, 0);

CT = nan(N, 4);
for n = 1:N
    border_bin = edges(n+1) - 1;              % last bin of "from" cluster
    CT(n, 1)   = border_bin;                  % border bin index
    CT(n, 2)   = clustertemplate(border_bin); % from label
    CT(n, 3)   = clustertemplate(border_bin+1); % to label
    CT(n, 4)   = ~support_mask(border_bin);   % 1 if from mode, 0 if minority-supported
end


% Per-trial border detection near each template border
trial_borders = nan(N, 3, T);
for t = 1:T
    labels = L(:, t);
    for n = 1:N
        bin_center = CT(n, 1);
        from_lab   = CT(n, 2);
        to_lab     = CT(n, 3);

        w1 = max(1, bin_center - window);
        w2 = min(B, bin_center + window);
        seg = labels(w1:w2);

        % candidate positions i where seg(i)==from & seg(i+1)==to
        cand_local = find(seg(1:end-1) == from_lab & seg(2:end) == to_lab);
        if isempty(cand_local)
            % no matching from->to transition in the window
            trial_borders(n, 1, t) = NaN;
            trial_borders(n, 2, t) = 0;
        else
            % choose the candidate whose boundary (i -> i+1) is closest to bin_center
            cand_abs = cand_local + w1 - 1;              % bin index of the "from" side
            [~,ci]   = min(abs(cand_abs - bin_center));  % closest boundary start
            tr_idx   = cand_abs(ci);

            trial_borders(n, 1, t) = tr_idx;             % record boundary bin (last from label)
            trial_borders(n, 2, t) = 1;                  % exact from->to match
        end

        % 1 if next CT border is minority-supported (0 for last)
        if n < N
            trial_borders(n, 3, t) = (CT(n+1, 4) == 0);
        else
            trial_borders(n, 3, t) = 0;
        end
    end
end
bstruct=struct('trial_borders',trial_borders,...
    'clustertemplate',clustertemplate,...
    'template_borders',CT);


%Calculate distribution of each border across trials
borderdist=zeros(N,B);
for m=1:N
    curborder=squeeze(trial_borders(m,1,:));

    curborder=curborder(~isnan(curborder));

    if ~isempty(curborder)
        borderdist(m,:)=accumarray(curborder(:),1,[B 1]);
    end   
end
borderdist=borderdist./T; %normalize by trial count to get occupancy ratio

end