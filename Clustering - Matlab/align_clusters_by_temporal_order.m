function [label_map, apply_map_fn, reorder_matrix_fn] =align_clusters_by_temporal_order(binInput,fixwrap,isLabelMatrix)
%ALIGN_CLUSTERS_BY_TEMPORAL_ORDER Reorders clusters based on center of mass over time
%
%   Inputs:
%     binInput      - Either:
%                       B×K matrix of cluster probabilities per time bin
%                       (rows sum to 1), or
%                       B×T matrix of integer labels (isLabelMatrix=true)
%     fixwrap       - boolean.  Forces the cluster highest in the first
%                       bin to be ordered first despite where its center of
%                       mass might be.  Fixes cluster that wraps around
%                       beginning and end with arbitrary CoM in middle.
%     isLabelMatrix - boolean. If true, treat binInput as B×T label matrix
%                       and convert to B×K probabilities.
%
%   Outputs:
%     label_map     - K × 1 vector; label_map(j) = i
%                       means original cluster j is reassigned to rank i
%     apply_map_fn  - Function handle to apply label_map to a label vector
%     reorder_matrix_fn - Function handle to reorder K-column matrices
%
%
%Jon Rueckemann 2025


if nargin < 2 || isempty(fixwrap),       fixwrap = true;        end
if nargin < 3 || isempty(isLabelMatrix), isLabelMatrix = false; end


% Convert labels -> per-bin probabilities if requested
if isLabelMatrix
    [B, T]=size(binInput);

    binInput(isnan(binInput)|binInput<1)=inf; %handle unclustered entries
    %places unclustered in the last column of probability matrix
    [origID,~,clustidx]=unique(binInput); %relabel with positive integers

    %Calculate cluster likelihood for each bin
    rowidx=repmat(1:B,1,T)';
    binLikelihoods=accumarray([rowidx clustidx],1,[B max(clustidx)])./T;

    %Drop the unclustered label column
    if any(isinf(origID))
        binLikelihoods(:,isinf(origID))=[];
        origID(isinf(origID))=[];
    end

    K=size(binLikelihoods,2);
else
    % binInput is B×K probabilities
    binLikelihoods=binInput;
    [B, K]=size(binLikelihoods);

origID=1:K;
end

%Compute center of mass (weighted average of time bins) for each cluster
time=(1:B)';
com=(binLikelihoods'*time)./sum(binLikelihoods,1)'; % K×1 vector

%Rank clusters by their CoM (ascending: early = 1, late = K)
[~,sorted_indices]=sort(com,'ascend');%earliest first

%Handle wrap-around: ensure cluster largest in first bin is first
[~,maxIdx]=max(binLikelihoods(1,:));
if fixwrap && maxIdx~=sorted_indices(1)
    tmp_sort=sorted_indices;
    tmp_sort(maxIdx==tmp_sort)=[];
    sorted_indices=[maxIdx';tmp_sort];
end

label_map = zeros(K,1);
label_map(sorted_indices) = 1:K;

%Apply mapping to an arbitrary label vector of IDs in 'origID'
apply_map_fn = @(labels) remap_labels_by_u(labels,origID,label_map);


%Function to reorder matrices with K columns according to the new order
reorder_matrix_fn = @(M) M(:,inverse_map(label_map));

end

function inv_map=inverse_map(label_map)
%INVERSE_MAP Returns permutation p such that newM(:,p) == oldM(:,label_map)
% i.e., inv_map(i) = j if label_map(j) == i
K=length(label_map);
inv_map=zeros(K,1);
for j=1:K
    inv_map(label_map(j))= j;
end
end

function out = remap_labels_by_u(labels, orig_labels, map_to_rank)
%Map label IDs in 'orig_labels' to temporal ranks via map_to_rank; leave others unchanged.
out=labels;
[tf,idx]=ismember(labels,orig_labels); % idx in 1..K for matched labels
out(tf)=map_to_rank(idx(tf)); % replace with temporal rank
end