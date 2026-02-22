function [binLikelihoods, clusterLabels, info] = clusterTrialwiseActivity(data,nClust,opts)
% clusterTrialwiseActivity clusters neural population activity across time bins/trials.
%
% Inputs:
%   data - Either:
%       A) B x N x T matrix of raw data (time bins x neurons x trials), OR
%       B) D x D distance matrix (D = B*T)
%
%   opts - struct with fields:
%       .method      - 'kmeans' | 'kmedoids' 
%           Note {'spectral','dbscan','agglomerative','hdbscan'} are
%           deprecated in this version.
%       .numClusters - number of clusters (if required)
%       .eps         - epsilon parameter for DBSCAN
%       .minpts      - minimum points for DBSCAN
%       .isDistanceMatrix - true if inputData is a BT×BT distance matrix
%
% Outputs:
%   binLikelihoods - B x K matrix of cluster likelihoods per bin
%   clusterLabels  - B x T array of cluster IDs
%   info           - struct summarizing clustering output
%
%
%Jon Rueckemann 2025

if nargin<2
    nClust=[];
end
if nargin<3
    opts=struct('method', 'kmeans');
end

defaultOpts = struct('method','kmeans','numClusters',8,'eps',.7, ...
    'minpts',30,'isDistanceMatrix',false,'replicates',10,'MaxIter',1000,...
    'distancemethod','correlation','linkmethod','average','B',[],'T',[]);

% Fill in missing fields in opts
optFields = fieldnames(defaultOpts);
for i = 1:numel(optFields)
    if ~isfield(opts, optFields{i}) || isempty(opts.(optFields{i}))
        opts.(optFields{i}) = defaultOpts.(optFields{i});
    end
end
if ~isempty(nClust)
    opts.numClusters=nClust;
end


% Determine input type [Note: Not used in present version]
if isfield(opts, 'isDistanceMatrix') && opts.isDistanceMatrix
    Dmat = data;
    BT = size(Dmat, 1);
    % Infer B and T if possible (assume square matrix)
    if isfield(opts, 'B') && isfield(opts, 'T')
        B = opts.B;
        T = opts.T;
    else
        error('When using a distance matrix, opts.B and opts.T must be specified.');
    end
    N = NaN;  % Not needed for distance matrix
    X = [];   % No raw data
elseif strcmpi(opts.method,'kmeans') || strcmpi(opts.method,'kmedoids')
    % Raw data
    [B, N, T] = size(data);
    BT = B * T;
    X = reshape(permute(data, [1 3 2]), BT, N);  % [BT x N]
    Dmat=[]; %skip distance calculation
else
    % Raw data
    [B, N, T] = size(data);
    BT = B * T;
    X = reshape(permute(data, [1 3 2]), BT, N);  % [BT x N]
    %Dmat = pdist(X, opts.distancemethod);
    Dmat=[]; %skip distance calculation
end

%Note: Not used in current version. kmeans/kmedoids needs to recalculate
%distances iteratively
% if strcmpi(opts.distancemethod,'correlation') && strcmpi(opts.method,'agglomerative')
%     Dmat=pdist(X, opts.distancemethod);
%     [Dmat,zMax]=covertCor2Z(1-Dmat);
% end

% Run clustering
clustinfo = struct();
switch lower(opts.method)
    case 'kmeans'
        if isempty(X), error('kmeans requires raw data, not distance matrix.'); end
        [idx, C, Cdist] = kmeans(X, opts.numClusters,...
            'Distance', opts.distancemethod, ...
            'Replicates', opts.replicates,'MaxIter',opts.MaxIter);
        clustinfo.centers = C;
        clustinfo.centerdist = Cdist;
        clustinfo.WCSS = sum(Cdist);

    case 'kmedoids' 
        if isempty(X), error('kmedoids requires raw data, not distance matrix.'); end
        [idx, C, Cdist] = kmedoids(X, opts.numClusters,...
            'Distance', opts.distancemethod, ...
            'Replicates', opts.replicates,'MaxIter',opts.MaxIter);
        clustinfo.centers = C;
        clustinfo.centerdist = Cdist;
        clustinfo.WCSS = sum(Cdist);

        %%Some unused clustering methods removed for code repo upload

    otherwise
        error('Unknown clustering method: %s', opts.method);
end

% Handle noise labels (-1) from DBSCAN
K = max(idx);
if any(idx < 1)
    warning('Some points are marked as noise/unclustered');
%     idx(idx < 1) = K + 1;
%     K = K + 1;
end

% Reshape labels back to B x T
clusterLabels = reshape(idx, B, T);

% Compute B x K likelihood matrix
binLikelihoods = zeros(B, K);
for b = 1:B
    labels = clusterLabels(b, :);
    for k = 1:K
        binLikelihoods(b, k) = sum(labels == k) / T;
    end
end


% Pack info
info = struct();
info.method = opts.method;
info.numClusters = K;
info.parameters = opts;
info.clustinfo = clustinfo;

end
