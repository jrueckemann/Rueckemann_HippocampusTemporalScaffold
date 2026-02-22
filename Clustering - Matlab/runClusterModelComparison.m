function [results, errmsg] = runClusterModelComparison(data, k_vals, clustmeth, TT, opts)
% RUNCLUSTERMODELCOMPARISON
% Performs trial-wise clustering using specified methods and k values,
% aligns cluster labels temporally, and computes per-trial-type likelihoods.
%
% INPUTS:
%   data      - [B x N x T] or [observations x features] matrix
%   k_vals    - array of integers, k values to evaluate (e.g., [5 6 7])
%   clustmeth - string or cell array of clustering method names
%   TT        - [B X T] matrix of trial type labels (1-based)
%   opts      - struct with clustering options (method, replicates, etc.)
%
% OUTPUTS:
%   results   - struct with likelihoods, labels, centers, and TT
%   errmsg    - cell array of error messages for each [k x method]
%
% Jon Rueckemann, 2025

if nargin < 2 || isempty(k_vals)
    k_vals = [5 6 7 8 9 10 11];
end
if nargin < 3 || isempty(clustmeth)
    clustmeth = {'kmeans'};
elseif ischar(clustmeth)
    clustmeth = {clustmeth};
end
[B, N, T] = size(data);
if nargin < 4 || isempty(TT)
    TT = ones(B,T);
end
if nargin < 5
    opts = struct();
end

%Prepare trial type indexing
[~,~,unqTTidx]=unique(TT(:));
n_types=max(unqTTidx);
unqTTidx=reshape(unqTTidx,size(TT));
  

%Run clustering
n_meth = numel(clustmeth);
n_k = numel(k_vals);
[binLikelihoods, binLikelihoods_all, clusterLabels, info, S, errmsg] = ...
    deal(cell(n_k, n_meth));
for m = 1:n_meth
    method = clustmeth{m};
    disp(['Method: ', method]);
    local_opts = opts;
    local_opts.method = method;

    for n = 1:n_k
        k = k_vals(n);
        disp(['  k = ', num2str(k)]);
        try
            [tmp_binLikelihoods, clusterLabels{n,m}, info{n,m}] = ...
                clusterTrialwiseActivity(data, k, local_opts);

            % Create cluster centers if missing
            if ~isempty(info{n,m}) && ~isfield(info{n,m}.clustinfo,'centers')
                C = nan(k,N);
                for c = 1:k
                    [b_id,t_id]=ind2sub([B,T],find(clusterLabels{n,m}==c));
                    points = zeros(numel(b_id), N);

                    for p = 1:numel(b_id)
                        points(p, :) = data(b_id(p), :, t_id(p));
                    end
                    C(c,:) = mean(points, 1, 'omitnan');  % [1 x N]
                end
                info{n,m}.clustinfo.centers = C;
            end

            % Align clusters to temporal order
            if ~isempty(info{n,m})
                [label_map, apply_map_fn, ~] = align_clusters_by_temporal_order(tmp_binLikelihoods,true);
                clusterLabels{n,m} = apply_map_fn(clusterLabels{n,m});
                
                % Your correct logic: assign aligned values into reordered space
                binLikelihoods_all{n,m} = nan(size(tmp_binLikelihoods));
                binLikelihoods_all{n,m}(:,label_map) = tmp_binLikelihoods;

                info{n,m}.clustinfo.centers(label_map,:) = info{n,m}.clustinfo.centers;
                if isfield(info{n,m}.clustinfo, 'centerdist')
                    info{n,m}.clustinfo.centerdist(label_map,:) = info{n,m}.clustinfo.centerdist;
                end
            else
                binLikelihoods_all{n,m} = tmp_binLikelihoods;
            end

            % Compute trial-type averaged bin likelihoods
            n_clusters = size(binLikelihoods_all{n,m}, 2);
            binLikelihoods{n,m} = zeros(B, n_clusters, n_types);
            for p = 1:n_types
                curTT=unqTTidx==p;
                for c = 1:n_clusters
                    binLikelihoods{n,m}(:,c,p)=...
                        mean(clusterLabels{n,m}==c & curTT,2);
                end
            end

            errmsg{n,m} = '';
        catch ME
            errmsg{n,m} = ME.message;
            disp(['Error: ', ME.message]);
            binLikelihoods{n,m} = [];
            binLikelihoods_all{n,m} = [];
            clusterLabels{n,m} = [];
            info{n,m} = [];
        end        
    end
end


results = struct();
results.binLikelihoods = binLikelihoods;
results.binLikelihoods_all = binLikelihoods_all;
results.clusterLabels = clusterLabels;
results.trialtype = TT;
results.Xbincorrelation = corr(mean(data,3)',mean(data,3)');
results.info = info;
end
