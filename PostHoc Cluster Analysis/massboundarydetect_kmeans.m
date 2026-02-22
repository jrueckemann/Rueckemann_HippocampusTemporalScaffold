function [resultstruct,settings]=massboundarydetect_kmeans(inputstruct)
% MASSBOUNDARYDETECT  Batch boundary detection for k-means clustered label 
% structs.
%
% Summary
%   Scans a folder of *.hd5 HDF5 outputs (e.g., from a Leiden/graph
%   clustering pipeline), loads per-repetition labelings, and calls
%   TRIAL_CLUSTER_BORDERS on each repetition and trial type to detect
%   temporal cluster boundaries. Results and per-file metadata are returned
%   in a structured format for downstream analysis.
%
% Inputs
%   labelfolder  (char/string)  Path to a directory containing *.hd5 files.
%
%   B            (scalar int)   Number of time bins per trial. Used to
%                               reshape linear label vectors into a B×T
%                               label matrix (bins × trials) before boundary
%                               detection.
%
% Outputs
%   resultstruct (struct)       Aggregated outputs with fields:
%       .filenames            : absolute paths to *.hd5 files
%       .quality              : per-replicate quality info 
%       .meta                 : per-file metadata struct
%       .nclust               : per-replicate cluster counts 
%       .seeds                : per-replicate RNG/seed info 
%       .border_matrix        : R×2 cell array where:
%                               border_matrix {r,s} contains the boundary
%                               distribution PxB matrix for repetition r 
%                               and trial type s.  (borderdist)                             
%       .mode_cluster_template: R×2 cell array where:
%                               mode_cluster_template {r,s} is the 
%                               B×1 mode cluster template for repetition r, 
%                               trial type s.
%
%   settings     (struct)       Parameters used internally:
%       .func                     : 'massboundarydetect.m'
%       .buffer                   : search window around template borders
%       .usesecondarymode         : (bool) allow secondary modal cluster
%       .secondarymodethreshold   : (0–1) fraction of trials required
%       .secondarymode_minadjacency: minimum adjacent bins 
%
%Jon Rueckemann 2025

wnd=3;
use_spt=false;
spt_thrsh=0.35;
min_adj=3;

settings=struct('func','massboundarydetect.m','buffer',wnd,...
    'usesecondarymode',use_spt,'secondarymodethreshold',spt_thrsh,...
    'secondarymode_minadjacency',min_adj);



%Iterate through folder files - each a specific parameter combination
resultstruct=struct('filename',[],'quality',[], ...
    'meta',[],'nclust',[],'seeds',[],'border_matrix',[], ...
    'mode_cluster_template',[]);
resultstruct=repmat(resultstruct,numel(inputstruct),1);

for n=1:numel(inputstruct)
    disp(n);

    resultstruct(n).filename='kmeansstruct';
    resultstruct(n).knn=nan;

    labels=inputstruct(n).clusterLabels;
    % labels=cell2mat(...
    %     cellfun(@(x) x(:),resultstruct(n).clusterLabels,'uni',0));
    resultstruct(n).meta=cell2mat(inputstruct(n).info);

    n_sorts=numel(inputstruct(n).info);
    resultstruct(n).seeds=zeros(n_sorts,1);
    nclust=nan(n_sorts,1);
    quality=nan(n_sorts,1);
    for m=1:n_sorts
        nclust(m)=inputstruct(n).info{m}.numClusters;
        quality(m)=inputstruct(n).info{m}.clustinfo.WCSS;
    end
    resultstruct(n).quality=quality;
    resultstruct(n).nclust=nclust;

    
    %Find boundaries for each repetition of the parameter combination
    [n_repetitions]=numel(labels);
    

    %Preallocate
    [bordermat,modetemplate]=deal(cell(n_repetitions,2));
    for m=1:n_repetitions
        n_samples=floor(size(labels{m},2)./2);

        %Sample each trial type independently
        for s=1:2
            curidx=(1:n_samples)+n_samples*(s-1);
            L=labels{m}(:,curidx);

            %Run boundary detection on each trial type
            [bordermat{m,s},bstruct,~]=...
                trial_cluster_borders(L,wnd,use_spt,spt_thrsh,min_adj);
            modetemplate{m,s}=bstruct.clustertemplate;
        end

        %Store results
        resultstruct(n).border_matrix=bordermat;
        resultstruct(n).mode_cluster_template=modetemplate;
    end
end
end