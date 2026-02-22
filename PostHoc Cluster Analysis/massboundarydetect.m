function [resultstruct,settings]=massboundarydetect(labelfolder,B)
% MASSBOUNDARYDETECT  Batch boundary detection for clustered label files.
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

if nargin<1 || isempty(labelfolder)
    labelfolder=uigetdir;
end

%Iterate through folder files - each a specific parameter combination
filenames=dir(fullfile(labelfolder,'*.h5'));
filenames=cellfun(@(x,y) fullfile(x,y),{filenames.folder},...
    {filenames.name},'uni',0)';

resultstruct=struct('filename',[],'quality',[], ...
    'meta',[],'nclust',[],'seeds',[],'border_matrix',[], ...
    'mode_cluster_template',[]);
resultstruct=repmat(resultstruct,numel(filenames),1);

for n=1:numel(filenames)
    disp(['File #' num2str(n) ': ' filenames{n}]);

    %Load file
    [labels, quality, meta,~, nclust, seeds]=load_parallel_leiden_hdf5(filenames{n});

    resultstruct(n).filename=filenames{n};
    resultstruct(n).quality=quality;
    resultstruct(n).meta=meta;
    resultstruct(n).nclust=nclust;
    resultstruct(n).seeds=seeds;
    resultstruct(n).knn=str2num(regexp(filenames{n},...
        '(?<=_k)\d+(?=__)','match','once')); %#ok<ST2NM>

    
    %Find boundaries for each repetition of the parameter combination
    [n_samples,n_repetitions]=size(labels);
    n_samples=floor(n_samples./2);

    %Preallocate
    [bordermat,modetemplate]=deal(cell(n_repetitions,2));
    for m=1:n_repetitions
        %Sample each trial type independently
        for s=1:2
            curidx=(1:n_samples)+n_samples*(s-1);
            L=reshape(labels(curidx,m),B,[])+1;

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