function [labels, quality, meta, filepath, nclust, seeds] = load_parallel_leiden_hdf5(filepath)
% load_leiden_hdf5: Loads leiden community results from HDF5 file.
%
%   [labels, quality, meta] = load_leiden_hdf5(filepath)
%
%   INPUT
%     filepath   - Full path to the HDF5 file saved by Python
%
%   OUTPUTS
%     labels, quality - Struct with fields like 'k10_n5', each a [T*B x k] matrix
%     meta       - Struct with metadata from the file
%
%Jon Rueckemann 2025


if nargin<1 || isempty(filepath)
    [fname,fdir]=uigetfile('*.h5','Choose *.h5 file');
    filepath=fullfile(fdir,fname);
end
    % --- Load metadata ---
    meta_info = h5info(filepath, '/meta');
    meta = struct();
    meta.filepath=filepath;
    
    for m = 1:length(meta_info.Datasets)
        name = meta_info.Datasets(m).Name;
        data = h5read(filepath, ['/meta/', name]);
        if iscell(data)
            data=data{1};
        end
        meta.(name) = data;
    end

    for m = 1:length(meta_info.Attributes)
        name = meta_info.Attributes(m).Name;
        val = meta_info.Attributes(m).Value;
        meta.(name) = val;
    end

    
    % --- Load data ---
    labels = h5read(filepath, '/results/labels');
    quality = h5read(filepath, '/results/quality');
    nclust = h5read(filepath, '/results/n_communities');
    seeds = h5read(filepath, '/results/seeds');
end