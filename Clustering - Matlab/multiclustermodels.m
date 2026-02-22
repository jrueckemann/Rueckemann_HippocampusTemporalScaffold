function [results, errmsg, opts] = multiclustermodels(data, clustmeth, rotatetrl, opts)
% MULTICLUSTERMODELS
% Runs multiple clustering models on neural population data.
%
% INPUTS:
%   data      - BxNxT matrix of binned firing rates (time x neurons x trials), 
%               or a 1x2 cell array of such matrices (two trial types).
%   clustmeth - String or cell array of clustering methods (e.g., 'kmeans', {'kmeans','optics'}).
%   rotatetrl - (Optional) Logical flag to rotate trial order (default: false).
%   opts      - (Optional) Struct or array of structs with clustering parameters.
%
% OUTPUTS:
%   results   - Cell array containing results of each clustering method.
%   errmsg    - Cell array of error messages for each method (empty if successful).
%
% Jon Rueckemann, 2025

if iscell(data) && numel(data) == 1
    data=data{1};
end
assert((iscell(data) && numel(data) == 2) || ~iscell(data), ...
    'data must be a BxNxT matrix or a 2-element cell array of matrices');

if nargin < 2 || isempty(clustmeth), clustmeth = {'kmeans'}; end
if ischar(clustmeth), clustmeth = {clustmeth}; end
if nargin < 3 || isempty(rotatetrl), rotatetrl = false; end

%'kvals',[5 6 7 8 9 10 11 12 13 14],
% 'kvals',[6 7 8 9 10 11 12],
defaultOpts = struct('clustmeth','kmeans','kvals',[6 7 8 9 10 11 12],...
    'eps',0.7,'minpts',30,'isDistanceMatrix',false,'replicates',100,...
    'MaxIter',1000,'distancemethod','correlation',...
    'linkmethod','average','B',[],'T',[]);

if nargin < 4 || isempty(opts)
    opts = repmat(defaultOpts, numel(clustmeth), 1);
    for m = 1:numel(clustmeth)
        opts(m).clustmeth = clustmeth{m};
    end
else
    % Fill in missing fields in opts
    optFields = fieldnames(defaultOpts);
    for w=1:numel(opts)
        for m = 1:numel(optFields)
            if ~isfield(opts, optFields{m}) || isempty(opts(w).(optFields{m}))
                opts(w).(optFields{m}) = defaultOpts.(optFields{m});
            end
        end
    end
end


if islogical(rotatetrl)
    if rotatetrl
        if iscell(data)
            [data{1},data{2},rotoffset]=rotatetrialdata(data{1},'both',data{2});
        else
            [data,~,rotoffset] = rotatetrialdata(data, 'train');
        end
    else
        rotoffset=[];
    end
else
    rotoffset=rotatetrl;
    if iscell(data)
        [data{1},data{2},rotoffset]=rotatetrialdata(data{1},'both',data{2},rotoffset);
    else
        [data,~,rotoffset] = rotatetrialdata(data, 'train',rotoffset);
    end
end

if iscell(data)
    TT = [ones(size(data{1},[1 3]))  2*ones(size(data{2},[1 3]))];
    data = cat(3, data{1}, data{2});
else
    TT = ones(size(data,[1 3]));
end

[B,~,T] = size(data);

for m = 1:numel(opts)
    if isempty(opts(m).B), opts(m).B = B; end
    if isempty(opts(m).T), opts(m).T = T; end
end


[results, errmsg] = deal(cell(size(opts)));
tme=tic;
for m = 1:numel(opts)
    cmeth = opts(m).clustmeth;
    try
        [results{m},errmsg{m}]=runClusterModelComparison(data,opts(m).kvals,cmeth,TT,opts(m));
        results{m}.rotoffest=rotoffset;
    catch ME
        results{m} = [];
        errmsg{m} = ME.message;
        warning('Method %s failed: %s', cmeth, errmsg{m});
    end
end
toc(tme);
end
