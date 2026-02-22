function [aggborders,aggpeakval,boundarymats,sortmats,sortdir,clustertemplate]=createboundarymats(resultstruct,sortfields,maxrep)
%
%resultstruct - FxW struct array.  Each element is a the processed output
%           from the boundary extraction code.  Rows should reflect source
%           files that should be concatenated in a singular boundary
%           matrix, i.e. from the same source BxNxT pop vec matrix.
%           Columns reflect different save files from different parameter
%           sets. Each element/parameter set contains multiple replications
%
%sortfields - cell array of strings specifying the fields used the sort the
%           boundary results in the boundary matrices
%
%
%Jon Rueckemann 2025


if nargin<2
    %sortfields=[];
    sortfields={'resolution','knn','nclust'};
end
if nargin<3
    maxrep=[];
end


%Collect data from files (contain multiple repetitions per parameter set)
[filebordermat,filesortmats,sortdir,fileclustertemplate]=deal(cell(size(resultstruct)));
for m=1:numel(resultstruct)
    
    %Extract border matrices
    if isempty(maxrep)
        filebordermat{m}=resultstruct(m).border_matrix;
        fileclustertemplate{m}=resultstruct(m).mode_cluster_template;
    else
        filebordermat{m}=resultstruct(m).border_matrix(1:maxrep,:);
        fileclustertemplate{m}=resultstruct(m).mode_cluster_template(1:maxrep,:);        
    end


    %Extract data from specified fields for ordering border matrices 
    if ~isempty(sortfields)
        [filesortmats{m},sortdir]=...
            extractSortData(resultstruct(m),sortfields,maxrep);
    end
end


%Aggregate data originating from the same population vector matrix and
%reorder based on the specified sort parameters
[aggborders,boundarymats,sortmats,clustertemplate]=...
    deal(cell(size(filebordermat,1),1));
aggpeakval=nan(size(filebordermat,1),1);
for n=1:size(filebordermat,1)

    %Combine boundary data
    curborder=filebordermat(n,:);
    n_trialtype=size(curborder{1},2);
    boundarymats{n}=cell(1,n_trialtype);
    aggborders{n}=cell(1,n_trialtype);
    clustertemplate{n}=cell(1,n_trialtype);
    
    for s=1:n_trialtype
        curborder_trialtype=cellfun(@(x) x(:,s),curborder,'uni',0);
        curborder_trialtype=[curborder_trialtype{:}];
        curborder_trialtype=curborder_trialtype(:);
        boundarymats{n}{s}=curborder_trialtype;

        curtemplate=cellfun(@(x) x(:,s),fileclustertemplate(n,:),'uni',0);
        curtemplate=[curtemplate{:}];
        curtemplate=curtemplate(:);
        curtemplate=cellfun(@(x) x',curtemplate,'uni',0);
        clustertemplate{n}{s}=curtemplate;
    end

    %Reorder data based on the specified fields
    if ~isempty(sortfields)
        %Combine ordering data
        cursortmat=filesortmats(n,:);
        cursortmat=cell2mat(cursortmat(:));

        %Reorder the boundary matrices based on the reordering fields
        [sortmats{n},sortidx]=sortrows(cursortmat,sortdir);

        for s=1:n_trialtype
            boundarymats{n}{s}=boundarymats{n}{s}(sortidx);
            clustertemplate{n}{s}=clustertemplate{n}{s}(sortidx);
        end
    end

    %Create summary across repetitions of parameter set (ie within file)
    maxval=nan(1,n_trialtype);
    for s=1:n_trialtype
        curbound=boundarymats{n}{s};
        curbound=cellfun(@(x) sum(x,1),curbound,'uni',0);
        aggborders{n}{s}=mean(cell2mat(curbound(:)));
        maxval(s)=max(aggborders{n}{s});
    end
    aggpeakval(n)=max(maxval);
end
end


%% Helper function

function [sortdata,sortdir]=extractSortData(curdata,sortfields,maxrep)
%Extract and reformat the data used for ordering boundary results
%Creates a numerical matrix of ordering data

if isempty(maxrep)
    n_rep=size(curdata.border_matrix,1);
else
    n_rep=maxrep;
end
n_sortfield=numel(sortfields);

sortdata=nan(n_rep,n_sortfield); %preallocate output
sortdir=cell(1,n_sortfield);
for n=1:n_sortfield
    %Extract from top level fields or use meta data
    switch sortfields{n}
        case 'quality'
            if isempty(maxrep)
                sortdata(:,n)=curdata.quality;
            else
                sortdata(:,n)=curdata.quality(1:maxrep);
            end
            sortdir{n}='ascend';
        case 'nclust'
            if isempty(maxrep)
                sortdata(:,n)=curdata.nclust;
            else
                sortdata(:,n)=curdata.nclust(1:maxrep);
            end
            sortdir{n}='ascend';
        case 'seeds'
            if isempty(maxrep)
                sortdata(:,n)=curdata.seeds;
            else
                sortdata(:,n)=curdata.seeds(1:maxrep);
            end
            sortdir{n}='ascend';
        case 'knn'
            sortdata(:,n)=repmat(curdata.knn,n_rep,1);
            sortdir{n}='descend';
        otherwise
            curmeta=curdata.meta;
            cursortval=curmeta.(sortfields{n});
            sortdata(:,n)=repmat(cursortval,n_rep,1);
            sortdir{n}='ascend';
    end
end
end