function [resultstruct_I,resultstruct_X]=mass_clustersimilarity_True(A,mapFineToCoarse,skipRelabeling,report_both,compactoutput)
%
%
%Jon Rueckemann 2025


%Within-resolution comparison 
resultstruct_I=rmfield(A,'labels');
tmp=cell(size(A));
[resultstruct_I.similarity]=tmp{:};
[resultstruct_I.similarity_pre]=tmp{:};
[resultstruct_I.sim_info]=tmp{:};
n_trueres=numel(A);
for n=1:n_trueres %Iterate through files in A
    disp(A(n).filename);

    labels=A(n).labels;
    if report_both
        [resultstruct_I(n).similarity,resultstruct_I(n).sim_info,...
            resultstruct_I(n).similarity_pre] = ...
            parameter_consistency_extended_V2(labels,mapFineToCoarse,...
            skipRelabeling,report_both,compactoutput);
    else
        [resultstruct_I(n).similarity,resultstruct_I(n).sim_info] = ...
            parameter_consistency_extended_V2(labels,mapFineToCoarse,...
            skipRelabeling,report_both,compactoutput);
    end

end

%Cross-resolution comparison
cmb=nchoosek(1:n_trueres,2);
resultstruct_X=struct('CompareIdx',[],'Idx1_resolution',[],'Idx1_knn',[],...
    'Idx2_resolution',[],'Idx2_knn',[],'similarity',[],'similarity_pre',[],...
    'sim_info',[]);
resultstruct_X=repmat(resultstruct_X,size(cmb,1),1);
for n=1:size(cmb,1)

    disp(['Processing ' num2str(n) ' of ' num2str(size(cmb,1)) ' pairs.']);

    %Populate cross-comparison result struct
    curidx1=cmb(n,1);
    curidx2=cmb(n,2);


    %Update resolution information
    resultstruct_X(n).CompareIdx=num2str(cmb(n,:));
    resultstruct_X(n).Idx1_resolution=A(curidx1).resolution;
    resultstruct_X(n).Idx1_knn=A(curidx1).knn;
    resultstruct_X(n).Idx2_resolution=A(curidx2).resolution;
    resultstruct_X(n).Idx2_knn=A(curidx2).knn;  

    
    labels1=A(curidx1).labels;
    labels2=A(curidx2).labels;
    if report_both
        [resultstruct_X(n).similarity,resultstruct_X(n).sim_info,...
            resultstruct_X(n).similarity_pre] = ...
            parameter_consistency_extended_Xcomp(labels1,labels2,...
            mapFineToCoarse,skipRelabeling,report_both,compactoutput);
    else
        [resultstruct_X(n).similarity,resultstruct_X(n).sim_info] = ...
            parameter_consistency_extended_Xcomp(labels1,labels2,...
            mapFineToCoarse,skipRelabeling,report_both,compactoutput);
    end


end
end