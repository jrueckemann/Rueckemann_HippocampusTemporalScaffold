function [resultstruct_Inull,resultstruct_Xnull]=mass_clustersimilarity_NullRes(A,Bfilename,mapFineToCoarse,skipRelabeling,report_both,compactoutput)
%
%
%Jon Rueckemann 2025




%Find resolution information across files in B (to match with A)
mfile=matfile(Bfilename);
B=mfile.null_leiden;
res_B=[[B.knn].' [B.resolution].'];
%res_B=cellfun(@(x) unique(x),{B.nclust}.');  %kMeans alternative -- but this may not work how I want
clear B;

%Within-resolution comparison
resultstruct_Inull=rmfield(A,'labels');
tmp=cell(size(A));
[resultstruct_Inull.similarity]=tmp{:};
[resultstruct_Inull.similarity_pre]=tmp{:};
[resultstruct_Inull.sim_info]=tmp{:};
n_trueres=numel(A);
for n=1:n_trueres %Iterate through files in A
    disp(A(n).filename);
    
    %Current resolution
    label1=A(n).labels;
    res_A=[A(n).knn A(n).resolution];
    %res_A=unique(A(n).nclust); %kMeans alternative -- but this may not work how I want


    %Find matches to the current resolution
    curidx=find(all(res_B==res_A,2));
    n_null=numel(curidx);

    %Preallocate
    [sim_pre,sim_post,sim_info]=deal(cell(n_null,1));
    for k=1:n_null
        disp(['Processing ' num2str(k) ' of ' num2str(n_null) ' nulls.']);


        curB=mfile.null_leiden(curidx(k),1);

        if report_both
            [sim_post{k},sim_info{k},sim_pre{k}]=...
                parameter_consistency_extended_Xcomp(label1,curB.labels,...
                mapFineToCoarse,skipRelabeling,report_both,compactoutput);
        else
            [sim_post{k},sim_info{k}]=...
                parameter_consistency_extended_Xcomp(label1,curB.labels,...
                mapFineToCoarse,skipRelabeling,report_both,compactoutput);
        end
    end
    resultstruct_Inull(n).similarity=cell2mat(sim_post);
    resultstruct_Inull(n).sim_info=cell2mat(sim_info);
    resultstruct_Inull(n).similarity_pre=cell2mat(sim_pre);
end


%Cross-resolution comparison
cmb=nchoosek(1:n_trueres,2);
cmb=[cmb; fliplr(cmb)]; %repeat w/ inverse to represent each true-null pair
resultstruct_Xnull=struct('CompareIdx',[],'Idx1_resolution',[],'Idx1_knn',[],...
    'Idx2_resolution',[],'Idx2_knn',[],'similarity',[],'similarity_pre',[],...
    'sim_info',[]);
resultstruct_Xnull=repmat(resultstruct_Xnull,size(cmb,1),1);
for n=1:size(cmb,1)

    disp(['Processing ' num2str(n) ' of ' num2str(size(cmb,1)) ' pairs.']);

    %Populate cross-comparison result struct
    curidx1=cmb(n,1);
    curidx2=cmb(n,2);


    %Update resolution information
    resultstruct_Xnull(n).CompareIdx=num2str(cmb(n,:));
    resultstruct_Xnull(n).Idx1_resolution=A(curidx1).resolution;
    resultstruct_Xnull(n).Idx1_knn=A(curidx1).knn;
    resultstruct_Xnull(n).Idx2_resolution=A(curidx2).resolution;
    resultstruct_Xnull(n).Idx2_knn=A(curidx2).knn;  

    
    %Current A resolution    
    label1=A(curidx1).labels;
    
    %Find matches to the current B resolution
    cur_res=[A(curidx2).knn A(curidx2).resolution];
    curBidx=find(all(res_B==cur_res,2));
    n_null=numel(curBidx);



    %Preallocate
    [sim_pre,sim_post,sim_info]=deal(cell(n_null,1));
    for k=1:n_null
        disp(['Processing ' num2str(k) ' of ' num2str(n_null) ' nulls.']);

        curB=mfile.null_leiden(curBidx(k),1);

        if report_both
            [sim_post{k},sim_info{k},sim_pre{k}]=...
                parameter_consistency_extended_Xcomp(label1,curB.labels,...
                mapFineToCoarse,skipRelabeling,report_both,compactoutput);
        else
            [sim_post{k},sim_info{k}]=...
                parameter_consistency_extended_Xcomp(label1,curB.labels,...
                mapFineToCoarse,skipRelabeling,report_both,compactoutput);
        end
    end
    resultstruct_Xnull(n).similarity=cell2mat(sim_post);
    resultstruct_Xnull(n).sim_info=cell2mat(sim_info);
    resultstruct_Xnull(n).similarity_pre=cell2mat(sim_pre);
end

end