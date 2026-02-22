function [Allremapstruct]=evt_remappingprocessing2(ratemats,occ,evttypes,condID,condbins,validevt,iter,calcS,droplastevt)
%
%Jon Rueckemann 2023


if nargin<9 || isempty(droplastevt)
    droplastevt=true;
end

validevt(cellfun(@isempty,ratemats))={[]};
if droplastevt
    for v=1:numel(validevt)
        if ~isempty(validevt{v})
            validevt{v}(end)=false;
        end
    end
end

%Reformat data
[ratemats,matdiv]=cellfun(@(x) reshapearray(x),ratemats,'uni',0);
ratemats=cellfun(@(x,y) x(:,y),ratemats,validevt,'uni',0);
[n_bins,maxidx]=max(cellfun(@numel,matdiv));
matdiv=matdiv{maxidx};
[occ,~]=cellfun(@(x) reshapearray(x),occ,'uni',0);
occ=cellfun(@(x,y) x(:,y),occ,validevt,'uni',0);
evttypes=cellfun(@(x,y) x(y),evttypes,validevt,'uni',0);

%Prepare output
[infogain,corP,corS,cosdist,dotdist]=deal(nan(numel(ratemats)));
[CH,DB,S,...
    infogain_MC,corP_MC,corS_MC,cosdist_MC,dotdist_MC,CH_MC,DB_MC,S_MC]=...
    deal(cell(numel(ratemats),numel(ratemats)));

if numel(ratemats)>1
    %Identify all combinations of condition comparisons
    cmb=nchoosek(1:numel(ratemats),2);

    %Iterate through combinations
    for m=1:size(cmb,1)
        idx1=cmb(m,1);
        idx2=cmb(m,2);
        rm1=ratemats{idx1};
        rm2=ratemats{idx2};
        if isempty(rm1) || isempty(rm2) || ...
                all(isnan(rm1),"all") ||all(isnan(rm2),"all") || ...
                size(rm1,1)~=n_bins || size(rm2,1)~=n_bins
            continue
        end
        remapstruct=evt_remappingtest2(rm1,rm2,iter,...
            evttypes{idx1},evttypes{idx2},occ{idx1},occ{idx2},...
            matdiv,calcS);

        %Reform true data into matrix (only lower triangle)
        infogain(idx2,idx1)=remapstruct.InfoGain;
        corP(idx2,idx1)=remapstruct.PearsonCorr;
        corS(idx2,idx1)=remapstruct.SpearmanCorr;
        cosdist(idx2,idx1)=remapstruct.CosineDist;
        dotdist(idx2,idx1)=remapstruct.DotProduct;
        CH{idx2,idx1}=remapstruct.CHindex;
        DB{idx2,idx1}=remapstruct.DBindex;
        S{idx2,idx1}=remapstruct.SilhouetteScore;

        %Reform simulation data into matrix (only lower triangle)
        infogain_MC{idx2,idx1}=remapstruct.InfoGain_MC;
        corP_MC{idx2,idx1}=remapstruct.PearsonCorr_MC;
        corS_MC{idx2,idx1}=remapstruct.SpearmanCorr_MC;
        cosdist_MC{idx2,idx1}=remapstruct.CosineDist_MC;
        dotdist_MC{idx2,idx1}=remapstruct.DotProduct_MC;
        CH_MC{idx2,idx1}=remapstruct.CHindex_MC;
        DB_MC{idx2,idx1}=remapstruct.DBindex_MC;
        S_MC{idx2,idx1}=remapstruct.SilhouetteScore_MC;
    end
end

%Output data for plotting
ratemaps=cellfun(@(x) mean(x,2,'omitnan'),ratemats,'uni',0);
Allremapstruct=struct('ConditionName',{condID},'Ratemaps',{ratemaps},...
    'ConditionBins',{condbins},'InfoGain',infogain,...
    'PearsonCorr',corP,'SpearmanCorr',corS,...
    'CosineDist',cosdist,'DotProduct',dotdist,...
    'CHindex',{CH},'DBindex',{DB},'SilhouetteScore',{S},...
    'InfoGain_MC',{infogain_MC},'PearsonCorr_MC',{corP_MC},...
    'SpearmanCorr_MC',{corS_MC},'CosineDist_MC',{cosdist_MC},...
    'DotProduct_MC',{dotdist_MC},'CHindex_MC',{CH_MC},...
    'DBindex_MC',{DB_MC},'SilhouetteScore_MC',{S_MC});
end

function [Y,matdiv]=reshapearray(X)
Y=num2cell(X,[1 2]);
Y=Y(:);
Y=cell2mat(Y);

[n_pred,~,n_cond]=size(X);
matdiv=zeros(n_pred*n_cond,1);
matdiv(1:n_pred:end)=1;
matdiv=cumsum(matdiv);
end