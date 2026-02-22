function [infostruct,trlbinrate,occ]=info_MC2(spk,bins,trlidx,predID,dropidx,iter,trlcond,trlgrp,scalefactor,k,sz)
%
%(Modified version of info_MC and infoEVT_MC2)
%
%spk - Nx1 array of spike timestamps
%bins - 1xT bin edges used for calculating information and rate maps
%trlidx - 1x(T-1) integer trial index number (M=max(trlidx))
%predID - 1x(T-1) integer vector containing indices for the predictor
%iter - integer. Monte Carlo simulation iterations
%trlcond - 1xM vector; M trials. Index of condition identity for each trial
%trlgrp - 1xM vector; M trials. Index of group identity for each trial.
%   Trials within a group will remain together, e.g Contiguous left and
%   right trials will remain paired if given the same event group number.
%scalefactor - integer.  Scale factor for binning spike counts.
%   A scale factor of 2 parses the spike count in bins of [0,2,4,6,...]
%   when calculating mutual information based scores, which helps to 
%   counteract underestimating information by amplifying the non-meaningful 
%   differences in spike count.
%k - integer.  Number of bins used to create gaussian smoothing kernel.
%   Uses the 'smoothdata' function.  Affects Skaggs info, sparsity, and 
%   reliability measures, and also changes trlbinrate, occ, and mean 
%   ratemap -- smoothes over NaNs.  Does not affect mutual info or iPos.
%sz - 2x1 integer. Dimensions of 2D matrix shaping predID. Empty avoids 2D
%   smoothing.
%
%
%Jon Rueckemann 2024

%#ok<*NANMEAN> 

%Format input
n_bins=numel(bins)-1;
if nargin<5 || isempty(dropidx)
    dropidx=false(n_bins,1);
end

if nargin<6 || isempty(iter)
    iter=1000;
end

if nargin<7 || isempty(trlcond)
    trlcond=ones(1,max(trlidx)); %each event belongs to the same condition
end
assert(all(trlcond>0) && all(rem(trlcond,1)==0),...
    'EVTCOND must be a vector of positive integers');
trlcond=trlcond(:);
[~,~,trlcond]=unique(trlcond);
n_cond=max(trlcond);

if nargin<8 || isempty(trlgrp)
    trlgrp=1:max(trlidx); %each event belongs to an independent group
end
trlgrp=trlgrp(:);

if nargin<9 || isempty(scalefactor)
    scalefactor=1; %no spike count binning
end

if nargin<10 || isempty(k)
    k=0; %no gaussian smoothing for Skaggs info or sparsity metrics
end
if nargin<11 || isempty(sz)
    sz=[]; %Dimensions of 2D matrix shaping predID.
end

%Format input
bins=bins(:);
trlidx=trlidx(:);
predID=predID(:);
dropidx=dropidx(:);
trlcond=trlcond(:);
trlgrp=trlgrp(:);
n_spk=numel(spk);
n_trl=max(trlidx,[],'omitnan');
n_pred=max(predID,[],'omitnan');
dt=diff(bins);
bincond=nan(n_bins,1);
bingrp=nan(n_bins,1);
for t=1:n_trl
    bincond(trlidx==t)=trlcond(t);
    bingrp(trlidx==t)=trlgrp(t);
end

%Modify predID to account for different trial types in information calc
predID_info=predID+n_pred*(bincond-1);
reshapesub_iPos=zeros(n_pred*n_cond,1);
reshapesub_iPos(1:n_pred:end)=1;
reshapesub_iPos=cumsum(reshapesub_iPos);
reshapesub_iPos=[repmat((1:n_pred)',n_cond,1) reshapesub_iPos];
reshapeind_iPos=sub2ind([n_pred n_cond],...
    reshapesub_iPos(:,1),reshapesub_iPos(:,2));
null_iPos=nan(n_pred,n_cond,numel(scalefactor));
occ=accumarray([predID(~dropidx) trlidx(~dropidx) bincond(~dropidx)],...
    1,[n_pred n_trl n_cond],[],nan);
if k>1
    occ=smoothdata(occ,1,'gaussian',k);
end

%% Calculate statistics and ratemaps for the true data
%Create results struct
tmp=nan(1,numel(scalefactor)); 
tmp2=nan(n_pred,n_cond,2); 
tmp3=nan(iter,1);
tmp4=nan(iter,numel(scalefactor));

infostruct=struct('Ratemap',nan(n_pred,n_cond),...
    'TrialRatemap',nan(n_pred,n_trl,n_cond),'MaxRate',nan,...
    'SpikeCountRescale',scalefactor,'MutualInfo',tmp,'SkaggsInfo',nan,...
    'Sparsity',nan,'iPos',null_iPos,'iPosmax',tmp,'iPosvar',tmp,...
    'HalfRatemap',tmp2,'HalfRateCorr',nan,'SplitRatemap',tmp2,...
    'SplitRateCorr',nan,'MaxRate_MC',tmp3,'MutualInfo_MC',tmp4,...
    'SkaggsInfo_MC',tmp3,'Sparsity_MC',tmp3,'iPosmax_MC',tmp4,...
    'iPosvar_MC',tmp4,'HalfRateCorr_MC',tmp3,'SplitRateCorr_MC',tmp3);
infostruct=repmat(infostruct,n_spk,1);
trlbinrate=repmat({nan(n_pred,n_trl,n_cond)},n_spk,1);

if isempty(predID(~dropidx & ~isnan(predID)))        
    return
end

%Analyze temporally binned spike data relative to warped time predictor
binspk=nan(n_bins,n_spk);
for s=1:n_spk
    if isempty(spk{s})
        continue
    end

    %Bin spike data into temporal bins
    binspk(:,s)=histcounts(spk{s},bins);
    if all(binspk(:,s)==0)
        infostruct(s).TrialRatemap=zeros(n_pred,n_trl,n_cond);    
        infostruct(s).Ratemap=zeros(n_pred,n_cond);
        infostruct(s).MaxRate=0;
        continue
    end

    %Find mean rates for individual trials and the aggregate
    trlbinrate{s}=accumarray(...
        [predID(~dropidx) trlidx(~dropidx) bincond(~dropidx)],...
        binspk(~dropidx,s)./dt(~dropidx),[n_pred n_trl n_cond],@nanmean,nan);
    ratemap=accumarray([predID(~dropidx) bincond(~dropidx)],...
        binspk(~dropidx,s)./dt(~dropidx),[n_pred n_cond],@nanmean,nan);
    if k>1
        trlbinrate{s}=smoothdata(trlbinrate{s},1,'gaussian',k);
        ratemap=smoothdata(ratemap,1,'gaussian',k);
    end
    infostruct(s).TrialRatemap=trlbinrate{s};    
    infostruct(s).Ratemap=ratemap;
    infostruct(s).MaxRate=max(ratemap,[],'all','omitnan');

    %Calculate relability correlations
    [infostruct(s).HalfRateCorr,infostruct(s).HalfRatemap,...
        infostruct(s).SplitRateCorr,infostruct(s).SplitRatemap]=...
        reliablecorr2(predID(~dropidx),binspk(~dropidx,s)./dt(~dropidx),...
        bincond(~dropidx),bingrp(~dropidx),k);
end

% TROUBLESHOOT NaNs; Only occur in sessions with <10 trials as of 12/03/24
tmprmap={infostruct.Ratemap};
testrmapnan=cellfun(@(x) any(isnan(x(:))),tmprmap);
if any(testrmapnan&~cellfun(@isempty,spk))
    warning('Nan problem');
end

%Calculate information metrics
iPos=nan(max(predID_info),numel(scalefactor),n_spk);
shannoninfo=nan(n_spk,numel(scalefactor));
for f=1:numel(scalefactor)
    [iPos_tmp,shannoninfo(:,f),Px,condspkmu,UpredID]=...
        Olypherinfo(predID_info(~dropidx),...
        ceil(binspk(~dropidx,:)./scalefactor(f)));
    iPos_tmp=[iPos_tmp;nan(max(predID_info)-size(iPos_tmp,1),n_spk)]; %#ok<AGROW> 
    iPos(:,f,:)=iPos_tmp;
end
if size(iPos,1)~=n_pred*n_cond %Pad to represent all (predictor X cond)
    iPos=[iPos;nan((n_pred*n_cond)-size(iPos,1),numel(scalefactor),n_spk)];
end
if isempty(sz)
    [skaggsinfo,sparsity]=infocalc(Px,condspkmu,reshapesub_iPos(UpredID,2),k);
else
    [skaggsinfo,sparsity]=infocalc2D(Px,condspkmu,UpredID,n_pred,k,sz);
end

for s=1:n_spk
    infostruct(s).MutualInfo=shannoninfo(s,:);
    infostruct(s).SkaggsInfo=skaggsinfo(s);
    infostruct(s).Sparsity=sparsity(s);

    cur_iPos=null_iPos;
    [iPosmax,iPosvar]=deal(nan(1,numel(scalefactor)));
    for f=1:numel(scalefactor)
        iPos_tmp2=nan(n_pred,n_cond);
        iPos_tmp2(reshapeind_iPos)=iPos(:,f,s);
        cur_iPos(:,:,f)=iPos_tmp2;

        iPosmax(f)=max(iPos_tmp2,[],'all',"omitnan");
        iPosvar(f)=var(iPos_tmp2,0,'all',"omitnan");
    end
    infostruct(s).iPos=cur_iPos;
    infostruct(s).iPosmax=iPosmax;
    infostruct(s).iPosvar=iPosvar;
end

%% Calculate statistics from Monte Carlo simulations for each neuron

%Find the beginning of each trial
[trlnumber,trlstartidx]=unique(trlidx);
trlstartidx=trlstartidx(~isnan(trlnumber));
trlstarts=bins(trlstartidx);
trlstarts=trlstarts(:);
trldur=diff([trlstarts; bins(end)]);

for s=1:n_spk %Iterate through neurons
    if isempty(spk{s}) || all(binspk(:,s)==0)
        continue
    end

    %Determine the trial in which each spike occurs
    maxtrlbin=max(bins(end),trlstarts(end)+0.000000001); 
        %in case of premature recording end
    spk_trlidx=discretize(spk{s},[trlstarts; maxtrlbin]);

    keep=~isnan(spk_trlidx);
    spk_trlidx=spk_trlidx(keep);
    
    if isempty(spk_trlidx)
        continue
    end

    %Determine timing of each spike
    spktrlts=spk{s}(keep)-trlstarts(spk_trlidx);

    %Create new spike rotated within the duration of each trial
    %*Algebraically convenient to rescale time relative to trial onset to 1
    trloffset=rand(n_trl,iter);
    newspk=(spktrlts./trldur(spk_trlidx))+trloffset(spk_trlidx,:);
    newspk(newspk>=1)=newspk(newspk>=1)-1;
    newspk=newspk.*trldur(spk_trlidx)+trlstarts(spk_trlidx);

    curbinspk=nan(n_bins,iter);
    for t=1:iter
        curbinspk(:,t)=histcounts(newspk(:,t),bins);
    end

    %Calculate maximum of the ratemap for each Monte Carlo iteration
    predinfoidx=repmat(predID_info(~dropidx),1,iter);
    iteridx=repmat(1:iter,n_bins,1);
    iteridx=iteridx(~dropidx,:);
    currates=curbinspk(~dropidx,:)./dt(~dropidx);
    rm_mc=accumarray([predinfoidx(:) iteridx(:)],currates(:),...
        [n_pred*n_cond iter],@nanmean,nan);
    infostruct(s).MaxRate_MC=max(rm_mc,[],1,'omitnan')';


    %Calculate relability correlations
    [halfcorrMC,splitcorrMC]=deal(nan(iter,1));
    for t=1:iter
        [halfcorrMC(t),~,splitcorrMC(t),~]=...
            reliablecorr2(predID(~dropidx),...
            curbinspk(~dropidx,t)./dt(~dropidx),...
            bincond(~dropidx),bingrp(~dropidx),k);
    end
    infostruct(s).HalfRateCorr_MC=halfcorrMC;
    infostruct(s).SplitRateCorr_MC=splitcorrMC;


    %Calculate information metrics
    [iPosmaxMC,iPosvarMC,shannoninfoMC]=deal(nan(iter,numel(scalefactor)));
    for f=1:numel(scalefactor)
        [curiPos,shannoninfoMC(:,f),~,condspkmu,UpredID]=...
            Olypherinfo(predID_info(~dropidx),...
            ceil(curbinspk(~dropidx,:)./scalefactor(f)),Px);

        iPosmaxMC(:,f)=max(curiPos,[],1,"omitnan");
        iPosvarMC(:,f)=var(curiPos,0,1,"omitnan");
    end
    infostruct(s).MutualInfo_MC=shannoninfoMC;
    infostruct(s).iPosmax_MC=iPosmaxMC;
    infostruct(s).iPosvar_MC=iPosvarMC;

    if isempty(sz)
        [infostruct(s).SkaggsInfo_MC,infostruct(s).Sparsity_MC]=...
            infocalc(Px,condspkmu,reshapesub_iPos(UpredID,2),k);
    else
        %Inactivated for code repo upload; not germane to paper
        % [infostruct(s).SkaggsInfo_MC,infostruct(s).Sparsity_MC]=...
        %     infocalc2D(Px,condspkmu,UpredID,n_pred,k,sz);
    end
end
end