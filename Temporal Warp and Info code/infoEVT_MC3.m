function [infostruct,trlbinrate,occ_out,spk_evttme,spk_evtidx]=infoEVT_MC3(spk,evtts,evtbins,spkdt,iter,wnd,evtcond,evtgrp,scalefactor,k)
%
%(Modified version of info_MC2.  Retooled for perievent activity and allows
% overlapping perievent windows. 2025 - added support for instantaneous 
% firing rate bins)
%
%spk - Nx1 array of spike timestamps
%evtts - Mx1 array of event timestamps
%evtbins - 1xT bin edges used for creating predictor identities used for 
%   calculating information and rate maps
%spkdt - scalar.  Bin duration for creating instantaneous firing rates
%iter - integer. Monte Carlo simulation iterations
%wnd - 2xM or 2x1 array defining perievent window for considering spikes
%evtcond - 1xM vector; M events. Index of condition identity for each event
%evtgrp - 1xM vector; M events. Index of group identity for each event.
%   Events within a group will remain together, e.g Contiguous left and
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
%
%
%Jon Rueckemann 2025


%Format input
evtbins=evtbins(:);
if nargin<6 || isempty(wnd)
    wnd=[min(evtbins);(max(evtbins)*(1+eps))];
end

if nargin<7 || isempty(evtcond)
    evtcond=ones(1,numel(evtts)); %each event belongs to the same condition
end
assert(all(evtcond>0) && all(rem(evtcond,1)==0),...
    'EVTCOND must be a vector of positive integers');
evtcond=evtcond(:);
n_cond=max(evtcond);

if nargin<8 || isempty(evtgrp)
    evtgrp=1:numel(evtts); %each event belongs to an independent group
end
evtgrp=evtgrp(:);

if nargin<9 || isempty(scalefactor)
    scalefactor=1; %no spike count binning
end

if nargin<10 || isempty(k)
    k=0; %no gaussian smoothing for Skaggs info or sparsity metrics
end


%Prepare perievent windows around each event
n_evt=numel(evtts);
assert(any(size(wnd)==2),'Window span argument must be a 2xm array')
% if size(wnd,1)~=2 %Disabled to ensure that input is formatted with intent
%     wnd=wnd';
% end
if size(wnd,2)==1
    wnd=repmat(wnd,1,n_evt);
end
assert(n_evt==size(wnd,2),['If window span is specified '...
    'for each event, then must be equal events and windows.']);


%Find spiking within perievent window and convert to perievent time
n_spk=numel(spk);
[spk_evttme,spk_evtidx]=deal(cell(n_spk,1));
for s=1:n_spk %iterate through units
    [spk_evtidx{s},spk_evttme{s}]=time2evt(spk{s},evtts,wnd);
end

%Create temporal bin edges for spikes
n_bins=numel(evtbins)-1;
spkevtbins=min(evtbins):spkdt:max(evtbins);
if max(spkevtbins)~=max(evtbins)
    spkevtbins=[spkevtbins; max(evtbins)]; %cover for uneven divisor
end
spkevtbins=spkevtbins(:);
n_spkbins=numel(spkevtbins)-1;

%Determine occupancy and predictor index for temporal bins in each event
occ=false(n_spkbins,n_evt);
for w=1:n_evt
    occ(:,w)=spkevtbins(1:end-1)>=wnd(1,w)&spkevtbins(2:end)<wnd(2,w);    
end
spkevtbins(end)=spkevtbins(end).*(1+eps); %add buffer for bin edge

evtidx=repmat(1:n_evt,n_spkbins,1);
dt=repmat(reshape(diff(spkevtbins),[],1),1,n_evt); 
            %differs from spkdt if spkdt does not factor max(evtbins)
spkbinctr=movmean(spkevtbins,2);
spkbinctr=spkbinctr(2:end);
predID=discretize(spkbinctr,evtbins);
predID=repmat(predID(:),1,n_evt);


%Modify predictor index for event-specific conditions
predID_info=predID+repmat(n_bins.*(evtcond'-1),n_spkbins,1);


%Drop unoccupied temporal bins and reshape into column vectors
predID=predID(occ);
predID_info=predID_info(occ);
evtidx=evtidx(occ);
dt=dt(occ);

%Convert event classifiers into bin classifiers
bincond=evtcond(evtidx);
bingrp=evtgrp(evtidx);

%Create reindexing scheme to reshape iPos data
reshapesub_iPos=zeros(n_bins*n_cond,1);
reshapesub_iPos(1:n_bins:end)=1;
reshapesub_iPos=cumsum(reshapesub_iPos);
reshapesub_iPos=[repmat((1:n_bins)',n_cond,1) reshapesub_iPos];
reshapeind_iPos=sub2ind([n_bins n_cond],...
    reshapesub_iPos(:,1),reshapesub_iPos(:,2));
null_iPos=nan(n_bins,n_cond,numel(scalefactor));
occ_out=accumarray([predID evtidx bincond],1,[n_bins n_evt n_cond],[],nan);
if k>1
    occ_out=smoothdata(occ_out,1,'gaussian',k);
end

%% Calculate statistics and ratemaps for the true data
%Create results struct
tmp=nan(1,numel(scalefactor)); 
tmp2=nan(n_bins,n_cond,2); 
tmp3=nan(iter,1);
tmp4=nan(iter,numel(scalefactor));

infostruct=struct('Ratemap',nan(n_bins,n_cond),...
    'TrialRatemap',nan(n_bins,n_evt,n_cond),'MaxRate',nan,...
    'SpikeCountRescale',scalefactor,'MutualInfo',tmp,'SkaggsInfo',nan,...
    'Sparsity',nan,'iPos',null_iPos,'iPosmax',tmp,'iPosvar',tmp,...
    'HalfRatemap',tmp2,'HalfRateCorr',nan,'SplitRatemap',tmp2,...
    'SplitRateCorr',nan,'MaxRate_MC',tmp3,'MutualInfo_MC',tmp4,...
    'SkaggsInfo_MC',tmp3,'Sparsity_MC',tmp3,'iPosmax_MC',tmp4,...
    'iPosvar_MC',tmp4,'HalfRateCorr_MC',tmp3,'SplitRateCorr_MC',tmp3);
infostruct=repmat(infostruct,n_spk,1);
trlbinrate=repmat({nan(n_bins,n_evt,n_cond)},n_spk,1);

if isempty(predID(~isnan(predID)))        
    return
end

%Bin spike data into perievent temporal bins and calculate reliability
binspk=nan(numel(predID),n_spk);
for s=1:n_spk
    if isempty(spk_evttme{s})
        continue
    end

    %Bin spike data into temporal bins
    evtbinspk=histcounts2(spk_evttme{s},spk_evtidx{s},spkevtbins,1:n_evt+1);
    binspk(:,s)=evtbinspk(occ);
    if all(binspk(:,s)==0)
        continue
    end

    %Find mean rates for individual trials and the aggregate
    trlbinrate{s}=accumarray([predID evtidx bincond],...
        binspk(:,s)./dt,[n_bins n_evt n_cond],@nanmean,nan); %#ok<*NANMEAN>
    ratemap=accumarray([predID bincond],...
        binspk(:,s)./dt,[n_bins n_cond],@nanmean,nan);
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
        reliablecorr2(predID,binspk(:,s)./dt,bincond,bingrp,k);
end

%Calculate information metrics
iPos=nan(max(predID_info),numel(scalefactor),n_spk);
shannoninfo=nan(n_spk,numel(scalefactor));
for f=1:numel(scalefactor)
    [iPos_tmp,shannoninfo(:,f),Px,condspkmu,UpredID]=...
        Olypherinfo(predID_info,ceil(binspk./scalefactor(f)));
    iPos_tmp=[iPos_tmp;nan(max(predID_info)-size(iPos_tmp,1),n_spk)]; %#ok<AGROW> 
    iPos(:,f,:)=iPos_tmp;
end
if size(iPos,1)~=n_bins*n_cond %Pad to represent all (predictor X cond)
    iPos=[iPos;nan((n_bins*n_cond)-size(iPos,1),numel(scalefactor),n_spk)];
end
[skaggsinfo,sparsity]=infocalc(Px,condspkmu,reshapesub_iPos(UpredID,2),k);

for s=1:n_spk
    infostruct(s).MutualInfo=shannoninfo(s,:);
    infostruct(s).SkaggsInfo=skaggsinfo(s);
    infostruct(s).Sparsity=sparsity(s);

    cur_iPos=null_iPos;
    [iPosmax,iPosvar]=deal(nan(1,numel(scalefactor)));
    for f=1:numel(scalefactor)
        iPos_tmp2=nan(n_bins,n_cond);
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

%Duration of each perievent window
wnddur=diff(wnd);

for s=1:n_spk %Iterate through neurons
    if isempty(spk_evttme{s})
        continue
    end

    %Prepare output
    [halfcorrMC,splitcorrMC]=deal(nan(iter,1));
    [iPosmaxMC,iPosvarMC,shannoninfoMC]=deal(nan(iter,numel(scalefactor)));

    %Create time rotation offset (unique random offsets for each neuron)
    curevtidx=spk_evtidx{s};    
    evtoffset=nan(numel(curevtidx),iter);
    for tt=1:iter
        rndoffset=rand(n_evt,1);
        evtoffset(:,tt)=rndoffset(curevtidx);
        %each column is an iteration of a wnd-relative offset for all
        % spikes within each event
    end

    %Create new spiketrain rotated within the window around each event
    %*Algebraically convenient to rescale time relative window duration
    evt_start=reshape(wnd(1,curevtidx),[],1);
    evt_dur=reshape(wnddur(curevtidx),[],1);
    spk_mc=((spk_evttme{s}-evt_start)./evt_dur)+evtoffset;
    spk_mc(spk_mc>=1)=spk_mc(spk_mc>=1)-1;
    spk_mc=spk_mc.*evt_dur+evt_start;



    %Bin spike data into perievent temporal bins and calculate reliability
    binspk_mc=nan(numel(predID),iter);
    for t=1:iter %Iterate through MC simulations within neuron
        evtbinspk_mc=histcounts2(spk_mc(:,t),curevtidx,spkevtbins,1:n_evt+1);
        binspk_mc(:,t)=evtbinspk_mc(occ);

        %Calculate relability correlations
        [halfcorrMC(t),~,splitcorrMC(t),~]=...
            reliablecorr2(predID,binspk_mc(:,t)./dt,bincond,bingrp,k);
    end
    infostruct(s).HalfRateCorr_MC=halfcorrMC;
    infostruct(s).SplitRateCorr_MC=splitcorrMC;


    %Calculate maximum of the ratemap for each Monte Carlo iteration
    predinfoidx=repmat(predID_info,1,iter);
    iteridx=repmat(1:iter,numel(predID_info),1);
    currates=binspk_mc./dt;
    rm_mc=accumarray([predinfoidx(:) iteridx(:)],currates(:),...
        [n_bins*n_cond iter],@nanmean,nan);
    infostruct(s).MaxRate_MC=max(rm_mc,[],1,'omitnan')';


    %Calculate information metrics
    for f=1:numel(scalefactor)
        [curiPos,shannoninfoMC(:,f),~,condspkmu_tmp,UpredID_tmp]=...
            Olypherinfo(predID_info,ceil(binspk_mc./scalefactor(f)),Px);

        iPosmaxMC(:,f)=max(curiPos,[],1,"omitnan");
        iPosvarMC(:,f)=var(curiPos,0,1,"omitnan");
    end
    infostruct(s).MutualInfo_MC=shannoninfoMC;
    infostruct(s).iPosmax_MC=iPosmaxMC;
    infostruct(s).iPosvar_MC=iPosvarMC;

    [infostruct(s).SkaggsInfo_MC,infostruct(s).Sparsity_MC]=...
        infocalc(Px,condspkmu_tmp,reshapesub_iPos(UpredID_tmp,2),k);

end
end
