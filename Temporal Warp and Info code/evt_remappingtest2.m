function [remapstruct]=evt_remappingtest2(ratemat1,ratemat2,iter,evttype1,evttype2,occ1,occ2,calcS)
%EVT_REMAPPINGTEST - Determines the similarity of ratemaps across two 
%conditions, and determines significance using Monte Carlo simulations that 
%shuffle the condition identity of each trial. Calculates information-
%gained by considering the conditions separately rather than as an 
%aggregate, and calculates similarity metrics.
%
%Note: This code assumes that the bins are in time and equal in length, as
%they are with a perievent raster.  NaNs are treated as a lack of occupancy
%with respect to information.
%
%ratemat1 - NxM matrix. N perievent bins, M events. Spike rates.
%ratemat2 - NxR matrix. N perievent bins, R events. Spike rates.
%iter - integer. Number of Monte Carlo simulations
%evttype1 - 1xM vector. M events. Index of event types. Events of each type
%   will be similarly split across conditions during simulations.
%evttype2 - 1xR vector.
%occ1 - NxM matrix. Occupancy of each bin for condition 1.
%occ2 - NxR matrix. Occupancy of each bin for condition 2.
%calcS - boolean.  Calculate silhouette score.
%
%Jon Rueckemann 2023

if nargin<4 || isempty(evttype1)
    evttype1=ones(size(ratemat1,2),1); %all events are in homogenous group
end
if nargin<5 || isempty(evttype2)
    evttype2=ones(size(ratemat2,2),1); %all events are in homogenous group
end
if nargin<6 || isempty(occ1)
    occ1=ones(size(ratemat1)); %all perievent time is equally represented
end
if nargin<7 || isempty(occ2)
    occ2=ones(size(ratemat2)); %all perievent time is equally represented
end
if nargin<8 || isempty(calcS)
    calcS=false;
end


%Create indices of the variety of event types across the conditions
allevttype=[evttype1(:);evttype2(:)];
[~,~,uidx]=unique(allevttype);
uidx=uidx(:);
ttidx={uidx(1:numel(evttype1));uidx((numel(evttype1)+1):end)};

%Calculate information gain for true data
[infogain,corP,corS,cosdist,dotdist,CH,DB,S]=...
    skaggsinfogain(ratemat1,ratemat2,occ1,occ2,calcS,ttidx);


%Set up the Monte Carlo simulations that randomize condition (e.g. maze ID)
allrate=[ratemat1 ratemat2];
allocc=[occ1 occ2];
max_typeidx=max(uidx);
tally_cond1=accumarray(uidx(1:size(ratemat1,2)),1,[max_typeidx 1]);
    %Tally the amount of each trial type in the first condition, so each
    %simulated condition can match that total

%Randomly select events across conditions within each event type to create
%artificial 'sessions' that can be compared to the true condition division
%(preserving the ratio of event type in each Monte Carlo simulation)
evtidx=cell(max_typeidx,1);
iteridx=cell(max_typeidx,1);
for n=1:max_typeidx
    curtypeidx=repmat(find(uidx==n),1,iter); %Find all cur evt type indices
    [~,temp]=sort(rand(size(curtypeidx)),1); %Columns of random index order
    evtidx{n}=curtypeidx(temp<=tally_cond1(n)); %Get rand 1st cond row idx
    iteridx{n}=cumsum(ones(tally_cond1(n),iter),2); %Make 1st cond col idx
    iteridx{n}=iteridx{n}(:);
end
evtidx=cell2mat(evtidx);
iteridx=cell2mat(iteridx);
rndcond1idx=false(numel(allevttype),iter); %logical indexing matrix
rndcond1idx(sub2ind(size(rndcond1idx),evtidx,iteridx))=true; %ID 1st cond
rndcond1idx=rndcond1idx'; %iter x evts

%allrate=[ratemat1 ratemat2];
%ratemat1 - NxM matrix; N perievent bins, M events. Spike rates.
%ratemat2 - NxR matrix. N perievent bins, R events. Spike rates.

%Test info gain via Monte Carlo simulation randomizing condition
[infogain_MC,corP_MC,corS_MC,cosdist_MC,dotdist_MC]=deal(nan(iter,1));
[CH_MC,DB_MC,S_MC]=deal(nan(iter,max(uidx)));
for t=1:iter
    currte1=allrate(:,rndcond1idx(t,:));
    currte2=allrate(:,~rndcond1idx(t,:));
    curocc1=allocc(:,rndcond1idx(t,:));
    curocc2=allocc(:,~rndcond1idx(t,:));
    curtt={uidx(rndcond1idx(t,:)); uidx(~rndcond1idx(t,:))};
    [infogain_MC(t),corP_MC(t),corS_MC(t),cosdist_MC(t),dotdist_MC(t),...
        CH_MC(t,:),DB_MC(t,:),S_MC(t,:)]=...
        skaggsinfogain(currte1,currte2,curocc1,curocc2,calcS,curtt);
end

%Package output into struct
remapstruct=struct('InfoGain',infogain,'PearsonCorr',corP,...
    'SpearmanCorr',corS,'CosineDist',cosdist,'DotProduct',dotdist,...
    'CHindex',CH,'DBindex',DB,'SilhouetteScore',S,...
    'InfoGain_MC',infogain_MC,'PearsonCorr_MC',corP_MC,...
    'SpearmanCorr_MC',corS_MC,'CosineDist_MC',cosdist_MC,...
    'DotProduct_MC',dotdist_MC,'CHindex_MC',CH_MC,'DBindex_MC',DB_MC,...
    'SilhouetteScore_MC',S_MC);
end


function [infogain,corP,corS,cosdist,dotdist,CH,DB,S]=...
    skaggsinfogain(ratemat1,ratemat2,occ1,occ2,calcS,ttidx)

%Concatenate the different trial types into one long vector for Skaggs calc
n_tt=max(cell2mat(ttidx));
[mu_ratemat1,mu_ratemat2,mu_ratematCMB,totocc1,totocc2,totoccCMB]=...
    deal(cell(n_tt,1));
for n=1:n_tt
    mu_ratemat1{n}=mean(ratemat1(:,ttidx{1}==n),2,'omitnan');
    mu_ratemat2{n}=mean(ratemat2(:,ttidx{2}==n),2,'omitnan');
    mu_ratematCMB{n}=mean(...
        [ratemat1(:,ttidx{1}==n) ratemat2(:,ttidx{2}==n)],2,'omitnan');
    totocc1{n}=sum(~isnan(occ1(:,ttidx{1}==n)),2);
    totocc2{n}=sum(~isnan(occ2(:,ttidx{2}==n)),2);
    totoccCMB{n}=sum(~isnan([occ1(:,ttidx{1}==n) occ2(:,ttidx{2}==n)]),2);
end
mu_ratemat1=cell2mat(mu_ratemat1);
mu_ratemat2=cell2mat(mu_ratemat2);
mu_ratematCMB=cell2mat(mu_ratematCMB);
totocc1=cell2mat(totocc1);
totocc2=cell2mat(totocc2);
totoccCMB=cell2mat(totoccCMB);


%Remove indices that do not have calculated rates in both conditions
badidx=any(isnan([mu_ratemat1 mu_ratemat2]),2);
mu_ratemat1=mu_ratemat1(~badidx);
mu_ratemat2=mu_ratemat2(~badidx);
mu_ratematCMB=mu_ratematCMB(~badidx);
totocc1=totocc1(~badidx,:);
totocc2=totocc2(~badidx,:);
totoccCMB=totoccCMB(~badidx,:);

if any(isempty(mu_ratematCMB)|isempty(mu_ratemat1)|isempty(mu_ratemat2))
    %warning('rate mat is empty')
    [infogain,corP,corS,cosdist,dotdist]=deal(nan(1));
    [CH,DB,S]=deal(nan(1,n_tt));
    return
end

%Weight by duration of each condition
time1=sum(~isnan(ratemat1(:)));
time2=sum(~isnan(ratemat2(:)));
w1=sum(time1,"all","omitnan")./(time1+time2);
w2=sum(time2,"all","omitnan")./(time1+time2);

%Calculate skaggs info for each condition and the "session" without partion
sk1=skaggsinfocalc(mu_ratemat1,totocc1);
sk2=skaggsinfocalc(mu_ratemat2,totocc2);
sktot=skaggsinfocalc(mu_ratematCMB,totoccCMB);

%Information gained is the weighted average of information
infogain=sk1*w1+sk2*w2-sktot;

%Calculate correlations between condition mean ratemaps
corP=corr(mu_ratemat1,mu_ratemat2,'type','Pearson','rows','complete');
corS=corr(mu_ratemat1,mu_ratemat2,'type','Spearman','rows','complete');

%Calculate dot product and cosine distance between condition mean ratemaps
dotdist=mu_ratemat1'*mu_ratemat2;
cosdist=1-(dotdist./(norm(mu_ratemat1)*norm(mu_ratemat2)));

%Calculate clustering metrics on rates of each event clustered by condition
allratemat=[ratemat1'; ratemat2'];
clusteridx=[ones(size(ratemat1,2),1);ones(size(ratemat2,2),1)*2];
uidx=cell2mat(ttidx);
[CH,DB,S]=deal(nan(1,max(uidx)));
for n=1:max(uidx)    
    curmat=allratemat(uidx==n,:);
    curmat=curmat(:,~any(isnan(curmat),1));
    if isempty(curmat)
        continue
    end
    CHout=evalclusters(curmat,clusteridx(uidx==n),'CalinskiHarabasz');
    CH(n)=CHout.CriterionValues; %Calinski-Harabasz index
    DBout=evalclusters(curmat,clusteridx(uidx==n),'DaviesBouldin');
    DB(n)=DBout.CriterionValues; %Davies-Bouldin index

    %Silhouette score
    %(note: often nearly anticorrelated with DB and therefore redundant for
    % good separation. Correlation decreases as separation gets worse, but 
    % it is unclear whether the Silhouette score is worth a 10X computation 
    % time.)
    if calcS
        S(n)=mean(silhouette(curmat,clusteridx(uidx==n)));
    end
end
end

function [skaggsinfo]=skaggsinfocalc(ratemap,Px)
Px=Px./sum(Px);
meanspk=mean(ratemap,"all",'omitnan');

%Calculate Skaggs Information Score
skaggsinfo=sum(Px.*ratemap.*log2(ratemap./meanspk),'omitnan')./meanspk;
end