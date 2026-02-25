function [spkstruct,Pstruct,err,errmsg,errID]=ymazemodule_warp2(unitstruct,evtdata,timetemplate,iter,savefolder,saveplotdata,varargin)
%
%Jon Rueckemann 2024

if nargin<4 || isempty(iter)
    iter=1000;
end
if nargin<5
    savefolder=[]; %empty will not incrementally save data
end
if nargin<6
    saveplotdata=false; %kept for backwards compatability
end

Pstruct=[]; %kept for backwards compatability

%Parse variable (optional) input
p=inputParser;
validScalarPosNum=@(x) isnumeric(x) && isscalar(x) && (x > 0);
validVector=@(x) isnumeric(x) && isvector(x);
validCellVector=@(x) iscell(x) && all(cellfun(@(y) isvector(y),x,'uni',1));

addParameter(p,'dt',0.05,validScalarPosNum);
addParameter(p,'maxtrial',25,validScalarPosNum);
addParameter(p,'bins2d',{225:0.5:265;195:0.5:235},validCellVector);
addParameter(p,'TTidx',2,validScalarPosNum); %TrialData column. Col 1=correct path; Col 2=chosen path
addParameter(p,'combTT',false,@(x) islogical(x)); %combine trialtypes for info
addParameter(p,'onlycorrect',true,@(x) islogical(x)); %only correct trials
addParameter(p,'reqvalidwarp',false,@(x) islogical(x)); %only valid warping
addParameter(p,'scalefactor',[1 2 5],validVector); %bin spike counts for info
addParameter(p,'k',3,validVector); %bin count for gaussian smoothing kernel
addParameter(p,'modendevt',true,@(x) islogical(x)); %trial end event comes from the next trial
addParameter(p,'checkITI',true,@(x) islogical(x)); %force ITI to be 4 sec if trials overlap


parse(p,varargin{:});
dt=p.Results.dt; %bin width
maxtrial=p.Results.maxtrial;%maximum duration of trial considered-true time
bins2d=p.Results.bins2d;
TTidx=p.Results.TTidx; %column index for trial type matrix
combTT=p.Results.combTT; %combine trialtypes
onlycorrect=p.Results.onlycorrect; %only correct trials
reqvalidwarp=p.Results.reqvalidwarp; %only include trials w/ valid warping
scalefactor=p.Results.scalefactor; %bin spike counts for info
k=p.Results.k;  %bin count for gaussian smoothing kernel; Skaggs & sparsity
modendevt=p.Results.modendevt;  %trial end event comes from the next trial
checkITI=p.Results.checkITI;  %force ITI to be 4 sec if trials overlap

%Prepare output
err=false(size(unitstruct));
errmsg=cell(size(unitstruct));
errID=cell(size(unitstruct));

% try
%Construct output struct elements using basic info from each neuron
n_spk=numel(unitstruct);
spk={unitstruct.spkts};
spkstruct=rmfield(unitstruct,'spkts');
curdate=spkstruct(1).Date;

f={'StUnitID','Subject','Date','Electrode','UnitID','Folder'};
s_fn=fieldnames(unitstruct);
keepidx=cellfun(@(x) any(strcmpi(x,f)),s_fn);
s_fn=s_fn(keepidx);
s_val=struct2cell(unitstruct);
s_val=s_val(keepidx,:);
Yinfo=cell2struct(s_val,s_fn,1);

mz_fn={'TaskFile','TaskMaze','TaskDate','TaskTime','TaskDuration',...
    'TaskMedianTrialDuration','TaskMeanRate','TaskN_spk'}';
mz_val={[],[],[],[],nan,nan,nan,nan}';
MZ=cell2struct(mz_val,mz_fn,1);

%Preallocate the output structs
for s=1:n_spk
    spkstruct(s).Y_warp=[];
    spkstruct(s).Y_warp_Remap=[];
end

%Iterate through mazes
n_mazes=numel(evtdata);
badmaze=false(n_mazes,1);
binrate=cell(n_spk,n_mazes);
occ=cell(1,n_mazes);
evtcond=cell(1,n_mazes);
mazename=cell(1,n_mazes);
validtrl=cell(1,n_mazes);
for m=1:n_mazes
    if isempty(evtdata{m})
        badmaze(m)=true;
        continue
    end

    %Update MZ struct for current maze
    tme=evtdata{m}.xydata(:,1)./1000;
    curMZ=MZ;
    curMZ.TaskFile=evtdata{m}.UnityFile;
    curMZ.TaskMaze=evtdata{m}.MazeName;
    mazename{m}=evtdata{m}.MazeName;
    curMZ.TaskDate=datestr(floor(datenum(curdate)));
    tmp=split(evtdata{m}.UnityFile,'_');
    tmp=[tmp{end-2} ':' tmp{end-1} ':' tmp{end}(1:end-4)];
    curMZ.TaskTime=datestr(tmp,'HH:MM:SS');
    curMZ.TaskDuration=range(tme);
    trlstrt=evtdata{m}.TrialStart(:);
    trldur=diff([trlstrt; tme(end)]);
    curMZ.TaskMedianTrialDuration=median(trldur);

    %Neuron-specific spiking statistics for current maze
    curMZ=repmat(curMZ,n_spk,1);
    for s=1:n_spk
        curMZ(s).TaskN_spk=sum(spk{s}>tme(1)&spk{s}<tme(end));
        curMZ(s).TaskMeanRate=curMZ(s).TaskN_spk./curMZ(s).TaskDuration;
    end
    curM_Yinfo=mergestruct(Yinfo,curMZ);

    
    %Define time bins for calculating information and rate maps
    binedges=min(tme):dt:max(tme); %bin time spanning the current maze
    binctr=movmean(binedges,2);
    binctr=binctr(2:end);


    %Build 'evtmatrix' from evtdata
    evtname=timetemplate.EventName;
    curTT=evtdata{m}.TrialData;
    if strcmpi(evtname(1),evtname(end))
        evtname=evtname(1:end-1);
        modendevt=true;

        fn=fieldnames(evtdata{m});
        [~,strctidx]=ismember(evtname,fn);
        evtmatrix=reshape(struct2cell(evtdata{m}),1,[]);
        evtmatrix=cell2mat(evtmatrix(:,strctidx));

        %Add start event from next trial to end of trial
        evtmatrix=[evtmatrix [evtmatrix(2:end,1);nan]]; %#ok<AGROW>
        evtmatrix(end,end)=evtmatrix(end,end-1) + ...
            mean(diff(evtmatrix(:,[end-1 end]),1,2),1,"omitnan");
    elseif modendevt
        fn=fieldnames(evtdata{m});
        [~,strctidx]=ismember(evtname,fn);
        evtmatrix=reshape(struct2cell(evtdata{m}),1,[]);
        evtmatrix=cell2mat(evtmatrix(:,strctidx));

        %Use event from next trial to create end of trial
        evtmatrix=[evtmatrix(:,1:end-1) [evtmatrix(2:end,end);nan]];
        evtmatrix(end,end)=evtmatrix(end,end-1) + ...
            mean(diff(evtmatrix(:,[end-1 end]),1,2),1,"omitnan");
    else
        fn=fieldnames(evtdata{m});
        [~,strctidx]=ismember(evtname,fn);
        evtmatrix=reshape(struct2cell(evtdata{m}),1,[]);
        evtmatrix=cell2mat(evtmatrix(:,strctidx));
    end
    if checkITI
        if any(evtmatrix(2:end,1)<evtmatrix(1:(end-1),end))
            warning('Event matrix events caused trial overlap')
            warning('Replacing all correct trials with a 4 second ITI')
            warning('and incorrect trials get a 3 second ITI')
            evtmatrix(:,end)=evtmatrix(:,end-1)+3+...
                strcmpi(num2cell(curTT(:,3)),'C');
            %Deals with an occassional minor misalignment of trial start
            %and movement start
        end
    end
%     assert(~any(evtmatrix(2:end,1)<evtmatrix(1:(end-1),end)),...
%         'Trial end events of evtmatrix are ill conditioned.')
%     assert(all(~(diff(evtmatrix,1,2)<0),"all"),...
%         'Events of evtmatrix are ill conditioned.')
    if any(evtmatrix(2:end,1)<evtmatrix(1:(end-1),end))
        warning('Trial end events of evtmatrix are ill conditioned.')
        badmaze(m)=true;
        continue
    end
    if ~all(~(diff(evtmatrix,1,2)<0),"all")
        warning('Events of evtmatrix are ill conditioned.')
        badmaze(m)=true;
        continue
    end


    %Determine the warped-time predictor index
    [predID,trlidx,validwarp,warpts,knotmap,epochmap,predmap]=...
        warptrial_fulltrial4info(evtmatrix,timetemplate,binctr);
%     [predID,trlidx,validwarp,warpts,knotmap,epochmap,predmap]=...
%         timewarppred4info(evtmatrix,timetemplate,binctr);
    predID=predID(:);
    trlidx=trlidx(:);    
    if all(isnan(trlidx))
        badmaze(m)=true;
        continue
    end
    %POSSIBLY ADD SPIKES AS AN INPUT FOR MORE PRECISE INTERPOLATION


    %Prepare trial type designations    
    if onlycorrect
        validtrl{m}=strcmpi(num2cell(curTT(:,3)),'C');
    else
        validtrl{m}=true(size(curTT,1));
    end
    if ~isempty(maxtrial) && maxtrial<inf
        validtrl{m}=validtrl{m}&trldur<maxtrial;
    end
    if reqvalidwarp
        validtrl{m}=validtrl{m}&validwarp;
    end

    [unqTT,~,mzTT]=unique(num2cell(curTT(:,TTidx)));
    assert(all(strcmpi(unqTT,'L')|strcmpi(unqTT,'R')), ...
        'Unexpected Trial Type');
    if ~combTT
        %Test that trial types are present after invalid trials are removed
        if numel(unique(mzTT))~=numel(unique(mzTT(validtrl{m})))
            badmaze(m)=true;
            continue
        else
            evtcond{m}=mzTT;
        end
    else
        evtcond{m}=ones(size(mzTT));
    end
    
    %Create identifiers that create groups of 1 trial for each of of the 
    % conditions. Used to ensure even distribution of condition types when
    % splitting the trials for reliability measures.
    evtgrp=nan(size(mzTT));
    for q=1:max(mzTT)
        evtgrp(mzTT==q)=(1:sum(mzTT==q))';
    end

%Trial Indexing is now produced identically to predID
%     %Designate a trial identity for each time bin
%     trlidx=zeros(numel(binedges)-1,1);
%     for st=1:numel(trlstrt)
%         trlidx(find(binedges(1:end-1)>=trlstrt(st),1,"first"))=1;
%     end
%     trlidx=cumsum(trlidx);
%     %trlidx=interp1(trlstrt,1:numel(trlstrt),binedges(1:end-1),'previous');

    %Determine which indices should not be included in analysis
    dropidx=false(size(trlidx));
    for n=1:max(trlidx)
        dropidx(trlidx==n)=~validtrl{m}(n);
    end
    dropidx=isnan(trlidx) | isnan(predID) | dropidx;


    %Modify struct for current event
    evtprefix=struct('Task','Ymaze','EventName','Y_warp');
    evtprefix=repmat(evtprefix,n_spk,1);
    curevt_Yinfo=mergestruct(evtprefix,curM_Yinfo);
    evtsuffix=struct('TaskMat',evtdata{m}.UnityFile,...
        'WarpTimeBins',warpts,'EpochMap',epochmap,'KnotMap',knotmap); 
    evtsuffix=repmat(evtsuffix,n_spk,1);
    curevt_Yinfo=mergestruct(curevt_Yinfo,evtsuffix);


    %Calculate peri-event information
    [istruct,binrate(:,m),occ{m}]=info_MC2(spk,binedges,trlidx,predID,...
        dropidx,iter,evtcond{m},evtgrp,scalefactor,k);

    %Integrate struct of event information into spkstruct
    curevt_Yinfo=mergestruct(curevt_Yinfo,istruct);
    for s=1:n_spk
        if isempty(spkstruct(s).Y_warp) %1st valid maze
            spkstruct(s).Y_warp=curevt_Yinfo(s);
        else
            spkstruct(s).Y_warp(m)=curevt_Yinfo(s);
        end
    end 
end

%Compare activity across mazes (remapping)
if ~all(badmaze)
    estruct=struct('EventName','Y_warp','WarpTimeBins',warpts);    
    for s=1:n_spk
        rstruct=evt_remappingprocessing2(binrate(s,:),occ,evtcond,...
            mazename,warpts,validtrl,iter,false);
        rstruct=mergestruct(estruct,rstruct);
        rstruct=mergestruct(Yinfo(s),rstruct);
        spkstruct(s).Y_warp_Remap=rstruct;
    end
end

%Save data for this session
if ~isempty(savefolder)
    [~,daylabel]=fileparts(spkstruct(1).Folder);
    savefilename=['WarpYmazeMC_' daylabel '.mat'];
    savefilename=fullfile(savefolder,savefilename);
    parforsave(savefilename,spkstruct);
end

% catch ME
%     err=true;
%     errmsg=ME.message;
%     errID=ME.identifier;
%     disp('Error Message:')
%     disp(ME.message)
% end
end

function []=parforsave(savefilename,spkstruct)
save(savefilename,'spkstruct','-v7.3');
end