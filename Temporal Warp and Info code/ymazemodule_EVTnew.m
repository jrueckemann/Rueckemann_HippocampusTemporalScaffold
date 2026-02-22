function [spkstruct,err,errmsg,errID]=ymazemodule_EVTnew(unitstruct,evtdata,timetemplate,iter,savefolder,varargin)
%
%Jon Rueckemann 2024

if nargin<4 || isempty(iter)
    iter=1000;
end
if nargin<5
    savefolder=[]; %empty will not incrementally save data
end


%Parse variable (optional) input
p=inputParser;
validScalarPosNum=@(x) isnumeric(x) && isscalar(x) && (x > 0);
validVector=@(x) isnumeric(x) && isvector(x);

addParameter(p,'dt',0.05,validScalarPosNum);
addParameter(p,'maxtrial',25,validScalarPosNum);
addParameter(p,'TTidx',2,validScalarPosNum); %TrialData column. Col 1=correct path; Col 2=chosen path
addParameter(p,'combTT',false,@(x) islogical(x)); %combine trialtypes for info
addParameter(p,'onlycorrect',true,@(x) islogical(x)); %only correct trials
addParameter(p,'scalefactor',1,validVector); %bin spike counts for info
addParameter(p,'k',0,validVector); %bin count for gaussian smoothing kernel


parse(p,varargin{:});
dt=p.Results.dt; %bin width
maxtrial=p.Results.maxtrial;%maximum duration of trial considered-true time
TTidx=p.Results.TTidx; %column index for trial type matrix
combTT=p.Results.combTT; %combine trialtypes
onlycorrect=p.Results.onlycorrect; %only correct trials
scalefactor=p.Results.scalefactor; %bin spike counts for info
k=p.Results.k;  %bin count for gaussian smoothing kernel; Skaggs & sparsity


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


%Extract event names from 'timetemplate' and create event-specific binning
evtname=timetemplate.EventName;
if strcmpi(evtname(1),evtname(end))
    evtname=evtname(1:end-1);
end
n_evt=numel(evtname);
evtbounds=timetemplate.EventBounds;
bindur=timetemplate.bindur;
evtbins=cellfun(@(x) x(1):bindur:x(2),evtbounds,'uni',0);
wnd=cellfun(@(x) [min(x);(max(x)*(1+eps))],evtbins,'uni',0);
    %wnd is (near) identical to evtbounds; syntax for compatibility
evtfieldnames=cellfun(@(x) ['Y_' x],evtname,'uni',0);


%Preallocate a field in the output struct for each event
for s=1:n_spk
    for v=1:n_evt
        spkstruct(s).(evtfieldnames{v})=[];
        spkstruct(s).([evtfieldnames{v} '_Remap'])=[];
    end
end

%Iterate through mazes
n_mazes=numel(evtdata);
badmaze=false(n_mazes,1);
binrate=cell(n_spk,n_mazes,n_evt);
occ=cell(1,n_mazes,n_evt);
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


    %Build 'evtts' from evtdata    
    fn=fieldnames(evtdata{m});
    [~,strctidx]=ismember(evtname,fn);
    evtts=reshape(struct2cell(evtdata{m}),1,[]);
    evtts=evtts(:,strctidx);


    %Prepare trial type designations
    curTT=evtdata{m}.TrialData;
    if onlycorrect
        validtrl{m}=strcmpi(num2cell(curTT(:,3)),'C');
    else
        validtrl{m}=true(size(curTT,1));
    end
    if ~isempty(maxtrial) && maxtrial<inf
        validtrl{m}=validtrl{m}&trldur<maxtrial;
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

    %Iterate through events
    for v=1:n_evt
        %Modify struct for current event
        evtprefix=struct('Task','Ymaze','EventName',evtfieldnames{v});
        evtprefix=repmat(evtprefix,n_spk,1);
        curevt_Yinfo=mergestruct(evtprefix,curM_Yinfo);
        evtsuffix=struct('TaskMat',evtdata{m}.UnityFile,...
            'EventBins',evtbins{v});
        evtsuffix=repmat(evtsuffix,n_spk,1);
        curevt_Yinfo=mergestruct(curevt_Yinfo,evtsuffix);


        %Calculate peri-event information
        [istruct,binrate(:,m,v),occ{1,m,v}]=infoEVT_MC3(spk,evtts{v},...
            evtbins{v},dt,iter,wnd{v},evtcond{m},evtgrp,scalefactor,k);


        %Integrate struct of event information into spkstruct
        curevt_Yinfo=mergestruct(curevt_Yinfo,istruct);
        for s=1:n_spk
            if isempty(spkstruct(s).(evtfieldnames{v})) %1st valid maze
                spkstruct(s).(evtfieldnames{v})=curevt_Yinfo(s);
            else
                spkstruct(s).(evtfieldnames{v})(m)=curevt_Yinfo(s);
            end
        end
    end
end

% %Compare activity across mazes (remapping)
% if ~all(badmaze)
%     for v=1:n_evt
%         estruct=struct('EventName',evtfieldnames{v},...
%             'EventBins',evtbins{v});
%         for s=1:n_spk
%             rstruct=evt_remappingprocessing2(binrate(s,:,v),occ(1,:,v),...
%                 evtcond,mazename,evtbins{v},validtrl,iter,false);
%             rstruct=mergestruct(estruct,rstruct);
%             rstruct=mergestruct(Yinfo(s),rstruct);
%             spkstruct(s).([evtfieldnames{v} '_Remap'])=rstruct;
%         end
%     end
% end

%Save data for this session
if ~isempty(savefolder)
    [~,daylabel]=fileparts(spkstruct(1).Folder);
    savefilename=['YmazeEVTinfoMC_' daylabel '.mat'];
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