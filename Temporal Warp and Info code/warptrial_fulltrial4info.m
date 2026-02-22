function [predID,trlidx,validwarp,warpts,knotmap,epochmap,predmap]=warptrial_fulltrial4info(evtmatrix,timetemplate,binctr)
%
%evtmatrix - NxK+1 matrix.  Timestamps of K events across N trials.  Final
%   column is the end of the trial (which may coincide with beginning
%   of next trial). Algorithm assumes evtmatrix is well-conditioned.
%timetemplate - struct.  Definition of time transform across event epochs.
%binctr - Tx1 vector.  Timestamps of center of spike bins to be mapped to
%   predictor.
%
%Jon Rueckemann 2024

%Create a warp mapping for each trial for each inter-event epoch
predID=nan(size(binctr));
trlidx=nan(size(binctr));
[n_trl,n_evt]=size(evtmatrix);
[warpts,knotmap,epochmap,predmap]=deal(cell(1,n_evt-1));

%Create mapping across all event epochs
bindur=timetemplate.bindur;
n_bins=round(timetemplate.duration./bindur);
newdur=n_bins.*bindur; %ensure each epoch duration is a multiple of bindur
for k=1:n_evt-1
    if k==1
        warpts{k}=0:bindur:newdur(k);
    else
        warpts{k}=(0:bindur:newdur(k))+max(warpts{k-1});
    end
    knotmap{k}=nan(size(warpts{k})); %knot template in rescaled time
    epochmap{k}=k*ones(1,numel(warpts{k})-1); 
    predoffset=sum(cellfun(@numel,predmap(1:k)));
    predmap{k}=(1:numel(epochmap{k}))+predoffset;
end
warpts(2:end)=cellfun(@(x) x(2:end),warpts(2:end),'uni',0); %delete overlap
warpts=cell2mat(warpts);
knotmap=cell2mat(knotmap);
epochmap=cell2mat(epochmap);
predmap=cell2mat(predmap);
validwarp=diff(evtmatrix,1,2)>0; %ensure each epoch has a positive duration


%Rescale time between events on each trial
for n=1:n_trl
    if ~all(validwarp(n,:))
        continue;
    end
    T=cell(1,n_evt-1);
    for k=1:n_evt-1
        T{k}=linspace(evtmatrix(n,k),evtmatrix(n,k+1),n_bins(k)+1);
    end
    T(2:end)=cellfun(@(x) x(2:end),T(2:end),'uni',0); %remove overlap
    T=cell2mat(T);

     %Find spike time bins within the current warped epoch of this trial
     curpredID=discretize(binctr,T);
     predID(~isnan(curpredID))=curpredID(~isnan(curpredID));
     trlidx(~isnan(curpredID))=n;
end
assert(all(diff(trlidx(~isnan(trlidx)))>=0),'trialidx is not ascending');
end