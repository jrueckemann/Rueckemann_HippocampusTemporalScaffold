function [predID,trlidx,validwarp,warptsOut,knotmap,epochmap,predmap]=warptrial_fulltrial4info_lindist3(evtmatrix,timetemplate,binctr,lindist)
%
%evtmatrix - NxK+1 matrix.  Timestamps of K events across N trials.  Final
%   column is the end of the trial (which may coincide with beginning
%   of next trial). Algorithm assumes evtmatrix is well-conditioned.
%timetemplate - struct.  Definition of time transform across event epochs.
%binctr - Tx1 vector.  Timestamps of center of spike bins to be mapped to
%   predictor.
%lindist - Mx2 array.  Col 1, timestamp.  Col 2, linear distance.
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
lindistscale=timetemplate.lindistscale;
[~,uidx]=unique(lindist(:,1));
lindist=lindist(uidx,:);
knot=timetemplate.knot;
n_knotbins=nan(size(n_bins));
for k=1:n_evt-1
    if k==1
        warpts{k}=0:bindur:newdur(k);
    else
        warpts{k}=(0:bindur:newdur(k))+max(warpts{k-1});
    end
    knotmap{k}=nan(size(warpts{k})); %knot template in rescaled time
    if lindistscale(k)
        n_knotbins(k)=round(knot{k}(1)./bindur);
        knotmap{k}(1:(n_knotbins(k)+1))=1;
    end
    epochmap{k}=k*ones(1,numel(warpts{k})-1); 
    predoffset=sum(cellfun(@numel,predmap(1:k)));
    predmap{k}=(1:numel(epochmap{k}))+predoffset;
end
warpts(2:end)=cellfun(@(x) x(2:end),warpts(2:end),'uni',0); %delete overlap
warptsOut=cell2mat(warpts);
knotmap=cell2mat(knotmap);
epochmap=cell2mat(epochmap);
predmap=cell2mat(predmap);
validwarp=diff(evtmatrix,1,2)>0; %ensure each epoch has a positive duration
curoffset=cumsum(cellfun(@(x) numel(x)-1,warpts));

%Determine size of minimum interval between temporal samples
dt=sum(newdur)*eps;

%Rescale time between events on each trial
for n=1:n_trl
    if ~all(validwarp(n,:))
        continue;
    end
    %T=cell(1,n_evt-1);
    for k=1:n_evt-1
        if lindistscale(k)
            %Find indices of current interval and current lindist
            curtrlidx=lindist(:,1)>=evtmatrix(n,k) ...
                & lindist(:,1)<=evtmatrix(n,k+1);
            curlintme=lindist(curtrlidx,1);
            curlindist=lindist(curtrlidx,2);
            ctrpt=find(curlindist>=0,1,"first");
            if isempty(ctrpt)
                continue%trial never passes center; effectively drops trial
            end

            %Convert distance to relative progress: center
            knotdur=n_knotbins(k)*bindur;
            ctrlindist=curlindist(1:(ctrpt-1));
            ctrprogress=(ctrlindist-min(curlindist))./abs(min(curlindist));
            warptmeprog_ctr=ctrprogress.*knotdur;

            %Convert distance to relative progress: after center
            postlindist=curlindist(ctrpt:end);
            postprogress=postlindist./max(curlindist);
            warptmeprog_post=(postprogress.*(newdur(k)-knotdur))+knotdur;

            %Aggregate trial progress
            warptmeprog=[warptmeprog_ctr; warptmeprog_post];

            %Add minimal granule of time to allow for interpolation
            warptmeprog=warptmeprog+dt*cumsum(ones(size(warptmeprog)));


            %Identify spike bins within trial linear traversal
            curidx=binctr>=evtmatrix(n,k) & binctr<evtmatrix(n,k+1);

            %Convert each spike bin to a predictor index
            curwarp=interp1(curlintme,warptmeprog,binctr(curidx),'pchip');
            curpred=discretize(curwarp,warpts{k});            

            predID(curidx)=curpred;
            trlidx(curidx)=n;

%             %Convert relative progress to correct sampling intervals
%             curwarpts=warpts{k}-warpts{k}(1); %match warpts to progress
%             T{k}=interp1(warptmeprog,curlintme,curwarpts,'pchip');
        else
            T=linspace(evtmatrix(n,k),evtmatrix(n,k+1),n_bins(k)+1);

            %Find spike time bins within current warped epoch of this trial
            curpred=discretize(binctr,T);
            predID(~isnan(curpred))=curpred(~isnan(curpred))+curoffset(k-1);
            trlidx(~isnan(curpred))=n;
        end
    end
end
end