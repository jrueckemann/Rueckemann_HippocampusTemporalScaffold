function [newrmp,trlrmp_mu,grprmp_mu]=trlratetransform(trlrmp,Ntrlavg,unitary,wnd,scalemeth,postregroup,valididx)
%
%Input:
%trlrmap - Nx{BxTxC} cell matrix.  
%   N-neurons, B-time bins, T-trials, C-conditions
%Ntrlavg - integer.  Number of randomly selected trials to be averaged
%unitary - boolean.  Collapse across conditions
%wnd - integer.  Number of bins in gaussian smooth kernel. 0 for no smooth
%scalemeth - 'max','zscore','softnorm'.  Rescale trial rates. Empty does no
%   rescaling.
%postregroup - boolean. Apply rescaling after combining data across trials
%   and smoothing; rmptrl_mu will not be rescaled. 'False' results in 
%   rescaling before combining data across trials and smoothing; 
%   applies rescaling to rmptrl_mu.
%valididx - boolean vector. Truncation of B
%
%
%Output:
%newrmp - Nx{{BxT}xC} or {{BxU}xC} matrix of regrouped and rescaled trials
%trlrmp_mu - Nx{BxC} matrix of mean trial response before regrouping
%grprmp_mu - Nx{BxC} matrix of mean trial response after regrouping
%   C=1 if unitary
%
%
%Jon Rueckemann 2025


if nargin<7
    valididx=[];
end

if ~isempty(valididx)
    trlrmp=cellfun(@(x) x(valididx,:,:),trlrmp,'uni',0);
end

%Iterate through neurons and format trials
[newrmp,trlrmp_mu,grprmp_mu]=deal(cell(size(trlrmp)));
for n=1:numel(trlrmp)
    %Find valid trials in each condition
    cur_unit=num2cell(trlrmp{n},[1 2]);
    validtrl=cellfun(@(x) all(~isnan(x),1),cur_unit,'uni',0);
    cur_unit=cellfun(@(x,y) x(:,y),cur_unit,validtrl,'uni',0);

    %Aggregate trials if treating conditions as unitary
    if unitary
        cur_unit={cell2mat(reshape(cur_unit,1,[]))};
    end

    %Rescale data before grouping into new faux trials and smoothing
    if ~isempty(scalemeth) && ~postregroup
        cur_unit=rescalerate(cur_unit,scalemeth);
    end

    %Mean firing rate across original trials for each bin in each condition
    trlrmp_mu{n}=...
        cell2mat(reshape(cellfun(@(x) mean(x,2),cur_unit,'uni',0),1,[]));

    %Create faux trials for each condition by averaging subsets of trials
    if Ntrlavg>1
        new_unit=cell(size(cur_unit));
        for m=1:numel(cur_unit)
            %Index trials within current condition
            [n_bin,cur_Ntrl]=size(cur_unit{m});
            cur_Niter=floor(cur_Ntrl./Ntrlavg);
            idx=randsample(cur_Ntrl,cur_Niter.*trlavg);
            new_unit{m}=nan(n_bin,cur_Niter);
            for t=1:cur_Niter %faux trials sampled from unique trials
                curidx=idx((((t-1)*trlavg)+1):(t*trlavg));
                new_unit{m}(:,t)=mean(cur_unit{m}(:,curidx),2);
            end
        end
    else
        new_unit=cur_unit;
    end

    %Smooth data
    if wnd>0
        %faux trials
        for m=1:numel(new_unit)
            new_unit{m}=smoothdata(new_unit{m},2,'gaussian',wnd);
        end

        %mean rate maps
        trlrmp_mu{n}=smoothdata(trlrmp_mu{n},2,'gaussian',wnd);
    end

    %Rescale data after grouping into new faux trials and smoothing
    if ~isempty(scalemeth) && postregroup
        new_unit=rescalerate(new_unit,scalemeth);
    end

    %Mean firing rate across new faux trials for each bin in each condition
    grprmp_mu{n}=...
        cell2mat(reshape(cellfun(@(x) mean(x,2),new_unit,'uni',0),1,[]));

    newrmp{n}=new_unit;
end
end


function [Xout]=rescalerate(X,scalemeth)

%Rescale data
Y=cell2mat(reshape(cellfun(@(x) x(:),X,'uni',0),[],1));
switch lower(scalemeth)
    case 'max'
        maxval=max(Y);
        Xout=cellfun(@(x) x./maxval,X,'uni',0);
    case 'zscore'
        meanval=mean(Y);
        stdval=std(Y);
        Xout=cellfun(@(x) (x-meanval)./stdval,X,'uni',0);
    case 'softnorm' %'Soft normalization'
        %from Yoo and Hayden, 2020 via Churchland et al, 2012
        scaleval=range(Y)+max(Y)*.2;
        meanval=mean(Y./scaleval);

        Xout=cellfun(@(x) (x./scaleval)-meanval,X,'uni',0);
end
end