function [trlcorr,meancorr,maxpos,pval_meancorr,pval_trlcorr]=corrdecoder(trldata,expval,iter)
%trldata - BxNxT matrix; Trial data.  B-bins, N-neurons, T-trials
%
%expval - SxNxC matrix; Expected value of firing rates.
%   S-states, N-neurons, C-conditions. S may not equal B.
%
%
%Jon Rueckemann 2025

assert(size(trldata,2)==size(expval,2),'Input number of units must match.')
n_trlbins=size(trldata,1);
n_trl=size(trldata,3);

%Find pairwise correlation between population vectors in each trial and
% expected value
true_expval=num2cell(expval,[1 2]);
true_expval=cell2mat(true_expval(:));
trlcorr=nan([n_trlbins size(true_expval,1) n_trl]);
true_expval=true_expval'; %units need to be rows
parfor n=1:n_trl
    trlcorr(:,:,n)=corr(trldata(:,:,n)',true_expval);%units need to be rows
end

%Find position of the prior with maximum correlation for each trial pop vec
maxpos=nan(size(trlcorr));
for n=1:n_trl
    [~,maxidx]=max(trlcorr(:,:,n),[],2);
    ind=sub2ind([size(maxpos,1) size(maxpos,2)],1:size(maxpos,1),maxidx');
    tmp=zeros([size(maxpos,1) size(maxpos,2)]);
    tmp(ind)=1;
    maxpos(:,:,n)=tmp;
end

%Calculate mean correlation value
meancorr=reversefishertransform(mean(fishertransform(trlcorr),3));


if iter>0
    %Rotate the expected value of each neuron WRT the task events and
    % recalculate correlations
    rndoffset=ceil(rand(size(expval,2),iter).*size(expval,1));
    mu_fauxcorr=nan([size(meancorr) iter]);
    count_fauxcorr=false([size(trlcorr) iter]); %4-D
    for m=1:iter
        fauxcorr=nan(size(trlcorr));
        faux_expval=nan(size(expval));
        for c=1:size(expval,3)
            for n=1:size(expval,2)
                faux_expval(:,n,c)=circshift(expval(:,n,c),rndoffset(n,m));
            end
        end

        faux_expval=num2cell(faux_expval,[1 2]);
        faux_expval=cell2mat(faux_expval(:));
        faux_expval=faux_expval'; %units need to be rows
        parfor n=1:n_trl
            fauxcorr(:,:,n)=corr(trldata(:,:,n)',faux_expval);%unit=rows
        end
        mu_fauxcorr(:,:,m)=...
            reversefishertransform(mean(fishertransform(fauxcorr),3));

        %Does each trial's correlation to true expected value exceed each 
        % trial's correlation to the shuffled expected value?
        count_fauxcorr(:,:,:,m)=trlcorr>fauxcorr;
    end


    %Use rotated data to develop p-values: mean correlation
    pval_meancorr=mean(repmat(meancorr,1,1,iter)>mu_fauxcorr,3);

    %Use rotated data to develop p-values: trial correlations
    %"How often does each trial's correlation to true expected value
    %exceed each trial's correlation to the shuffled expected value?"
    pval_trlcorr=mean(count_fauxcorr,4);

else
    pval_meancorr=[];
    pval_trlcorr=[];
end

end


function [z]=fishertransform(r)
r(abs(r)==1)=sign(r(abs(r)==1))*(1-eps); %modify correlation to avoid inf
z=0.5*log((1+r)./(1-r));
end

function [r]=reversefishertransform(z)
r=(exp(2*z)-1)./(exp(2*z)+1);
end
