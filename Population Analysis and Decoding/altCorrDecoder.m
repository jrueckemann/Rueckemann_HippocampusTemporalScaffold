function [trlcor,meancor,maxposcor]=altCorrDecoder(trlrte,n_iter,n_pseudotrl,samp_meth,rotate_data)
%Compute correlation matrix between mean of training and test data across
% the trial.  Compute correlation matrices of mean of training data and
% each pseudopopulation vector in test data
%
%
%trlrte - Nx{{BxT}xC} cell matrix. (use trlratetransform before input)
%   N-neurons, B-time bins, T-trials, C-conditions
%n_iter - integer. Number of cross-neuron rotation iterations
%n_pseudotrl - integer. Number of pseudopopulation trials created
%samp_meth - struct. Fields: SampleRatio-ratio, SampleReplacement-boolean,
%   CondSpecific-boolean [currently not implemented]
%rotate_data - {'train','test','both','both_ind','none'}
%   Arbitrarily rotate each neuron template made from training data.
%
%
%Jon Rueckemann 2025



if nargin<4 || isempty(samp_meth)
    samp_meth=struct('SampleRatio',0.5,'SampleReplacement',true,...
        'CondSpecific',true);
end
if nargin<5 || isempty(rotate_data)
    rotate_data='none';
end


%Preallocate data splits
n_unit=numel(trlrte);
n_cond=cellfun(@numel,trlrte);
n_cond=unique(n_cond);
assert(numel(n_cond)==1,'Inconsistent number of conditions in each neuron')
trlsplit=repmat(cell(1,1,n_cond),n_unit,1);
if ~samp_meth.SampleReplacement
    assert(n_iter*samp_meth.SampleRatio<=1,...
        ['The sampling method and n_iter are incompatibile. '...
        'The ratio of sample trials must be less than 1/n_iter.'])
end
%   Seems like I need some total trial within unit checks
for n=1:n_unit %Iterate across neurons
    for c=1:n_cond %Iterate across conditions
        trlsplit{n}{c}=randsampling(trlrte{n}{c},n_iter,samp_meth);
    end
end



%Iterate through data splits
[trlcor,meancor,maxposcor]=deal(cell(n_cond,n_iter));
parfor s=1:n_iter
    train_trl=repmat({cell(1,1,n_cond)},n_unit,1);
    test_trl=repmat({cell(1,1,n_cond)},n_unit,1);

    %Sample training and test data
    for n=1:n_unit
        for c=1:n_cond
            curtrlidx=trlsplit{n}{c}(s,:);
            train_trl{n}{c}=trlrte{n}{c}(:,curtrlidx); %training data

            testtrlidx=true(1,size(trlrte{n}{c},2));
            testtrlidx(curtrlidx)=false;
            test_trl{n}{c}=trlrte{n}{c}(:,testtrlidx); %test data
        end
    end


    %Convert training data to mean rate
    mu_train=cell(n_unit,n_cond);
    for c=1:n_cond
        mu_train(:,c)=cellfun(@(x) mean(x{c},2),train_trl,'uni',0);
    end
    mu_train=cellfun(@(x) cell2mat(x),num2cell(mu_train,2),'uni',0);


    %Construct pseudopopulation trials for test data
    P=createpseudopopulation(test_trl,n_pseudotrl);

    %Rotate data
    P=cell2mat(reshape(P,1,1,n_cond));
    mu_train=cell2mat(reshape(cellfun(@(x) permute(x,[1 3 2]),...
        mu_train,'uni',0),1,n_unit));
    [mu_train,P]=rotatetrialdata(mu_train,rotate_data,P);

    %Create template that spans trial types
    exp_val=num2cell(mu_train,[1 2]);
    exp_val=cell2mat(exp_val(:));
    exp_val=exp_val'; %unit --> rows

    %Reseparate pseudotrials by condition/trial type
    n_bins=size(P,1);
    P=mat2cell(P,n_bins,n_unit,repmat(n_pseudotrl,1,n_cond));


    for c=1:n_cond
        %Find pairwise correlation between population vectors in each trial
        % and expected value based on training data        
        cur_trlcor=nan([n_bins size(exp_val,2) n_pseudotrl]);
        for t=1:n_pseudotrl
            cur_trlcor(:,:,t)=corr(P{c}(:,:,t)',exp_val);%unit --> rows
        end


        %Find position of maximum correlation for each trial pop vec
        cur_maxpos=nan(size(cur_trlcor));
        for t=1:n_pseudotrl
            [~,maxidx]=max(cur_trlcor(:,:,t),[],2);
            ind=sub2ind([size(cur_maxpos,1) size(cur_maxpos,2)], ...
                1:size(cur_maxpos,1),maxidx');
            tmp=zeros([size(cur_maxpos,1) size(cur_maxpos,2)]);
            tmp(ind)=1;
            cur_maxpos(:,:,t)=tmp;
        end

        
        %Calculate mean correlation value
        meancor{c,s}=reversefishertransform(...
            mean(fishertransform(cur_trlcor),3));

        
        %Store output
        trlcor{c,s}=cur_trlcor;
        maxposcor{c,s}=cur_maxpos;
    end
end
end



function [trlidx]=randsampling(unit_trlrte,n_iter,samp_meth)
n_trl=size(unit_trlrte,2);
n_samp=floor(n_trl.*samp_meth.SampleRatio);
assert(n_samp+1<n_trl,...
    'There are not enough trials to support the SampleRatio');

trlidx=nan(n_iter,n_samp);
if samp_meth.SampleReplacement
    %Sample with replacement
    for n=1:n_iter
        [~,rndidx]=sort(rand(1,n_trl));
        trlidx(n,:)=rndidx(1:n_samp);
    end
else
    %Sample without replacement
    [~,rndidx]=sort(rand(1,n_trl));
    for n=1:n_iter
        trlidx(n,:)=rndidx((n_samp*(n-1)+1):(n_samp*n));
    end
end
end


function [z]=fishertransform(r)
r(abs(r)==1)=sign(r(abs(r)==1))*(1-eps); %modify correlation to avoid inf
z=0.5*log((1+r)./(1-r));
end


function [r]=reversefishertransform(z)
r=(exp(2*z)-1)./(exp(2*z)+1);
end