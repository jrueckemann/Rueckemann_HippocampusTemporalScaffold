function [postprob,LL,entropy,maxIdx,...
    cormat,trlcor,meancor,maxposcor,...
    pval_meancor,pval_poolcor,pval_trlcor]=...
    CVdecoding(trlrte,n_cv,bayesdist,n_pseudotrl,samp_meth,rotate_data,cortest,coriter)
%
%trlrte - Nx{{BxT}xC} cell matrix. (use trlratetransform before input)
%   N-neurons, B-time bins, T-trials, C-conditions
%n_cv - integer. Number of cross-validation tests
%bayesdist - string. Parametric distribution used for Bayesian decoding
%n_pseudotrl - integer. Number of pseudopopulation trials created
%samp_meth - struct. Fields: SampleRatio-ratio, SampleReplacement-boolean,
%   CondSpecific-boolean [currently not implemented]
%rotate_data - {'train','test','both','both_ind','none'} 
%   Arbitrarily rotate each neuron template made from training data.
%   Backward compatability: boolean - rotate_train; 
%
%corrtest - boolean. 1) Compute correlation matrix between mean of training
%   and test data across the trial.  2) Compute correlation matrices of
%   mean of training data and each pseudopopulation vector in test data
%coriter - integer. Iteration number to randomize to determine corr p-value
%   Default - 0.  (No p-value calculation)
%
%Jon Rueckemann 2025


if nargin<5 || isempty(samp_meth)
    samp_meth=struct('SampleRatio',0.5,'SampleReplacement',false,...
        'CondSpecific',true);
end
if nargin<6 || isempty(rotate_data)
    rotate_data=false;
end
if nargin<7 || isempty(cortest)
    cortest=false;
end
if nargin<8 || isempty(coriter)
    coriter=0;
end


%Preallocate data splits
n_unit=numel(trlrte);
n_cond=cellfun(@numel,trlrte);
n_cond=unique(n_cond);
assert(numel(n_cond)==1,'Inconsistent number of conditions in each neuron')
trlsplit=repmat(cell(1,1,n_cond),n_unit,1);
if ~samp_meth.SampleReplacement
    assert(n_cv*samp_meth.SampleRatio<=1,...
        ['The sampling method and n_cv are incompatibile. '...
        'The ratio of sample trials must be less than 1/n_cv.'])
end

for n=1:n_unit %Iterate across neurons
    for c=1:n_cond %Iterate across conditions
        trlsplit{n}{c}=randsampling(trlrte{n}{c},n_cv,samp_meth);
    end
end


%Iterate through data splits
[postprob,LL,entropy,maxIdx,trlcor,meancor,maxposcor,...
    pval_meancor,pval_poolcor,pval_trlcor]=deal(cell(n_cond,n_cv));
cormat=cell(1,n_cv);
for s=1:n_cv
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



    %Determine distribution parameters for Bayesian decoding
    [bayesstruct]=estimatebayesianparameter(train_trl,bayesdist);

    assert(~any(cellfun(@(x) any(isinf(x(:))),bayesstruct.param1,'uni',1)),...
        'A distribution parameter is infinite');
    assert(~any(cellfun(@(x) any(isnan(x(:))),bayesstruct.param1,'uni',1)),...
        'A distribution parameter is NaN');
    if ~isempty(bayesstruct.param2)
        assert(~any(cellfun(@(x) any(isinf(x(:))),bayesstruct.param2,'uni',1)),...
            'A distribution parameter is infinite');
        assert(~any(cellfun(@(x) any(isnan(x(:))),bayesstruct.param2,'uni',1)),...
            'A distribution parameter is NaN');
    end


    %Construct pseudopopulation trials for test data
    P=createpseudopopulation(test_trl,n_pseudotrl);

    %Calculate Bayesian decoding results
    for p=1:n_cond
        [postprob{p,s},LL{p,s},entropy{p,s},maxIdx{p,s}]=...
            bayesiandecoder_parfor(P{p},bayesstruct,rotate_data);
    end

    %Calculate correlation tests
    if cortest
        [mu_train,mu_test]=deal(cell(1,n_unit,n_cond));
        for c=1:n_cond
            %Mean across trials for each bin w/in each condition
            %1xNxC cell array, each {Bx1}
            mu_train(1,:,c)=cellfun(@(x) mean(x{c},2),train_trl,'uni',0);
            mu_test(1,:,c)=cellfun(@(x) mean(x{c},2),test_trl,'uni',0);
        end
        %Convert to BxNxC matrix
        mu_train=cell2mat(mu_train);
        mu_test=cell2mat(mu_test);
        % %Convert to Nx{BxC} of mean rates
        % mu_train=cellfun(@(x) cell2mat(x),num2cell(mu_train,2),'uni',0);
        % mu_test=cellfun(@(x) cell2mat(x),num2cell(mu_test,2),'uni',0);

        %Calculate cross-correlation matrix of training and test mean data
        tmp_mu_train=num2cell(mu_train,[1 2]);
        tmp_mu_train=cell2mat(tmp_mu_train(:));
        tmp_mu_test=num2cell(mu_test,[1 2]);
        tmp_mu_test=cell2mat(tmp_mu_test(:));
        cormat{s}=corr(tmp_mu_train',tmp_mu_test');

        %Cross-correlation of test pseudopopulation with training mean data
        for p=1:n_cond
            [trlcor{p,s},meancor{p,s},maxposcor{p,s},...
                pval_meancor{p,s},pval_poolcor{p,s},pval_trlcor{p,s}]=...
                corrdecoder(P{p},mu_train,coriter);
        end
    end
end
end


function [trlidx]=randsampling(unit_trlrte,n_cv,samp_meth)
n_trl=size(unit_trlrte,2);
n_samp=floor(n_trl.*samp_meth.SampleRatio);
assert(n_samp+1<n_trl,...
    'There are not enough trials to support the SampleRatio');

trlidx=nan(n_cv,n_samp);
if samp_meth.SampleReplacement
    %Sample with replacement
    for n=1:n_cv
        [~,rndidx]=sort(rand(1,n_trl));
        trlidx(n,:)=rndidx(1:n_samp);
    end
else
    %Sample without replacement
    [~,rndidx]=sort(rand(1,n_trl));
    for n=1:n_cv
        trlidx(n,:)=rndidx((n_samp*(n-1)+1):(n_samp*n));
    end
end
end
