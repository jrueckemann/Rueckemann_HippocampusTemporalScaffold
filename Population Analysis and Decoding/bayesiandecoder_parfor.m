function [postprob,LL,entropy,maxIdx,cumprob]=bayesiandecoder_parfor(trlrte,bayesstruct,rotate_data)
%BAYESIANDECODER - Creates posterior probabilities for each bin of trial
% data relative to expected values (lambda).  
%
%trlrte - BxNxT binned firing rates. N-neurons, B-time bins, T-trials
%bayesstruct - struct with the distribution model and parameters
%       distribution - 'poisson','gaussian','lognormal'
%       param1/param2 - BxN parameter matrix per neuron, eg expected value
%rotate_data - {'train','test','both','both_ind','none'} 
%   Arbitrarily rotate each neuron template made from training data.
%   Backward compatability: boolean - rotate_train; 
%   
% 
%
%postprob - BxBxT posterior probability. True bins x decoded bins x trials.
%LL - BxBxT Log-likelihood of the model fit by state
%entropy - BxT. True bins x trials. Decoding entropy @ each time (true bin)
%maxIdx - BxBxT binary matrix.  Max posterior probability for each true bin
%cumprob - BxBxT matrix tracking the accumulated posterior probabilty
%   for each true bin.
%
%
%Jon Rueckemann 2025


if nargin<3
    rotate_data='none';
end


%Backward compatibility for data rotation
if islogical(rotate_data)
    if rotate_data
        rotate_data='train';
    else
        rotate_data='none';
    end
end


%Extract model parameters - Expected firing rates for each time bin
bayesdist=bayesstruct.distribution;
p1=bayesstruct.param1';
p2=bayesstruct.param2';
[n_bins,n_neuron,n_trl]=size(trlrte);

%Independently rotate template from training data for each neuron
switch lower(rotate_data)
    case 'train'
        rndoffset=num2cell(round(rand(n_neuron,1).*n_bins))';

        %Rotate each trial type template (columns) together
        p1=cellfun(@(x,y) circshift(x,y,1),p1,rndoffset,'uni',0);
        if ~isempty(p2)
            p2=cellfun(@(x,y) circshift(x,y,1),p2,rndoffset,'uni',0);
        end
    case 'test'
        rndoffset=num2cell(round(rand(n_neuron,1).*n_bins))';
        
        %Rotate test data each neuron independently
        trlrte=num2cell(trlrte,[1 3]);
        trlrte=cellfun(@(x,y) circshift(x,y,1),trlrte,rndoffset,'uni',0);
        trlrte=cell2mat(trlrte);        
    case 'both'
        rndoffset=num2cell(round(rand(n_neuron,1).*n_bins))';

        %Rotate each trial type template (columns) together
        p1=cellfun(@(x,y) circshift(x,y,1),p1,rndoffset,'uni',0);
        if ~isempty(p2)
            p2=cellfun(@(x,y) circshift(x,y,1),p2,rndoffset,'uni',0);
        end

        %Rotate test data each neuron independently
        trlrte=num2cell(trlrte,[1 3]);
        trlrte=cellfun(@(x,y) circshift(x,y,1),trlrte,rndoffset,'uni',0);
        trlrte=cell2mat(trlrte);
    case 'both_ind'
        %Rotate each trial type template (columns) together
        rndoffset1=num2cell(round(rand(n_neuron,1).*n_bins))';
        p1=cellfun(@(x,y) circshift(x,y,1),p1,rndoffset1,'uni',0);
        if ~isempty(p2)
            p2=cellfun(@(x,y) circshift(x,y,1),p2,rndoffset1,'uni',0);
        end

        %Rotate test data each neuron independently
        rndoffset2=num2cell(round(rand(n_neuron,1).*n_bins))';
        trlrte=num2cell(trlrte,[1 3]);
        trlrte=cellfun(@(x,y) circshift(x,y,1),trlrte,rndoffset2,'uni',0);
        trlrte=cell2mat(trlrte);
    otherwise
end
p1=cell2mat(cellfun(@(x) x(:),p1,'uni',0));
if ~isempty(p2)
    p2=cell2mat(cellfun(@(x) x(:),p2,'uni',0));
end
[n_Dbins]=size(p1,1);

%Calculate log-probability of occupying a bin (assumes equipotentiality)
logProb_S=log(1/n_Dbins);

%Probabilistic decoding of each trial
postprob=zeros(n_bins,n_Dbins,n_trl); % true bins x decoded bins x trials
LL=zeros(n_bins,n_Dbins,n_trl); % true bins x decoded bins x trials
maxIdx=zeros(n_bins,n_Dbins,n_trl); % true bins x decoded bins x trials
cumprob=zeros(n_bins,n_Dbins,n_trl); % true bins x decoded bins x trials
entropy=zeros(n_bins,n_trl); % true bins x trials


%Prep for parfor loop
switch lower(bayesdist)
    case 'poisson' %Poisson distribution
        %Ensure that the expected value is greater than zero
        % (We imagine one extra trial with a single spike in
        %  the bin to find new rate)
        p1(p1==0)=1./(n_trl+1);

        log_prob_zero=nan;
    case 'lognormal' %Lognormal distribution
        %Create values for the possibility of a zero firing rate
        epsilon=1e-6;
        log_prob_zero=log(normcdf((log(epsilon)-p1)./p2,0,1));
        log_prob_zero(isinf(log_prob_zero))=...
            min(log_prob_zero(~isinf(log_prob_zero)));%handle small values
    otherwise
        log_prob_zero=nan;
end


parfor t=1:n_trl
    for b=1:n_bins
        %Observed population activity at time b on trial t
        % (replicated for vectorized calculations)
        r=repmat(trlrte(b,:,t),n_Dbins,1); % [n_Dbins x N]

        %Log-Likelihood; Log-probability of current rates for each state
        %Probability of seeing the observed r given the expected lambda
        % log(P(r|S)); (sum across all neuron rate log-probabilities)
        logProb_RS=compute_LL(r,p1,p2,bayesdist,log_prob_zero);

        %Save the log-likelihood of the model fit, i.e. log(P(r|S))
        LL(b,:,t)=logProb_RS;


        %%log(P(r)): Log-probability of current rates given expected values
        %   P(r)=sum(P(r|S).*P(S))
        %   log(P(r)=sum( exp( log(P(r|S))+log(P(S)) ) )
        %   log(P(r)=alpha+sum( exp( log(P(r|S))+log(P(S))-alpha ) )
        %   alpha subtraction moves the quanties closer to 0 which prevents
        %   underflow when exponentiating very negative numbers, avoiding
        %   false rounding to 0 during exponentiation
        log_Prob_terms=logProb_RS+logProb_S;
        alpha=max(log_Prob_terms); %Underflow stability constant
        log_Prob_R=alpha+log(sum(exp(log_Prob_terms-alpha)));

        %Posterior probablity of a bin (state) for the given rate vector
        %
        %   P(State|rates)=P(rates|State)*P(State)/P(rates)
        %   log(P(S|r)=log(P(r|S))+log(P(S)-log(P(r))
        %   log(P(S|r)=log_Prob_terms-log(P(r))
        alpha2=max(log_Prob_terms-log_Prob_R);
        postprob(b,:,t)=exp(alpha2)*exp(log_Prob_terms-log_Prob_R-alpha2);
        
        %Alt versions *should* be mathematically equivalent 
        % (unclear w underflow)
        % postprob(b,:,t)=exp(log_Prob_terms-log_Prob_R-alpha2);
        % postprob(b,:,t)=postprob(b,:,t)./sum(postprob(b,:,t)); %Rescale
        %
        % postprob(b,:,t)=exp(log_Prob_terms-log_Prob_R); %Simple version


        %Calculate entropy of decoding for each time within the trial
        % Entropy H(S)=-sum(P(S|r) * log(P(S|r)))
        entropy(b,t)=-sum(postprob(b,:,t).*log(postprob(b,:,t)+eps));
        %eps fixes log(0); 0*log(0+eps)=0


        %Aggregate the cumulative probability by position
        [sortval,sortIdx]=sort(postprob(b,:,t),'descend');
        tmpcumprob=zeros(size(postprob(b,:,t))); %hack for parfor
        tmpcumprob(sortIdx)=cumsum(sortval);
        cumprob(b,:,t)=tmpcumprob;


        %Find the maximal decoding state
        [~,maxidx]=max(postprob(b,:,t));
        tmpmaxidx=zeros(size(postprob(b,:,t))); %hack for parfor
        tmpmaxidx(maxidx)=1;
        maxIdx(b,:,t)=tmpmaxidx;
    end
end
end


function logProb_RS=compute_LL(r,p1,p2,bayesdist,log_prob_zero)
switch lower(bayesdist)
    case 'poisson' %Poisson distribution
        % P(r|S) = lambda^r * e^(-lambda) / r!
        logProb_RS=sum(r.*log(p1)-p1-gammaln(r+1),2);

    case {'gaussian','normal'} %Gaussian/Normal distribution
        % P(r|S) = 1/sqrt(2pi*sigma^2) * e^(-((r-lambda)^2)/2*sigma^2)
        tmp_logProb_RS=-0.5*(log(2*pi*p2.^2)+...
            ((r-p1).^2)./p2.^2);

        %Replace values with 0 variance, ie. 100% certainty
        %   Note: Model will not be suitable if test data ~=0
        tmp_logProb_RS(p2==0)=0;
        logProb_RS=sum(tmp_logProb_RS,2);

    case 'lognormal' %Lognormal distribution
        % P(r|S) = 1/(r*sqrt(2pi*phi^2)) * e^(-((log(r)-mu)^2)/2*phi^2)
        % mu = mean(log(X));  phi = std(log(X));
        tmp_logProb_RS=-0.5*(log(2*pi*p2.^2)+...
            ((log(r)-p1).^2)./p2.^2)-log(r);

        %Replace undefined values for 0 firing rates before sum
        tmp_logProb_RS(r==0)=log_prob_zero(r==0);

        %Replace values with 0 variance, ie. 100% certainty
        %   Note: Model will not be suitable if test data ~=0
        tmp_logProb_RS(p2==0)=0;

        logProb_RS=sum(tmp_logProb_RS,2);


    otherwise
        error('Unsupported distribution: %s', bayesdist);
end
end