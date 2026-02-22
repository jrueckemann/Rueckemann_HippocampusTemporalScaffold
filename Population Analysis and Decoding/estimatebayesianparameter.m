function [bayesstruct]=estimatebayesianparameter(trlrate,distribution)
%
%trlrte - Nx{{BxT}xC} cell matrix.  Formatted by 'trlratetransform'.
%   N-neurons, B-time bins, T-trials, C-conditions
%
%Output:
%bayesstruct - struct with the distribution model and parameters
%       distribution - 'poisson','gaussian','lognormal'
%       param1/param2 - expected value parameter matrix per neuron.
%
%Jon Rueckemann 2025


%Aggregate trial rate within neuron across conditions as a column vector
aggtrlrate=cellfun(@(x) cellfun(@(y) y(:),x,'uni',0),trlrate,'uni',0);
aggtrlrate=cellfun(@(x) cell2mat(x(:)),aggtrlrate,'uni',0);

%Find overall mean and variance of the firing rate for each neuron
aggmean=cellfun(@(x) mean(x),aggtrlrate,'uni',1);
aggvar=cellfun(@(x) var(x,0),aggtrlrate,'uni',1);

%Calculate Fano factor for each neuron across all state bins
bayesstruct.fano=aggvar./aggmean;


%Find mean and variance of each unit's spiking in each state and condition
n_units=numel(trlrate);
n_cond=cellfun(@numel,trlrate);
n_cond=unique(n_cond);
assert(numel(n_cond)==1,'Inconsistent number of conditions in each neuron')
n_bin=cell2mat(cellfun(@(x) cellfun(@(y) size(y,1),x,'uni',1),trlrate,'uni',0));
n_bin=unique(n_bin(:));
assert(numel(n_bin)==1,'Inconsistent number of bins in each neuron')
[mu_rate,var_rate]=deal(cell(n_units,n_cond));
for c=1:n_cond
    mu_rate(:,c)=cellfun(@(x) mean(x{c},2),trlrate,'uni',0);
    var_rate(:,c)=cellfun(@(x) var(x{c},0,2),trlrate,'uni',0);
end
mu_rate=cellfun(@(x) cell2mat(x),num2cell(mu_rate,2),'uni',0);
var_rate=cellfun(@(x) cell2mat(x),num2cell(var_rate,2),'uni',0);



%Estimate distribution parameters for each neuron and state
bayesstruct.distribution=distribution;
bayesstruct.mean=mu_rate;

switch lower(distribution)
    case 'poisson'
        %poisson is a single parameter distribution that assumes
        %variance is equal to the mean. (no second parameter)
        bayesstruct.param1=mu_rate;
        bayesstruct.param2=[];
        distname='Poisson';
    case 'gaussian'
        [lambda,sigma]=deal(repmat({nan(n_bin,n_cond)},n_units,1));
        for n=1:n_units
            for c=1:n_cond
                %Iterate through bins across trials
                for b=1:n_bin
                    r=trlrate{n}{c}(b,:);
                    [lambda{n}(b,c),sigma{n}(b,c)]=normfit(r);
                end
            end
        end
        bayesstruct.param1=lambda; %mean
        bayesstruct.param2=sigma; %standard deviation
        distname='Normal';
    case 'lognormal'
        [mu,sigma]=deal(repmat({nan(n_bin,n_cond)},n_units,1));
        for n=1:n_units
            for c=1:n_cond
                %Iterate through bins across trials
                for b=1:n_bin
                    r=trlrate{n}{c}(b,:);
                    r(r==0)=eps;
                    mdl=lognfit(r);

                    %Convert to MLE output to lambda and sigma
                    mu{n}(b,c)=mdl(1); %mean of log(rates)
                    sigma{n}(b,c)=mdl(2); %std deviation of log(rates)
                end
            end
        end
        bayesstruct.param1=mu;
        bayesstruct.param2=sigma;
        distname='Lognormal';
    otherwise
        warning('Specified distribution is not supported.')
end
bayesstruct.distribution=distname;


end



