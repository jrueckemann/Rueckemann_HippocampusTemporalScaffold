function [pseudopop]=createpseudopopulation(trlrmp,iter)
%
%trlrmap - Nx{{BxT}xC} cell matrix.  
%   N-neurons, B-time bins, T-trials, C-conditions
%iter - integer. Number of pseudopopulation trials created
%
%
%Jon Rueckemann 2025


n_cond=cellfun(@numel,trlrmp);
n_cond=unique(n_cond);
assert(numel(n_cond)==1,'Inconsistent number of conditions in each neuron')


%Iterate through conditions
pseudopop=cell(n_cond,1);
for c=1:n_cond
    curdata=cellfun(@(x) x{c},trlrmp,'uni',0);

    %Create sampling scheme
    trlnum=cellfun(@(x) size(x,2), curdata);%trials per neuron in condition
    rndtrlidx=ceil(trlnum.*rand(size(trlnum,1),iter));

    %Create pseudopopulation trials
    poptrials=cell(1,size(curdata,1),iter);
    for n=1:iter
        poptrials(1,:,n)=cellfun(@(x,y) x(:,y),curdata,...
            num2cell(rndtrlidx(:,n)),'uni',0);
    end
    pseudopop{c}=cell2mat(poptrials);
end
end