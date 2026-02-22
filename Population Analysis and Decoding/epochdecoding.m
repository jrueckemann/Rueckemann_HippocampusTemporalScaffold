function [accuracystruct]=epochdecoding(PP,epoch,Lside,w)
%
%Jon Rueckemann 2025


n_epoch=max(epoch);
[B,T]=size(PP);
assert(2*B==T,'Posterior Probability matrix is ill-formed')

%Reconstruct posterior probabilities if trial type is ignored
PP_unitary=PP(:,1:B)+PP(:,B+1:T);


[binacc_TT,binacc_unitary]=deal(nan(B,1));
[epochacc_TT,epochacc_unitary,chance_epoch_TT,chance_epoch_U,...
    pval_TT,pval_U,effect_TT,effect_U]=deal(nan(n_epoch,1));
[epochdata_TT,epochdata_unitary]=deal(cell(n_epoch,1));
for m=1:n_epoch
    %Establish indices for current epoch; generously include bins outside
    %the epoch when w>0
    curBidx=epoch==m;
    if ~isempty(w) && w>0
        idxstart=find(curBidx,1,"first");
        lo=max(1,idxstart-w);
        idxend=find(curBidx,1,"last");
        hi=min(B,idxend+w);
        curTidx = false(1,B);          
        curTidx(lo:hi)=true;
    else
        curTidx=curBidx;
    end
    if Lside
        curTidx_TT=[curTidx false(size(curTidx))];
    else
        curTidx_TT=[false(size(curTidx)) curTidx];
    end

    %Extract all the posterior probabilities within the current epoch
    cur_binacc_TT=sum(PP(curBidx,curTidx_TT),2);
    binacc_TT(curBidx)=cur_binacc_TT;

    cur_binacc_unitary=sum(PP_unitary(curBidx,curTidx),2);
    binacc_unitary(curBidx)=cur_binacc_unitary;


    %Calculate mean
    epochacc_TT(m)=mean(cur_binacc_TT);
    epochacc_unitary(m)=mean(cur_binacc_unitary);

    epochdata_TT{m}=cur_binacc_TT;
    epochdata_unitary{m}=cur_binacc_unitary;


    %Determine chance
    chance_TT = sum(curTidx) / (2*B);
    chance_U  = sum(curTidx) / B;

    %Stats: true vs chance (one-sample Wilcoxon, exact for small n)
    % Right-tailed: H1 median(cur - chance) > 0
    xTT = cur_binacc_TT - chance_TT;
    xU  = cur_binacc_unitary  - chance_U;
    xTT = xTT(xTT ~= 0);
    xU  = xU(xU ~= 0);


    [pTT,rrb_TT]=signrank_rankbiserial(xTT);

    [pU,rrb_U]=signrank_rankbiserial(xU);


    chance_epoch_TT(m,1) = chance_TT;
    chance_epoch_U(m,1) = chance_U;
    pval_TT(m,1) = pTT;
    pval_U(m,1) = pU;
    effect_TT(m,1) = rrb_TT;
    effect_U(m,1) = rrb_U;



end

if Lside
    curside='Left';
else
    curside='Right';
end

%Package for export
accuracystruct=struct('EpochDef',epoch,'Side',curside,'Window',w,...
    'EpochAccuracy_TT',epochacc_TT,'EpochAccuracy',epochacc_unitary,...
    'BinAccuracy_TT',binacc_TT,'BinAccuracy',binacc_unitary,...
    'EpochData_TT',{epochdata_TT},'EpochData',{epochdata_unitary});
accuracystruct.Chance_TT = chance_epoch_TT;
accuracystruct.Chance_Unitary = chance_epoch_U;
accuracystruct.PValue_TT = pval_TT;           % right-tailed vs chance
accuracystruct.PValue_Unitary = pval_U;
accuracystruct.Effect_RB_TT = effect_TT;      % rank-biserial effect sizes
accuracystruct.Effect_RB_Unitary = effect_U;
end

function [p,rrb]=signrank_rankbiserial(X)
if numel(X)>=1
    [p,~,s]=signrank(X,0,'tail','right');  % exact for small n
    n=numel(X); Wp=s.signedrank; Wtot=n*(n+1)/2;
    rrb=(2*Wp - Wtot)/Wtot;                % rank-biserial ∈ [-1,1]
else
    p=NaN; rrb=NaN;
end
end