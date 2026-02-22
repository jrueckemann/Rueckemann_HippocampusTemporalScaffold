function [iPos,shannoninfo,Px,condspkmu,unqpredID]=Olypherinfo(predID,spkcounts,Px)
%
%
%Note 1: Assumes equal durations for the spkcounts, esp Skaggs and sparsity.
%Calculations done on spike counts instead of rate, because dt cancels out.
%Olypher-Fenton, 2003 'Information by position'
%
%Note 2: The number of elements in Px and condspkmu might not equal the
%elements of iPos.  iPos values will be assigned to the index specified by
%predID; Px and condspkmu only reflect the elements of predID that were
%occupied.
%
%Jon Rueckemann 2022

%Index binned predictors
[unqpredID,~,pIdx]=unique(predID);
n_bins=numel(pIdx);
temp_iPos=zeros(max(pIdx),size(spkcounts,2));
condspkmu=zeros(max(pIdx),size(spkcounts,2));

if nargin<3
    %Probability of each predictor condition
    Px=accumarray(pIdx,1,[max(pIdx) 1])./n_bins;
end

%Calculate information scores for each spike train
for n=1:size(spkcounts,2)
    %Probability of each observed spike count
    [~,~,spkidx]=unique(spkcounts(:,n));
    Pk=accumarray(spkidx,1)./n_bins;
    max_spkidx=max(spkidx);

    %mean spike counts for Skaggs Info and Sparsity calculations
    condspkmu(:,n)=accumarray(pIdx,spkcounts(:,n),[],@mean);

    %Determine the information content of spike train for each condition
    for m=1:max(pIdx)
        curidx=pIdx==m; %find all entries for condition m

        %iPos
        Pkx=accumarray(spkidx(curidx),1,[max_spkidx 1])./sum(curidx);
            %probability of spike counts for condition m
        temp_iPos(m,n)=sum(Pkx.*log2(Pkx./Pk),'omitnan');
            %info during condition m
    end
end

%Restore the condition bin index for iPos. Creates a 1D array for each unit
iPos=nan(max(unqpredID),size(spkcounts,2));
iPos(unqpredID,:)=temp_iPos;

%Calculate Shannon Mutual Information Score
shannoninfo=sum(Px.*temp_iPos,1);
end