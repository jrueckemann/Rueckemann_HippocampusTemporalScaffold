function [runboundary,flat_trlboundary,trlboundary_peak]=trial2runboundary(trlboundary,nbin)
%
%Jon Rueckemann 2025

if nargin<2
    nbin=68;
end




[runboundary,flat_trlboundary]=deal(cell(size(trlboundary)));
trlboundary_peak=nan(size(trlboundary));
for m=1:numel(trlboundary)
    %Collapse trial boundaries across cluster IDs
    flat_trlboundary{m}=sum(trlboundary{m},1);


    %Convert trial boundaries to aggregate binary
    [maxval,maxidx]=max(trlboundary{m},[],2);
    tmp=zeros(1,nbin);
    tmp(maxidx)=1;
    runboundary{m}=tmp;

    %Retain the peak density of a cluster boundary
    trlboundary_peak(m)=max(maxval);
end