function [halfcorr,halfrate,spltcorr,spltrate]=reliablecorr2(predID,spkrates,bincond,bingrp,k)
%RELIABLECORR2 - Calculate the reliability of spike rates within predictor
%bins across trials via correlation.  Yields the comparison of the first 
% and second half of the data and the comparison of alternating events.  If 
% trials are not homogenous, 'evtgrp' specifies which events should remain 
% together to ensure that reliablity is not affected by sampling events 
% across different types. (Modified from reliablecorr and evt_reliability)
%
%predID - Nx1 matrix; Predictor ID for the spike rate bin.
%spkrates - Nx1 matrix. Spike rates.
%bincond - Nx1 vector; Index of condition for each bin (trial type)
%bingrp - Nx1 vector; Index of group identity for each bin (parsed by
%   trial). Events within a group will remain together.
%k - integer.  Number of bins used to create gaussian smoothing kernel.
%   Uses the 'smoothdata' function.
%
%Jon Rueckemann 2024

if nargin<5 || isempty(k)
    k=0; %no gaussian smoothing for Skaggs info or sparsity metrics
end
%#ok<*NANMEAN>

%Convert evtgrp into index that splits the session into halves
halfidx=(bingrp>floor(max(bingrp)./2)); %true = second half

%Convert evtgrp into index that splits the session in alternating pairs
spltidx=logical(rem(bingrp,2)); %true = odd;

%Calculate the mean rate map for each split
goodidx=~isnan(predID);
dims=[max(predID,[],'omitnan') max(bincond,[],'omitnan')];
h_rate1=accumarray([predID(~halfidx&goodidx) bincond(~halfidx&goodidx)],...
    spkrates(~halfidx&goodidx),dims,@nanmean,nan);  
h_rate2=accumarray([predID(halfidx&goodidx) bincond(halfidx&goodidx)],...
    spkrates(halfidx&goodidx),dims,@nanmean,nan);
s_rate1=accumarray([predID(spltidx&goodidx) bincond(spltidx&goodidx)],...
    spkrates(spltidx&goodidx),dims,@nanmean,nan);
s_rate2=accumarray([predID(~spltidx&goodidx) bincond(~spltidx&goodidx)],...
    spkrates(~spltidx&goodidx),dims,@nanmean,nan);

if k>1
    h_rate1=smoothdata(h_rate1,1,'gaussian',k,'omitnan');
    h_rate2=smoothdata(h_rate2,1,'gaussian',k,'omitnan');
    s_rate1=smoothdata(s_rate1,1,'gaussian',k,'omitnan');
    s_rate2=smoothdata(s_rate2,1,'gaussian',k,'omitnan');
end


%Concatenate split maps for output
halfrate=cat(3,h_rate1,h_rate2);
spltrate=cat(3,s_rate1,s_rate2);


%Calculate correlations
halfcorr=corr(h_rate1(:),h_rate2(:),"rows","pairwise");
spltcorr=corr(s_rate1(:),s_rate2(:),"rows","pairwise");
end