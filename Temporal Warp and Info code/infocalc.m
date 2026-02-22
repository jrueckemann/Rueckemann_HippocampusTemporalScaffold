function [skaggsinfo,sparsity1,sparsity2,condspkmu]=infocalc(Px,condspkmu,trltype,k)
%
%Skaggs et al, 1993 Information by spike
%Skaggs et al, 1996 sparsity
%Ahmed & Mehta, 2009 sparsity (ignores unequal probability of conditions)
%
%Jon Rueckemann 2022

if nargin==4 && ~isempty(k) && k>1
    %Iterate through trial types
    for m=1:max(trltype)
        %Smooth each neuron spike count separately
        for n=1:size(condspkmu,2)
            condspkmu(trltype==m,n)=...
                smoothdata(condspkmu(trltype==m,n),'gaussian',k,...
                'omitnan');
        end
        Px(trltype==m)=smoothdata(Px(trltype==m),'gaussian',k,...
            'omitnan');
    end
end
meanspk=sum(Px.*condspkmu,1,'omitnan');

%Calculate Skaggs Information Score
skaggsinfo=sum(Px.*condspkmu.*log2(condspkmu./meanspk),1,'omitnan')./meanspk;

%Sparsity scores
sparsity1=sum(Px.*condspkmu,1,'omitnan').^2./sum(Px.*condspkmu.^2,1,'omitnan');%Skaggs 1996
sparsity2=1./size(condspkmu,1).*sum(condspkmu,1,'omitnan').^2./sum(condspkmu.^2,1,'omitnan');%Ahmed 2009
end