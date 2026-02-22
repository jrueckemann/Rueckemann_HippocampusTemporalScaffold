function [trlrte,trlrte2,rndoffset]=rotatetrialdata(trlrte,rotate_data,trlrte2,rndoffset,stream)
%
%trlrte - BxNxT binned firing rates. B-time bins, N-neurons, T-trials
%rotate_data - {'train','test','both','both_ind'} 
%   Arbitrarily rotate each neuron template made from training data.
%trlrte2 - BxNxT binned firing rates. B-time bins, N-neurons, T-trials
%rndoffset - 1xN cell array of integers; each element is in the range [1 B]
%               2xN cell array of integers when rotate_data = 'both_ind'
%stream - stream object for seeding RNG
%
%
%Jon Rueckemann 2025


if nargin<2
    rotate_data='train';
end
if nargin<3
    trlrte2=[];
end
if nargin<4
    rndoffset=[];
end
if nargin<5
    stream=[];
end

assert((isempty(trlrte2)&&...
    (strcmpi(rotate_data,'train')||strcmpi(rotate_data,'none')))||...
    ~isempty(trlrte2),...
    '''train'' or ''none'' are the only valid options for one input');


[n_bins,n_neuron,~]=size(trlrte);
if isempty(rndoffset)
    if ~isempty(stream)
        if strcmpi(rotate_data,'both_ind')
            rndoffset=num2cell(randi(stream,n_bins,[2,n_neuron]));
        elseif ~strcmpi(rotate_data,'none')
            rndoffset=num2cell(randi(stream,n_bins,[1,n_neuron]));
        end
    else
        if strcmpi(rotate_data,'both_ind')
            rndoffset=num2cell(randi(n_bins,[2,n_neuron]));
        elseif ~strcmpi(rotate_data,'none')
            rndoffset=num2cell(randi(n_bins,[1,n_neuron]));
        end
    end
end
assert(size(rndoffset,2)==n_neuron,...
    'length of rndoffset input must match the number of neurons')

%Independently rotate template from training data for each neuron
switch lower(rotate_data)
    case 'train'
        %Rotate test data each neuron independently
        trlrte=num2cell(trlrte,[1 3]);
        trlrte=cellfun(@(x,y) circshift(x,y,1),trlrte,rndoffset,'uni',0);
        trlrte=cell2mat(trlrte); 
    case 'test'       
        %Rotate test data each neuron independently
        trlrte2=num2cell(trlrte2,[1 3]);
        trlrte2=cellfun(@(x,y) circshift(x,y,1),trlrte2,rndoffset,'uni',0);
        trlrte2=cell2mat(trlrte2);        
    case 'both'
        %Rotate test data each neuron independently
        trlrte=num2cell(trlrte,[1 3]);
        trlrte=cellfun(@(x,y) circshift(x,y,1),trlrte,rndoffset,'uni',0);
        trlrte=cell2mat(trlrte);

        %Rotate test data each neuron independently
        trlrte2=num2cell(trlrte2,[1 3]);
        trlrte2=cellfun(@(x,y) circshift(x,y,1),trlrte2,rndoffset,'uni',0);
        trlrte2=cell2mat(trlrte2);
    case 'both_ind'
        %Rotate test data each neuron independently
        trlrte=num2cell(trlrte,[1 3]);
        trlrte=cellfun(@(x,y) circshift(x,y,1),trlrte,....
            rndoffset(1,:),'uni',0);
        trlrte=cell2mat(trlrte);

        %Rotate test data each neuron independently
        trlrte2=num2cell(trlrte2,[1 3]);
        trlrte2=cellfun(@(x,y) circshift(x,y,1),trlrte2,...
            rndoffset(2,:),'uni',0);
        trlrte2=cell2mat(trlrte2);
    case 'none'
        %
    otherwise
        error('Invalid rotation method');
end
end