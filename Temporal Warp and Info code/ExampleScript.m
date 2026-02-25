%The following short script implements the key analyses for individual
%neurons, converting spike and event timestamps to rate maps that span a
%standardized trial in "The Primate Hippocampus Constructs a Temporal 
% Scaffold Anchored to Behavioral Events".

load('Neuron_ExampleData.mat') %alter for local path

TimeTemplateSetUp %Create 'timetemplate'

nulliter=1000; %Iterations composing the null distribution for each neuron
spkstruct=cell(numel(fig1_neurons),1); %preallocate

for m=1:numel(fig1_neurons)
    [spkstruct{m},~,~,~,~]=ymazemodule_warp2(...
        fig1_neurons(m),{fig1_neurons(m).evt},timetemplate,nulliter,[],[]);
end
spkstruct=cell2mat(spkstruct);