timetemplate=struct();
timetemplate.EventName=...
    [{'MovementStart'} {'PreITIStart'} {'ITIStart'} {'TrialStart'}];
timetemplate.duration=[12.0 1.0 4.0];
% timetemplate.EventName=...
%     [ {'TrialStart'}    {'MovementStart'}    {'ChoiceTurn'}  ...
%     {'ChoicePath'}    {'OutcomeTurn'}    {'RewardStart'}    ...
%     {'PreITIStart'}    {'ITIStart'}    {'TrialStart'}];
% timetemplate.duration=[0.5 3.75 1.0 3.25 2.0 2.0 1.0 4.0];

timetemplate.bindur=0.25;
timetemplate.favorknot=nan(size(timetemplate.duration));
timetemplate.knot=repmat({[ 0 0 ]}, size(timetemplate.duration));
timetemplate.EventBounds=repmat({[-3 3]},size(timetemplate.duration));
timetemplate.EventNameF=...
    [{'TrialStart'} {'SmallAndLargeTurn'} {'LargeTurn'} {'Reward'}];
timetemplate.EventBoundsF=repmat({[-3 3]},1,4);