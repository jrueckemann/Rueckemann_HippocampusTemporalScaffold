timetemplate=struct();
timetemplate.EventName=...
    [ {'MovementStart'}   {'RewardStart'}];
timetemplate.duration=[10];
timetemplate.lindistscale=true;

% timetemplate.EventName=...
%     [ {'MovementStart'}   {'ITIStart'}    {'TrialStart'}];
% timetemplate.duration=[13.0 4.0];
% timetemplate.EventName=...
%     [ {'TrialStart'}   {'ITIStart'}    {'TrialStart'}];
% timetemplate.duration=[13.5 4.0];
% timetemplate.lindistscale=[true false];
% timetemplate.EventName=...
%     [ {'TrialStart'}    {'MovementStart'}    {'ChoiceTurn'}  ...
%     {'ChoicePath'}    {'OutcomeTurn'}    {'RewardStart'}    ...
%     {'PreITIStart'}    {'ITIStart'}    {'TrialStart'}];

timetemplate.bindur=0.25;
timetemplate.favorknot=nan(size(timetemplate.duration));
timetemplate.knot={[3.75 0]};
%timetemplate.knot={[3.75 0] [ 0 0 ]};
% timetemplate.knot={[4.25 0] [ 0 0 ]};

timetemplate.EventBounds=repmat({[-3 3]},size(timetemplate.duration));
timetemplate.EventNameF=...
    [{'TrialStart'} {'SmallAndLargeTurn'} {'LargeTurn'} {'Reward'}];
timetemplate.EventBoundsF=repmat({[-3 3]},1,4);