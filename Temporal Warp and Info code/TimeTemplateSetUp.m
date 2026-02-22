timetemplate=struct();

timetemplate.EventName=...
    [  {'MovementStart'}    {'ChoiceTurn'}  ...
    {'ChoicePath'}    {'OutcomeTurn'}    {'RewardStart'}    ...
    {'PreITIStart'}    {'ITIStart'}    {'TrialStart'}];
timetemplate.duration=[3.75 1.0 3.25 2.0 2.0 1.0 4.0];

timetemplate.bindur=0.25;
timetemplate.favorknot=nan(size(timetemplate.duration));
timetemplate.knot=repmat({[ 0 0 ]}, size(timetemplate.duration));
timetemplate.EventBounds=repmat({[-2 2]},size(timetemplate.EventName));
timetemplate.EventNameF=...
    [{'TrialStart'} {'LastMajorTurn'} {'Reward'}];
timetemplate.EventBoundsF=repmat({[-2 2]},size(timetemplate.EventNameF));