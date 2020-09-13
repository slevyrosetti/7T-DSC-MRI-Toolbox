function [BAT, TT, TTP, rTTP] = dsc_calculate_BAT_TT_TTP_rTTP(conc,time,injectionTime)

% defaults
if ~exist('TE','var'), TE = 42; end
if ~exist('r2GdInBlood','var'), r2GdInBlood = 3.55; end

           
% define threshold based on peak value
[peak, peakIdx] = max(conc);
if peak ~= 0
    % first positive value
    bolusStart_idx = find(conc>0.00001*peak, 1, 'first');
    % last positive value
    bolusEnd_idx = find(conc>0.00001*peak, 1, 'last');

    % time between injection and bolus arrival (BAT)
    BAT = time(bolusStart_idx) - injectionTime;
    % duration of bolus (Transit Time)
    TT = time(bolusEnd_idx) - time(bolusStart_idx);
    % time between injection and peak (rTTP)
    rTTP = time(peakIdx) - injectionTime;
    % time between bolus arrival and peak (TTP)
    TTP = time(peakIdx) - time(bolusStart_idx);
else
    warning('Peak was 0 (fit probably impossible) >> all metrics were set to 0.');
    BAT = 0; TT = 0; rTTP = 0; TTP = 0;
end

end
