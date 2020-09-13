function [signalsConcFit,time] = dsc_fitGamma2listConc(FnameMatFileConcTime,outFname, limitCropInjTime)
%UNTITLED3 Summary of this function goes here
%   time: time in seconds (need to have a regular sampling)

% -------------------------------------------------------------------------
% default parameter values
% -------------------------------------------------------------------------
if nargin<3
    limitCropInjTime = false;
end

% -------------------------------------------------------------------------
% load data
% -------------------------------------------------------------------------
load(FnameMatFileConcTime, 'concAllSignals', 'acqTimeRegrid', 'injTime', 'TE', 'r2GdInBlood', 'firstPassStartTime', 'firstPassEndTime');

% duration to keep after injection
upperBoundCropDelay = 1.3*(firstPassEndTime - injTime)/1000;
lowerBoundCropDelay = 4.0*(firstPassEndTime - firstPassStartTime)/1000; % with respect to first pass end

% time in seconds starting at 0
time = (acqTimeRegrid - repmat(min(acqTimeRegrid,[],2), 1, size(acqTimeRegrid,2)))/1000;
injTime = (injTime - min(acqTimeRegrid(:)))/1000;
firstPassEndTime = (firstPassEndTime - min(acqTimeRegrid(:)))/1000;

% -------------------------------------------------------------------------
% crop signals after bolus because of big differences in steady-stade
% signal
% -------------------------------------------------------------------------
[ ~, upperCropBound] = min( abs( time(1,:) - (injTime+upperBoundCropDelay) ) );
[ ~, lowerCropBound] = min( abs( time(1,:) - (firstPassEndTime-lowerBoundCropDelay) ) );
if limitCropInjTime
    [ ~, injRep] = min( abs( time(1,:) - injTime ) );
    lowerCropBound = max([lowerCropBound, injRep]);
end
concAllSignalsCrop = concAllSignals(:, lowerCropBound:upperCropBound);
timeCrop = time(:, lowerCropBound:upperCropBound);


% -------------------------------------------------------------------------
% fit gamma curve to each signal in array
% -------------------------------------------------------------------------
signalsConcFit = zeros(size(concAllSignals));
progressbar = waitbar(0,['Fitting ' num2str(size(concAllSignals,1)) ' signals...']);
for i_sig=1:size(concAllSignals,1)
    % fit cropped signal
    signalCropFit = dsc_fitGamma(timeCrop(i_sig, :), concAllSignalsCrop(i_sig, :));
    % extend signal before and after cropping
    signalsConcFit(i_sig, upperCropBound:end) = signalCropFit(end)*ones(1,size(concAllSignals,2)-(upperCropBound-1));
    signalsConcFit(i_sig, 1:lowerCropBound) = signalCropFit(1)*ones(1,lowerCropBound);
    signalsConcFit(i_sig, lowerCropBound:upperCropBound) = signalCropFit;
    
    waitbar(i_sig/size(concAllSignalsCrop,1),progressbar, [num2str(round(100*i_sig/size(concAllSignalsCrop,1),2)) ' % of signals processed...'])
end
delete(progressbar);

% -------------------------------------------------------------------------
% calculate relative metrics
% -------------------------------------------------------------------------
PR = 100*(1 - min(exp(-r2GdInBlood*TE*concAllSignals/1000), [], 2));
BAT = zeros(size(concAllSignals,1),1);
TT = zeros(size(concAllSignals,1),1);
TTP = zeros(size(concAllSignals,1),1);
rTTP = zeros(size(concAllSignals,1),1);
rBV = zeros(size(concAllSignals,1),1);
rBF = zeros(size(concAllSignals,1),1);
rMTT = zeros(size(concAllSignals,1),1);
for i_sig=1:size(concAllSignals,1)
    [BAT(i_sig), TT(i_sig), TTP(i_sig), rTTP(i_sig)] = dsc_calculate_BAT_TT_TTP_rTTP(signalsConcFit(i_sig, :),time(i_sig, :),injTime);
    [rBV(i_sig), rBF(i_sig), rMTT(i_sig)] = dsc_calculate_rBV_rBF_rMTT(signalsConcFit(i_sig, :), time(i_sig, :));
end

% -------------------------------------------------------------------------
% save results
% -------------------------------------------------------------------------
% matfile
save([outFname '.mat'], 'signalsConcFit', 'injTime', 'TE', 'r2GdInBlood', 'time', 'acqTimeRegrid', 'PR', 'BAT', 'TT', 'TTP', 'rTTP', 'rBV', 'rBF', 'rMTT');

end

