function [imgConcFit,timeForFit,mask] = dsc_fitGamma2imgConc(FnameMatFileConcTime,imgConcFname,maskFname,outFnamePrefix, cropBounds)
%UNTITLED3 Summary of this function goes here
%   time: time in seconds (need to have a regular sampling)

% -------------------------------------------------------------------------
% load data
% -------------------------------------------------------------------------
load(FnameMatFileConcTime);
mask_nii = load_untouch_nii(maskFname);
mask = mask_nii.img;
imgConc_nii = load_untouch_nii(imgConcFname);

% time in seconds starting at 0
time = (timeRegrid - min(timeRegrid))/1000;

% -------------------------------------------------------------------------
% crop signals because of corrupted repetitions (due to trigger missing)
% -------------------------------------------------------------------------
timeForFit = time;
if nargin>=5
    if strcmp(class(cropBounds),'double')
        % crop signal here
        fprintf('\nCrop signal in time from rep %d to rep %d...', cropBounds(1), cropBounds(2));
        imgConcToFit = imgConc(:,:,:,cropBounds(1):cropBounds(2));
        timeForFit = time(cropBounds(1):cropBounds(2));
    else
        % use the cropped signal smoothly performed in
        % dsc_image2concentration.py
        fprintf('\nUse the smoothly cropped signal performed in dsc_image2concentration.py...');
        imgConcToFit = imgConcCrop;
    end
else
    % do not crop signal at all
    fprintf('\nDo not crop signal at all and fit it directly...');
    imgConcToFit = imgConcNoCrop;
end

% -------------------------------------------------------------------------
% fit gamma curve to each voxel in mask
% -------------------------------------------------------------------------
imgConcFit = zeros(size(imgConcToFit));
[xMask, yMask, zMask] = ind2sub(size(mask),find(mask));
progressbar = waitbar(0,['Fitting ' num2str(length(xMask)) ' voxels...']);
for i_vox=1:length(xMask)
    imgConcFit(xMask(i_vox), yMask(i_vox), zMask(i_vox), :) = dsc_fitGamma(timeForFit, squeeze(imgConcToFit(xMask(i_vox), yMask(i_vox), zMask(i_vox), :)));
    waitbar(i_vox/length(xMask),progressbar, [num2str(round(100*i_vox/length(xMask),2)) ' % of voxels processed...'])
end
delete(progressbar);
% -------------------------------------------------------------------------
% save results
% -------------------------------------------------------------------------
% matfile
save([outFnamePrefix 'concFit.mat'], 'imgConcFit', 'timeForFit', 'mask');
% NIFTI file
% origin = [mask_nii.hdr.hist.qoffset_x,mask_nii.hdr.hist.qoffset_y,mask_nii.hdr.hist.qoffset_z];
% imgConcFit_nii = make_nii(imgConcFit, mask_nii.hdr.dime.pixdim(2:4), origin, 64);
% imgConcFit_nii.untouch=1; imgConcFit_nii.hdr.hist.magic=mask_nii.hdr.hist.magic;
imgConcFit_nii = imgConc_nii;
imgConcFit_nii.img = imgConcFit;
imgConcFit_nii.hdr.dime.dim(5) = size(imgConcFit,4);
save_untouch_nii(imgConcFit_nii, [outFnamePrefix 'concFit.nii.gz']);

end

