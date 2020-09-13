function [cbv,cbv_lc,cbf,mtt,TTP,fwhm] = dsc_calculatePerfMetricsVoxelWise(ConcConvertMatFile,ConcFitMatFile,AIFfitMatFile,outFolderFname,maskFname,originalImgFname)
%UNTITLED4 Summary of this function goes here
%   All times must be in seconds and starting at 0 in this script.

%----- load data ----------------------------------------------------------
load(ConcConvertMatFile, 'imgS0', 'firstPassStartRepRegrid', 'timeRegrid', 'injTime', 'TE', 'r2GdInBlood');
load(ConcFitMatFile, 'imgConcFit', 'timeForFit', 'mask');
% load(AIFfitMatFile, 'AIFfit','AIFconc','AIFvoxels');
injTime = (injTime - min(timeRegrid))/1000;  % set to ms and start to 0

%----- set DSC-MRI-TOOLBOX parameters -------------------------------------
options = DSC_mri_getOptions();
% options.deconv.method = {'SSD'}; % choose the deconvolution methods to apply for CBF calculation
options.time = timeForFit;
options.tr = timeForFit(3)-timeForFit(2);
[options.nR,options.nC,options.nS,options.nT]=size(imgConcFit);
maskFormat.data = double(mask);
% [~,~,baselineEndRep]=unique(round(abs(time-baselineEndTime)),'stable');
bolus = ones(size(imgConcFit,3),1)*(double(firstPassStartRepRegrid)+1); % index of the beginning of the bolus for all slices

%----- "relative" perfusion metrics (without AIF) -------------------------
[rBV, rBF]=dsc_map_rBV_rBF(imgConcFit,maskFormat,options);
rMTT=rBV./rBF;
[BAT, TT, TTP, rTTP, PR]=dsc_map_BAT_TT_TTP_rTTP_PR(imgConcFit,maskFormat,options,injTime, TE, r2GdInBlood);

%----- CBV calculation ---------------------------------------------------
% [cbv]=DSC_mri_cbv(imgConcFit,AIFfit,maskFormat,options);

%----- CBV leackage correction  ------------------------------------------
% [cbv_lc,~,K2_map,~,~]=DSC_mri_cbv_lc(imgConcFit,AIFfit,maskFormat,bolus,options);

%----- CBF calculation ---------------------------------------------------
% [cbf]=DSC_mri_cbf(imgConcFit,AIFfit,maskFormat,options);

%----- MTT calculation ---------------------------------------------------
% [mtt]=DSC_mri_mtt(cbv_lc,cbf,options);

%----- TTP calculation ---------------------------------------------------
% [ttp]=DSC_mri_ttp(imgConcFit,maskFormat.data,options);
% [fwhm]=DSC_mri_fwhm(imgConcFit,maskFormat.data,options);

%----- save results as NIFTI file ----------------------------------------- 
mkdir(outFolderFname)
% rCBV
saveAsNifti(maskFname, rBV, [outFolderFname '/rBV.nii.gz']);
saveAsNifti(maskFname, rBF, [outFolderFname '/rBF.nii.gz']);
saveAsNifti(maskFname, rMTT, [outFolderFname '/rMTT.nii.gz']);
saveAsNifti(maskFname, TT, [outFolderFname '/TT.nii.gz']);
saveAsNifti(maskFname, BAT, [outFolderFname '/BAT.nii.gz']);
saveAsNifti(maskFname, TTP, [outFolderFname '/TTP.nii.gz']);
saveAsNifti(maskFname, rTTP, [outFolderFname '/rTTP.nii.gz']);
saveAsNifti(maskFname, PR, [outFolderFname '/PR.nii.gz']);
% CBV
% saveAsNifti(maskFname, cbv, [outFolderFname '/CBV.nii.gz']);
% CBV with leakage correction
% saveAsNifti(maskFname, cbv_lc, [outFolderFname '/CBVlc.nii.gz']);
% CBF
% saveAsNifti(maskFname, cbf.svd.map, [outFolderFname '/CBFsvd.nii.gz']);
% saveAsNifti(maskFname, cbf.csvd.map, [outFolderFname '/CBFcsvd.nii.gz']);
% saveAsNifti(maskFname, cbf.osvd.map, [outFolderFname '/CBFosvd.nii.gz']);
% saveAsNifti(maskFname, cbf.ssd.map, [outFolderFname '/CBFssd.nii.gz']);
% MTT
% saveAsNifti(maskFname, mtt.svd, [outFolderFname '/MTTsvd.nii.gz']);
% saveAsNifti(maskFname, mtt.csvd, [outFolderFname '/MTTcsvd.nii.gz']);
% saveAsNifti(maskFname, mtt.osvd, [outFolderFname '/MTTosvd.nii.gz']);
% TTP
% saveAsNifti(maskFname, TTP, [outFolderFname '/TTP.nii.gz']);
% Residue function
% saveAsNifti(originalImgFname, cbf.svd.residual, [outFolderFname '/RESsvd.nii.gz']);
% saveAsNifti(originalImgFname, cbf.csvd.residual, [outFolderFname '/REScsvd.nii.gz']);
% saveAsNifti(originalImgFname, cbf.osvd.residual, [outFolderFname '/RESosvd.nii.gz']);
% saveAsNifti(originalImgFname, cbf.ssd.residual, [outFolderFname '/RESssd.nii.gz']);

% ------  View Results --------------------------------------------------- 
% aifTOOLBOXstruct.conc = AIFconc;
% aifTOOLBOXstruct.voxels = AIFvoxels;
% aifTOOLBOXstruct.fit.gv = AIFfit;
% aifTOOLBOXstruct.fit.time = timeForFit;
% DSC_mri_show_results(cbv_lc,cbf,mtt,ttp,maskFormat.data,aifTOOLBOXstruct,imgConcFit,imgS0);

end

