#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program denoises all voxels within given ROI from effective TR variations and respiration, and converts them to concentration.


Created on Mon Oct  14 19:21:50 2019

@author: slevy
"""

import dsc_utils
import nibabel as nib
import numpy as np
from scipy.io import savemat
import argparse
import sys
import os

def main(iFname, maskFname, paramFilePath, oFname, r2GdInBlood, physioLogFname, TE, injRep, firstPassStart, firstPassEnd):
    """Main.
    All times are in milliseconds from now on.
    """

    # ----------------------------------------------------------------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------------------------------------------------------------
    img_nii = nib.load(iFname)  # MRI image
    img = np.copy(img_nii.get_data())
    mask_nii = nib.load(maskFname)
    mask = np.copy(mask_nii.get_data())  # masks

    # ----------------------------------------------------------------------------------------------------------------------
    # Physio processing
    # ----------------------------------------------------------------------------------------------------------------------
    if not (physioLogFname and TE and injRep):
        subjID = os.path.abspath(iFname).split('/')[-3]
        baseDir = '/'.join(os.path.abspath(iFname).split('/')[0:-3])
        physioLogFname, TE, injRep, gap, TR, acqTime_firstImg, firstPassStart, firstPassEnd, resolution = dsc_utils.get_physiologFname_TE_injRep(subjID, filename=paramFilePath, baseDir=baseDir)
    repsAcqTime, timePhysio, valuesPhysio = dsc_utils.extract_acqtime_and_physio_by_slice(physioLogFname, img.shape[2], img.shape[3], acqTime_firstImg, TR=TR)
    # repsAcqTime_PulseOx, Time_PulseOx, values_PulseOx, repsAcqTime_Resp, Time_Resp, values_Resp = dsc_utils.extract_acqtime_and_physio(physioLogFname, img.shape[3])
    # reorder the acquisition times of each slice according to the acquisition scheme ("interleaved", "ascending",
    # "descending")
    acqScheme = "interleaved"
    if acqScheme == "interleaved":
        actualAcqTime_idx = [0, 2, 1]
    elif acqScheme == "ascending":
        actualAcqTime_idx = [0, 1, 2]
    elif acqScheme == "descending":
        actualAcqTime_idx = [2, 1, 0]
    slicesAcqTime = repsAcqTime[1:, :, :]
    repsAcqTime[1:, :, :] = slicesAcqTime[actualAcqTime_idx, :, :]

    injTime = repsAcqTime[0, injRep, 0]
    print('\n\t>> Contrast agent infected at repetition #%i <=> t=%.1fms.\n\n' % (injRep, injTime))
    # # ----------------------------------------------------------------------------------------------------------------------
    # # Normalize by B1
    # # ----------------------------------------------------------------------------------------------------------------------
    # # load flip angle map
    # FAmap = nib.load(FAmapFname).get_data()/10.0
    # # calculate factor to apply to normalize across voxels and subjects
    # availableSignalUsed = dsc_utils.calculateB1Factor(Time_PulseOx, values_PulseOx, FAmap, b1mapVoltage, DSCvoltage, selectedFlipAngle)
    # # normalize by fraction of available signal used
    # imgB1norm = np.nan_to_num(img.astype(float) / np.repeat(availableSignalUsed[:, :, :, np.newaxis], img.shape[3], axis=3))

    # ----------------------------------------------------------------------------------------------------------------------
    # Normalize by 1 - exp(-TReffective/T1)
    # ----------------------------------------------------------------------------------------------------------------------
    TReff = np.append(np.diff(repsAcqTime[1, :, 0])[0], np.diff(repsAcqTime[1, :, 0]))
    imgTRnorm = np.divide(img, np.tile(1 - np.exp(-TReff/1251), (img.shape[0], img.shape[1], img.shape[2], 1)))

    # ----------------------------------------------------------------------------------------------------------------------
    # Discard acquisitions with an effective TR of two cardiac cycles (missed a trigger)
    # ----------------------------------------------------------------------------------------------------------------------
    imgTRfiltered, repsAcqTime_PulseOx_TRfiltered, repsAcqTime_Resp_TRfiltered, __, cardiacPeriod = dsc_utils.discardWrongTRs(TReff, timePhysio[:,0], valuesPhysio[:,0], imgTRnorm, repsAcqTime[:,:,0], repsAcqTime[:,:,1])

    # ----------------------------------------------------------------------------------------------------------------------
    # Regrid all signals with regular sampling (twice more sampling) except the MRI signal (will be done in the
    # following for loop)
    # ----------------------------------------------------------------------------------------------------------------------
     # PulseOx
    Time_PulseOxRegrid = np.linspace(np.min(timePhysio[:,0]), np.max(timePhysio[:,0]), 2 * timePhysio.shape[0])
    values_PulseOxRegrid = np.interp(Time_PulseOxRegrid, timePhysio[:,0], valuesPhysio[:,0])
    # Respiration
    Time_RespRegrid = np.linspace(np.min(timePhysio[:,1]), np.max(timePhysio[:,1]), 2 * timePhysio.shape[0])
    values_RespRegrid = np.interp(Time_RespRegrid, timePhysio[:,1], valuesPhysio[:,1])
    # Acquisition times
    TimeRegGrid = np.linspace(np.min(repsAcqTime_PulseOx_TRfiltered, axis=1), np.max(repsAcqTime_PulseOx_TRfiltered, axis=1), 2 * repsAcqTime.shape[1]).T

    # injection rep
    injRepRegrid = np.abs(TimeRegGrid[0, :] - injTime).argmin()
    print('\n\t>> Injection occurred at t=%.1fms <=> repetition #%i on original time axis <=> repetition #%i on regridded time axis.\n\n' %
          (injTime, injRep, injRepRegrid))
    # first pass start rep
    firstPassStartRep = np.abs(repsAcqTime[0, :,0] - firstPassStart).argmin()
    firstPassStartRepRegrid = np.abs(TimeRegGrid[0, :] - firstPassStart).argmin()
    print('\n\t>> First pass starts at t=%.1fms <=> repetition #%i on original time axis <=> repetition #%i on regridded time axis.\n\n' %
          (firstPassStart, firstPassStartRep, firstPassStartRepRegrid))
    # first pass end rep
    firstPassEndRep = np.abs(repsAcqTime[0,:,0] - firstPassEnd).argmin()
    firstPassEndRepRegrid = np.abs(TimeRegGrid[0, :] - firstPassEnd).argmin()
    print('\n\t>> First pass ends at t=%.1fms <=> repetition #%i on original time axis <=> repetition #%i on regridded time axis.\n\n' %
          (firstPassEnd, firstPassEndRep, firstPassEndRepRegrid))


    # ----------------------------------------------------------------------------------------------------------------------
    # for loop on all voxels within mask
    # ----------------------------------------------------------------------------------------------------------------------
    idxROI = np.argwhere(mask == 1)
    imgConcCrop = np.zeros((img.shape[0:3]+(TimeRegGrid.shape[1],)))
    imgConc_cropForAIFdetection = np.zeros((img.shape[0:3]+(TimeRegGrid.shape[1],)))
    imgConcNoCrop = np.zeros((img.shape[0:3]+(TimeRegGrid.shape[1],)))
    imgS0 = np.zeros((img.shape[0:3]))
    nVoxProcessed = 0
    nVoxToProcess = len(idxROI)
    for idx in idxROI:

        mriSignal = imgTRfiltered[idx[0], idx[1], idx[2], :]

        # regrid MRI signal on regular sampling
        mriSignalRegGrid = np.interp(TimeRegGrid[1+idx[2], :], repsAcqTime_PulseOx_TRfiltered[1+idx[2], :], mriSignal)

        # ----------------------------------------------------------------------------------------------------------------------
        # Filter only the baseline to remove high frequencies
        # ----------------------------------------------------------------------------------------------------------------------
        mriSignalRegGrid[0:firstPassStartRepRegrid] = dsc_utils.filterHighFreq(mriSignalRegGrid[0:firstPassStartRepRegrid], TimeRegGrid[1+idx[2], 0:firstPassStartRepRegrid], values_RespRegrid, Time_RespRegrid, '')

        # ----------------------------------------------------------------------------------------------------------------------
        # Filter signal so as to remove frequencies in the range of the respiration frequency
        # ----------------------------------------------------------------------------------------------------------------------
        mriSignal_filtered, breathingFreqCutOff = dsc_utils.filterResp(mriSignalRegGrid, TimeRegGrid[1+idx[2],:], values_RespRegrid, Time_RespRegrid, '', cardiacPeriod)
        # dsc_mean_slices_filtered = dsc_utils.filterHighFreq(dscMeanSlicesRegGrid, dscTimeRegGrid, dscRespSignalRegrid, dscTimeRespRegrid, oFname+'_highFreq_filtering_results.png')

        # ----------------------------------------------------------------------------------------------------------------------
        # Smooth signal
        # ----------------------------------------------------------------------------------------------------------------------
        mriSignal_smoothed = dsc_utils.smooth_signal(mriSignal_filtered, outPlotFname='')

        # ----------------------------------------------------------------------------------------------------------------------
        # Smoothly crop signal
        # ----------------------------------------------------------------------------------------------------------------------
        mriSignal_crop, mriSignal_cropForAIFdetection = dsc_utils.smoothlyCropSignal(mriSignal_smoothed, firstPassStartRepRegrid, firstPassEndRepRegrid, injRepRegrid, outPlotFname='')

        # ----------------------------------------------------------------------------------------------------------------------
        # Compute baseline (S0)
        # ----------------------------------------------------------------------------------------------------------------------
        S0 = np.mean(mriSignal_crop[0:firstPassStartRepRegrid])

        # ----------------------------------------------------------------------------------------------------------------------
        # Convert signal to concentration in (mmol/L): C(t) = -1/(r*TE)*log(S(t)/S0)
        # ----------------------------------------------------------------------------------------------------------------------
        concCrop = - np.log(mriSignal_crop/S0)/(r2GdInBlood*TE/1000)
        conc_cropForAIFdetection = - np.log(mriSignal_cropForAIFdetection/S0)/(r2GdInBlood*TE/1000)
        concNoCrop = - np.log(mriSignal_smoothed/S0)/(r2GdInBlood*TE/1000)

        # ----------------------------------------------------------------------------------------------------------------------
        # Save to new matrix
        # ----------------------------------------------------------------------------------------------------------------------
        imgS0[idx[0], idx[1], idx[2]] = S0
        imgConcCrop[idx[0], idx[1], idx[2], :] = concCrop
        imgConc_cropForAIFdetection[idx[0], idx[1], idx[2], :] = conc_cropForAIFdetection
        imgConcNoCrop[idx[0], idx[1], idx[2], :] = concNoCrop
        nVoxProcessed += 1
        sys.stdout.flush()
        print('{}% of voxels processed...'.format(round(100*nVoxProcessed/nVoxToProcess, 2)), end="\r")

    # ----------------------------------------------------------------------------------------------------------------------
    # Save as NIFTI
    # ----------------------------------------------------------------------------------------------------------------------
    dsc_utils.saveAsNifti(img_nii, imgConcCrop, oFname, dataType=np.float64)
    dsc_utils.saveAsNifti(mask_nii, imgS0, oFname+'_S0', dataType=np.float64)

    # ----------------------------------------------------------------------------------------------------------------------
    # Also save .mat file for further use in Matlab
    # ----------------------------------------------------------------------------------------------------------------------
    # ***** FORGET THE SHIFT OF ACQUISITION TIME BETWEEN THE DIFFERENT SLICES FROM NOW ON (FOR FURTHER CALCULATION *****
    # *****                          OF BOLUS ARRIVAL TIME AND TIME-TO-PEAK MAPS                                   *****
    savemat(oFname+'.mat', {"imgConcCrop": imgConcCrop,
                            "imgConc_cropForAIFdetection": imgConc_cropForAIFdetection,
                            "imgConcNoCrop": imgConcNoCrop,
                            "timeRegrid": TimeRegGrid[0, :],
                            "imgS0": imgS0,
                            "injTime": injTime,
                            "firstPassStartRep": firstPassStartRep,
                            "firstPassStart": firstPassStart,
                            "firstPassStartRepRegrid": firstPassStartRepRegrid,
                            "firstPassEndRep": firstPassEndRep,
                            "firstPassEnd": firstPassEnd,
                            "firstPassEndRepRegrid": firstPassEndRepRegrid,
                            "TE": TE,
                            "r2GdInBlood": r2GdInBlood})

    print('\n=== All done! ===')


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program denoises all voxels within given ROI from effective TR variations and respiration, and converts them to contrast agent concentration.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-i', dest='iFname', help='Path to MRI data file.', type=str, required=True)
    requiredArgs.add_argument('-m', dest='maskFname', help='NIFTI volume defining the region of interest.', type=str, required=True)
    requiredArgs.add_argument('-o', dest='oFname', help='Filename for output image and mat file.', type=str, required=True)

    optionalArgs.add_argument('-param', dest='paramFilePath', help='Path to file giving specific parameters (injection repetition, dicom path).', type=str, required=False, default='')
    optionalArgs.add_argument('-physio', dest='physioLogFname', help='Basename of physio log for Pulse Ox and Respiration.', type=str, required=False, default='')
    optionalArgs.add_argument('-inj', dest='injRep', help='Number of the repetition when contrast agent injection was launched.', type=int, required=False, default=0)
    optionalArgs.add_argument('-s', dest='firstPassStartTime', help='Start time (on original time grid) of first pass (in seconds).', type=float, required=False, default=50.0)
    optionalArgs.add_argument('-e', dest='firstPassEndTime', help='Time (on original time grid) of first pass end (in seconds).', type=float, required=False, default=71.0)
    optionalArgs.add_argument('-te', dest='TE', help='Echo time in milliseconds.', type=float, required=False, default=0.0)
    optionalArgs.add_argument('-r2', dest='r2GdInBlood', help='Transverve relaxivity (in s-1.mmol-1.L = s-1.mM-1) of Gadolinium in blood.'
                                                     ' Default = 3.55 s-1.mmol-1.L [from Proc. Intl. Soc. Mag. Reson. Med. 16 (2008) 1457]', type=float, required=False, default=3.55)
    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()
    # FAparamsArgs = np.array(args.FAparams.strip().split(','), dtype=float)

    # run main
    main(iFname=args.iFname, maskFname=args.maskFname, oFname=args.oFname, r2GdInBlood=args.r2GdInBlood, physioLogFname=args.physioLogFname, injRep=args.injRep, firstPassStart=1000*args.firstPassStartTime,
         firstPassEnd=1000*args.firstPassEndTime, TE=args.TE, paramFilePath=args.paramFilePath)
