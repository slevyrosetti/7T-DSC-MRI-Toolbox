#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program extracts temporal signal within a given ROI and denoise it from effective TR variations and respiration.


Created on Mon Oct  14 19:21:50 2019

@author: slevy
"""

import dsc_utils
import nibabel as nib
import numpy as np
from scipy.io import savemat
import argparse
import _pickle as pckl
import dsc_pipelines


def main(iFname, maskFname, physioLogFname, oFname, injectionRep, firstPassStartTime, firstPassEndTime, TE, r2GdInBlood):
    """Main."""

    # ----------------------------------------------------------------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------------------------------------------------------------
    img = nib.load(iFname).get_data()  # MRI image
    mask = nib.load(maskFname).get_data()  # masks

    # ----------------------------------------------------------------------------------------------------------------------
    # extract mean in mask
    # ----------------------------------------------------------------------------------------------------------------------
    mriSignal_meanAllSlices, mriSignal_bySlice = dsc_utils.extract_signal_within_roi(img, mask)
    allMRIsignals = np.vstack((mriSignal_meanAllSlices, mriSignal_bySlice.T))

    # ----------------------------------------------------------------------------------------------------------------------
    # Physio processing
    # ----------------------------------------------------------------------------------------------------------------------
    # repsAcqTime: ((SC+all slices) x Nacq x (PulseOx, Resp)
    # timePhysio: N_pulseOx_points x ((PulseOx, Resp)
    # valuesPhysio: N_pulseOx_points x ((PulseOx, Resp)
    repsAcqTime, timePhysio, valuesPhysio = dsc_utils.extract_acqtime_and_physio_by_slice(physioLogFname, img.shape[2], img.shape[3])
    injTime = repsAcqTime[0, injectionRep, 0]
    print('\n\t>> Injection occurs at repetition #%i which happens at %.1f ms.' % (injectionRep, injTime))


    # reorder the acquisition times of each slice according to the acquisition scheme ("interleaved", "ascending",
    # "descending")
    acqScheme = "interleaved"
    if acqScheme == "interleaved":
        actualAcqTime_idx = [0, 2, 1]
    elif acqScheme == "ascending":
        actualAcqTime_idx = [0, 1, 2]
    elif acqScheme == "descending":
        actualAcqTime_idx = [2, 1, 0]
    SlicesAcqTime = repsAcqTime[1:, :, :]
    repsAcqTime[1:, :, :] = SlicesAcqTime[actualAcqTime_idx, :, :]

    # # ----------------------------------------------------------------------------------------------------------------------
    # # Normalize all signals by B1
    # # ----------------------------------------------------------------------------------------------------------------------
    # # extract measured FA for each signal
    # FAmap = nib.load(FAmapFname).get_data()/10.0
    # mgeToInjectSCseg = nib.load(mgeToInjectSCsegFname).get_data()
    # FAmeasuredInWholeMask, FAmeasuredBySlice = dsc_utils.extract_signal_within_roi(FAmap, mgeToInjectSCseg)
    # FAmeasuredAllSignals = np.append(FAmeasuredInWholeMask, FAmeasuredBySlice)
    #
    # # calculate factor to apply to normalize across slices and subjects
    # availableSignalUsed = dsc_utils.calculateB1Factor(timePhysio[:, 0], valuesPhysio[:, 0], FAmeasuredAllSignals, b1mapVoltage, DSCvoltage, selectedFlipAngle)
    #
    # # normalize by fraction of available signal used
    # allMRIsignals_B1norm = np.divide(allMRIsignals, np.tile(availableSignalUsed, (allMRIsignals.shape[1], 1)).T)

    # ----------------------------------------------------------------------------------------------------------------------
    # Normalize all signals by 1 - exp(-TReffective/T1)
    # ----------------------------------------------------------------------------------------------------------------------
    TReff = np.append(np.diff(repsAcqTime[0, :, 0])[0], np.diff(repsAcqTime[0, :, 0]))
    allMRIsignals_TRnorm = np.divide(allMRIsignals, np.tile(1 - np.exp(-TReff/1251), (allMRIsignals.shape[0], 1)))

    # ----------------------------------------------------------------------------------------------------------------------
    # Discard acquisitions with an effective TR of two cardiac cycles (missed a trigger)
    # ----------------------------------------------------------------------------------------------------------------------
    allMRIsignals_TRfiltered, repsAcqTime_PulseOx_TRfiltered, repsAcqTime_Resp_TRfiltered, __, cardiacPeriod = dsc_utils.discardWrongTRs(TReff, timePhysio[:, 0], valuesPhysio[:, 0], allMRIsignals_TRnorm, repsAcqTime[:, :, 0], repsAcqTime[:, :, 1], outPlotFname='')
    repsAcqTime_TRfiltered = np.stack((repsAcqTime_PulseOx_TRfiltered, repsAcqTime_Resp_TRfiltered), axis=2)

    # ----------------------------------------------------------------------------------------------------------------------
    # Regrid physio signals with regular sampling (twice more sampling) except the MRI signal (will be done in the
    # following for loop)
    # ----------------------------------------------------------------------------------------------------------------------
    timePhysioRegrid = np.zeros((2*timePhysio.shape[0], timePhysio.shape[1]))
    valuesPhysioRegrid = np.zeros((2*timePhysio.shape[0], timePhysio.shape[1]))
    # PulseOx
    timePhysioRegrid[:, 0] = np.linspace(np.min(timePhysio[:, 0]), np.max(timePhysio[:, 0]), 2 * timePhysio.shape[0])
    valuesPhysioRegrid[:, 0] = np.interp(timePhysioRegrid[:, 0], timePhysio[:, 0], valuesPhysio[:, 0])
    # Respiration
    timePhysioRegrid[:, 1] = timePhysioRegrid[:, 0]
    valuesPhysioRegrid[:, 1] = np.interp(timePhysioRegrid[:, 1], timePhysio[:, 1], valuesPhysio[:, 1])

    # ----------------------------------------------------------------------------------------------------------------------
    # filter each signal individually
    # ----------------------------------------------------------------------------------------------------------------------
    # signal in whole SC
    # signal0Filtered, signal0FilteredCrop, acqTimeRegrid0 = filterSignal(allMRIsignals_TRfiltered[0, :], repsAcqTime_TRfiltered[0, :, 0], timePhysioRegrid, valuesPhysioRegrid, firstPassStartTime, firstPassEndTime, injTime, cardiacPeriod)
    acqTimeRegrid0, signal0_breathFilt, signal0Filtered, _ = dsc_pipelines.filterSignal(allMRIsignals_TRfiltered[0, :], repsAcqTime_TRfiltered[0, :, 0], timePhysioRegrid, valuesPhysioRegrid, cardiacPeriod)

    signalsFiltered = np.zeros((img.shape[2]+1, signal0Filtered.size))  # (SC+all slices) x time
    # signalsFilteredCrop = np.zeros((img.shape[2]+1, signal0FilteredCrop.size))  # (SC+all slices) x time
    acqTimeRegrid = np.zeros((img.shape[2]+1, signal0Filtered.size))  # (SC+all slices) x time
    # store first subject (already processed)
    signalsFiltered[0, :] = signal0Filtered
    # signalsFilteredCrop[0, :] = signal0FilteredCrop
    acqTimeRegrid[0, :] = acqTimeRegrid0
    # signal in individual slices
    for i_slice in range(img.shape[2]):
        # signalsFiltered[i_slice+1, :], signalsFilteredCrop[i_slice+1, :], acqTimeRegrid[i_slice+1, :] = filterSignal(allMRIsignals_TRfiltered[i_slice+1, :], repsAcqTime_TRfiltered[i_slice+1, :, 0], timePhysioRegrid, valuesPhysioRegrid, firstPassStartTime, firstPassEndTime, injTime, cardiacPeriod)
        acqTimeRegrid[i_slice+1, :], signal_i_slice_breathFilt, signalsFiltered[i_slice+1, :], _ = dsc_pipelines.filterSignal(allMRIsignals_TRfiltered[i_slice+1, :], repsAcqTime_TRfiltered[i_slice+1, :, 0], timePhysioRegrid, valuesPhysioRegrid, cardiacPeriod)

    # ----------------------------------------------------------------------------------------------------------------------
    # Convert signal to concentration in (mmol/L): C(t) = -1/(r*TE)*log(S(t)/S0)
    # ----------------------------------------------------------------------------------------------------------------------
    S0AllSignals = np.zeros(img.shape[2]+1)  # (SC+all slices)
    concAllSignals = np.zeros(signalsFiltered.shape)  # (SC+all slices) x time

    # signal within whole SC
    S0AllSignals[0], concAllSignals[0, :] = convSignalToConc(signalsFiltered[0, :], acqTimeRegrid[0, :], injTime, TE, r2GdInBlood)
    # signal in individual slices
    for i_slice in range(img.shape[2]):
        S0AllSignals[i_slice+1], concAllSignals[i_slice+1, :] = convSignalToConc(signalsFiltered[i_slice+1, :], acqTimeRegrid[i_slice+1, :], injTime, TE, r2GdInBlood)

    # ----------------------------------------------------------------------------------------------------------------------
    # save processed data for further application of DSC models
    # ----------------------------------------------------------------------------------------------------------------------
    savemat(oFname+'.mat', {"allMRIsignals_TRfiltered": allMRIsignals_TRfiltered,
                            "repsAcqTime_TRfiltered": repsAcqTime_TRfiltered,
                            "signalsFiltered": signalsFiltered,
                            # "signalsFilteredCrop": signalsFilteredCrop,
                            "acqTimeRegrid": acqTimeRegrid,
                            "S0AllSignals": S0AllSignals,
                            "concAllSignals": concAllSignals,
                            "injectionRep": injectionRep,
                            "injTime": injTime,
                            "firstPassStartTime": firstPassStartTime,
                            "firstPassEndTime": firstPassEndTime,
                            "TE": TE,
                            "r2GdInBlood": r2GdInBlood})
    pckl.dump({"allMRIsignals_TRfiltered": allMRIsignals_TRfiltered,
                            "repsAcqTime_TRfiltered": repsAcqTime_TRfiltered,
                            "signalsFiltered": signalsFiltered,
                            # "signalsFilteredCrop": signalsFilteredCrop,
                            "acqTimeRegrid": acqTimeRegrid,
                            "S0AllSignals": S0AllSignals,
                            "concAllSignals": concAllSignals,
                            "injectionRep": injectionRep,
                            "injTime": injTime,
                            "firstPassStartTime": firstPassStartTime,
                            "firstPassEndTime": firstPassEndTime,
                            "TE": TE,
                            "r2GdInBlood": r2GdInBlood},
              open(oFname + '.pickle', 'wb'))


def filterSignal(mriSignal, acqTime, timePhysioRegrid, valuesPhysioRegrid, firstPassStartTime, firstPassEndEndTime, injTime, cardiacPeriod):

    # Acquisition times
    acqTimeRegrid = np.linspace(np.min(acqTime), np.max(acqTime), 2 * len(acqTime))
    # regrid MRI signal on regular sampling
    mriSignalRegrid = np.interp(acqTimeRegrid, acqTime, mriSignal)

    # injection rep
    injRepRegrid = np.abs(acqTimeRegrid - injTime).argmin()
    print('\n\t>> Injection occurred at t=%.1fms <=> repetition #%i on regridded time axis.\n\n' % (injTime, injRepRegrid))
    # first pass start rep
    firstPassStartRepRegrid = np.abs(acqTimeRegrid - firstPassStartTime).argmin()
    print('\n\t>> First pass starts at %.1f ms, which corresponds to repetition #%i on the resampled time axis.' % (firstPassStartTime, firstPassStartRepRegrid))
    # first pass last rep
    firstPassEndRepRegrid = np.abs(acqTimeRegrid - firstPassEndEndTime).argmin()
    print('\t>> First pass ends at %.1f ms, which corresponds to repetition #%i on the resampled time axis.\n\n' % (firstPassEndEndTime, firstPassEndRepRegrid))

    # ----------------------------------------------------------------------------------------------------------------------
    # Filter only the baseline to remove high frequencies
    # ----------------------------------------------------------------------------------------------------------------------
    mriSignalRegrid[0:firstPassStartRepRegrid] = dsc_utils.filterHighFreq(mriSignalRegrid[0:firstPassStartRepRegrid], acqTimeRegrid[0:firstPassStartRepRegrid], valuesPhysioRegrid[:, 1], timePhysioRegrid[:, 1], outPlotFname='')

    # ----------------------------------------------------------------------------------------------------------------------
    # Filter signal so as to remove frequencies in the range of the respiration frequency
    # ----------------------------------------------------------------------------------------------------------------------
    mriSignal_filtered, breathingFreqCutOff = dsc_utils.filterResp(mriSignalRegrid, acqTimeRegrid, valuesPhysioRegrid[:, 1], timePhysioRegrid[:, 1], outPlotFname='sliceWiseProcessing_respFilt.png', cardiacPeriod=cardiacPeriod)
    # mriSignal_filtered = dsc_utils.filterHighFreq(mriSignalRegrid, acqTimeRegrid, valuesPhysioRegrid[:, 1], timePhysioRegrid[:, 1], outPlotFname='sliceWiseProcessing_highFreqFilt.png')

    # ----------------------------------------------------------------------------------------------------------------------
    # Smooth signal
    # ----------------------------------------------------------------------------------------------------------------------
    mriSignal_smoothed = dsc_utils.smooth_signal(mriSignal_filtered, outPlotFname='')

    # ----------------------------------------------------------------------------------------------------------------------
    # Smoothly crop signal
    # ----------------------------------------------------------------------------------------------------------------------
    mriSignal_crop, _ = dsc_utils.smoothlyCropSignal(mriSignal_smoothed, firstPassStartRepRegrid, firstPassEndRepRegrid, injRepRegrid)

    return mriSignal_smoothed, mriSignal_crop, acqTimeRegrid


def convSignalToConc(mriSignal, acqTimeRegrid, injTime, TE, r2GdInBlood):

    # ----------------------------------------------------------------------------------------------------------------------
    # baseline last rep
    # ----------------------------------------------------------------------------------------------------------------------
    injRepRegrid = np.abs(acqTimeRegrid - injTime).argmin()

    # ----------------------------------------------------------------------------------------------------------------------
    # Compute baseline (S0)
    # ----------------------------------------------------------------------------------------------------------------------
    S0 = np.mean(mriSignal[0:injRepRegrid])

    # ----------------------------------------------------------------------------------------------------------------------
    # Convert signal to concentration in (mmol/L): C(t) = -1/(r*TE)*log(S(t)/S0)
    # ----------------------------------------------------------------------------------------------------------------------
    conc = - np.log(mriSignal / S0) / (r2GdInBlood * TE / 1000)

    return S0, conc


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program extracts temporal signal within a given ROI and denoise it from effective TR variations and respiration.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-i', dest='iFname', help='Path to MRI data file.', type=str, required=True)
    requiredArgs.add_argument('-m', dest='maskFname', help='NIFTI volume defining the region of interest.', type=str, required=True)
    requiredArgs.add_argument('-l', dest='physioLogFname', help='Basename of physio log for Pulse Ox and Respiration.', type=str, required=True)
    requiredArgs.add_argument('-o', dest='oFname', help='Filename for the output plots and data.', type=str, required=True)

    optionalArgs.add_argument('-inj', dest='injectionRep', help='Number of the repetition when contrast agent injection was launched.', type=int, required=False, default=51)
    optionalArgs.add_argument('-s', dest='firstPassStartTime', help='Start time (on original time grid) of first pass (in seconds).', type=float, required=False, default=50.0)
    optionalArgs.add_argument('-e', dest='firstPassEndTime', help='Time (on original time grid) of first pass end (in seconds).', type=float, required=False, default=71.0)
    optionalArgs.add_argument('-te', dest='TE', help='Echo time in milliseconds.', type=float, required=False, default=43.0)
    # optionalArgs.add_argument('-b1', dest='FAmapFname', help='Path to flip angle map NIFTI.', type=str, required=False, default='')
    # optionalArgs.add_argument('-mgeSeg', dest='mgeToInjectSCsegFname', help='Path to spinal cord segmentation on flip angle map.', type=str, required=False, default='')
    # optionalArgs.add_argument('-FAparams', dest='FAparams', help='Voltage used for flip angle map, voltage used for DSC MRA data, flip angle for DSC MRI data with'
    #                                                              ' format <FA map voltage>,<DSC voltage>,<DSC flip angle>', type=str, required=False, default=',,')
    optionalArgs.add_argument('-r2', dest='r2GdInBlood', help='Transverve relaxivity (in s-1.mmol-1.L = s-1.mM-1) of Gadolinium in blood.'
                                                     ' Default = 3.55 s-1.mmol-1.L [from Proc. Intl. Soc. Mag. Reson. Med. 16 (2008) 1457]', type=float, required=False, default=3.55)
    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()
    # FAparamsArgs = np.array(args.FAparams.strip().split(','), dtype=float)

    # run main
    main(iFname=args.iFname, maskFname=args.maskFname, physioLogFname=args.physioLogFname, oFname=args.oFname, injectionRep=args.injectionRep, firstPassStartTime=1000*args.firstPassStartTime,
         firstPassEndTime=1000*args.firstPassEndTime, TE=args.TE, r2GdInBlood=args.r2GdInBlood)

