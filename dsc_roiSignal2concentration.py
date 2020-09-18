#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program extracts temporal signal within a given ROI and applies the implemented signal processing
pipeline (effective TR correction, discard missed triggers data, breathing frequencies filtering, smoothing).


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
import os
import matplotlib.pyplot as plt


def main(iFname, maskFname, physioLogFname, oFname, injRep, firstPassStartTime, firstPassEndTime, TE, r2GdInBlood, paramFilePath):
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
    if not (injRep and physioLogFname and TE):
        # get file name of physiolog and other parameters automatically
        subjID = os.path.abspath(iFname).split('/')[-3]
        baseDir = '/'.join(os.path.abspath(iFname).split('/')[0:-3])
        physioLogFname, TE, injRep, gap, TR, acqTime_firstImg, firstPassStart, firstPassEnd, resolution = dsc_utils.get_physiologFname_TE_injRep(subjID, filename=paramFilePath, baseDir=baseDir)

    # process phyiolog informations
    # repsAcqTime: ((SC+all slices) x Nacq x (PulseOx, Resp)
    # timePhysio: N_pulseOx_points x ((PulseOx, Resp)
    # valuesPhysio: N_pulseOx_points x ((PulseOx, Resp)
    repsAcqTime, timePhysio, valuesPhysio = dsc_utils.extract_acqtime_and_physio_by_slice(physioLogFname, img.shape[2], img.shape[3], acqTime_firstImg)

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

    # ----------------------------------------------------------------------------------------------------------------------
    # Normalize all signals by 1 - exp(-TReffective/T1)
    # ----------------------------------------------------------------------------------------------------------------------
    TReff = np.append(np.diff(repsAcqTime[0, :, 0])[0], np.diff(repsAcqTime[0, :, 0]))
    allMRIsignals_TRnorm = np.divide(allMRIsignals, np.tile(1 - np.exp(-TReff/1251), (allMRIsignals.shape[0], 1)))

    # ----------------------------------------------------------------------------------------------------------------------
    # Discard acquisitions with an effective TR of two cardiac cycles (missed a trigger)
    # ----------------------------------------------------------------------------------------------------------------------
    allMRIsignals_TRfiltered, repsAcqTime_PulseOx_TRfiltered, repsAcqTime_Resp_TRfiltered, idxAcqToDiscard, cardiacPeriod = dsc_utils.discardWrongTRs(TReff, timePhysio[:, 0], valuesPhysio[:, 0], allMRIsignals_TRnorm, repsAcqTime[:, :, 0], repsAcqTime[:, :, 1], outPlotFname='')
    repsAcqTime_TRfiltered = np.stack((repsAcqTime_PulseOx_TRfiltered, repsAcqTime_Resp_TRfiltered), axis=2)

    # ----------------------------------------------------------------------------------------------------------------------
    # Regrid physio signals with regular sampling (twice more sampling) except the MRI signal (will be done in the
    # following for loop)
    # ----------------------------------------------------------------------------------------------------------------------
    interpFactor = 1
    timePhysioRegrid = np.zeros((interpFactor*timePhysio.shape[0], timePhysio.shape[1]))
    valuesPhysioRegrid = np.zeros((interpFactor*timePhysio.shape[0], timePhysio.shape[1]))
    # PulseOx
    timePhysioRegrid[:, 0] = np.linspace(np.min(timePhysio[:, 0]), np.max(timePhysio[:, 0]), interpFactor * timePhysio.shape[0])
    valuesPhysioRegrid[:, 0] = np.interp(timePhysioRegrid[:, 0], timePhysio[:, 0], valuesPhysio[:, 0])
    # Respiration
    timePhysioRegrid[:, 1] = timePhysioRegrid[:, 0]
    valuesPhysioRegrid[:, 1] = np.interp(timePhysioRegrid[:, 1], timePhysio[:, 1], valuesPhysio[:, 1])

    # ----------------------------------------------------------------------------------------------------------------------
    # filter each signal individually
    # ----------------------------------------------------------------------------------------------------------------------
    # signal in whole SC
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
        acqTimeRegrid[i_slice+1, :], signal_i_slice_breathFilt, signalsFiltered[i_slice+1, :], _ = dsc_pipelines.filterSignal(allMRIsignals_TRfiltered[i_slice+1, :], repsAcqTime_TRfiltered[i_slice+1, :, 0], timePhysioRegrid, valuesPhysioRegrid, cardiacPeriod)

    # injection time
    injRepRegrid = np.abs(acqTimeRegrid - injTime).argmin(axis=-1)

    # # ----------------------------------------------------------------------------------------------------------------------
    # # Convert signal to concentration in (mmol/L): C(t) = -1/(r*TE)*log(S(t)/S0)
    # # ----------------------------------------------------------------------------------------------------------------------
    # S0AllSignals = np.zeros(img.shape[2]+1)  # (SC+all slices)
    # concAllSignals = np.zeros(signalsFiltered.shape)  # (SC+all slices) x time
    #
    # # signal within whole SC
    # S0AllSignals[0], concAllSignals[0, :] = convSignalToConc(signalsFiltered[0, :], acqTimeRegrid[0, :], injTime, TE, r2GdInBlood)
    # # signal in individual slices
    # for i_slice in range(img.shape[2]):
    #     S0AllSignals[i_slice+1], concAllSignals[i_slice+1, :] = convSignalToConc(signalsFiltered[i_slice+1, :], acqTimeRegrid[i_slice+1, :], injTime, TE, r2GdInBlood)

    # ----------------------------------------------------------------------------------------------------------------------
    # Convert signal to concentration in (mmol/L): C(t) = -1/(r*TE)*log(S(t)/S0)
    # ----------------------------------------------------------------------------------------------------------------------
    concAllSignals, _, S0AllSignals = dsc_utils.calculateDeltaR2(signalsFiltered, TE, injRepRegrid, r2GdInBlood)



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
                            "injectionRep": injRep,
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
                            "injectionRep": injRep,
                            "injTime": injTime,
                            "firstPassStartTime": firstPassStartTime,
                            "firstPassEndTime": firstPassEndTime,
                            "TE": TE,
                            "r2GdInBlood": r2GdInBlood},
              open(oFname + '.pickle', 'wb'))

    # ----------------------------------------------------------------------------------------------------------------------
    # plot results
    # ----------------------------------------------------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(17, 9.5))
    plt.subplots_adjust(wspace=0.07, left=0.25, right=0.99, hspace=0.2, bottom=0.07, top=0.89)
    # Signal by slice and all slices averaged
    ylims = dsc_utils.plot_DeltaR2_perSlice(acqTimeRegrid/1000, signalsFiltered, TE, axes[0], xlabel='Time (s)', lateralTitle='Slice-wise\nprofile', injTime=injTime/1000, stats=True, cardiacPeriod=cardiacPeriod/1000, timeAcqToDiscard=repsAcqTime[0, idxAcqToDiscard, 0]/1000)
    # Signal from all slices averaged
    ylims = dsc_utils.plot_DeltaR2_perSlice(acqTimeRegrid[np.newaxis, 0, :]/1000, signalsFiltered[np.newaxis, 0, :], TE, axes[1], xlabel='Time (s)', lateralTitle='Average in ROI\n(all slices averaged)', injTime=injTime/1000, stats=False, signalLabels=[], cardiacPeriod=cardiacPeriod/1000, timeAcqToDiscard=repsAcqTime[0, idxAcqToDiscard, 0]/1000, ylims=ylims)
    fig.savefig(oFname+'.pdf', transparent=True)
    plt.show(fig)


# def convSignalToConc(mriSignal, acqTimeRegrid, injTime, TE, r2GdInBlood):
#
#     # ----------------------------------------------------------------------------------------------------------------------
#     # baseline last rep
#     # ----------------------------------------------------------------------------------------------------------------------
#     injRepRegrid = np.abs(acqTimeRegrid - injTime).argmin()
#
#     # ----------------------------------------------------------------------------------------------------------------------
#     # Compute baseline (S0)
#     # ----------------------------------------------------------------------------------------------------------------------
#     S0 = np.mean(mriSignal[0:injRepRegrid])
#
#     # ----------------------------------------------------------------------------------------------------------------------
#     # Convert signal to concentration in (mmol/L): C(t) = -1/(r*TE)*log(S(t)/S0)
#     # ----------------------------------------------------------------------------------------------------------------------
#     conc = - np.log(mriSignal / S0) / (r2GdInBlood * TE / 1000)
#
#     return S0, conc


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program extracts temporal signal within a given ROI and applies '
                                                 'the implemented signal processing pipeline (effective TR correction, '
                                                 'discard missed triggers data, breathing frequencies filtering, '
                                                 'smoothing).')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-i', dest='iFname', help='Path to MRI data file.', type=str, required=True)
    requiredArgs.add_argument('-m', dest='maskFname', help='NIFTI volume defining the region of interest.', type=str, required=True)
    requiredArgs.add_argument('-o', dest='oFname', help='Filename for the output plots and data.', type=str, required=True)

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
    main(iFname=args.iFname, maskFname=args.maskFname, oFname=args.oFname, physioLogFname=args.physioLogFname, injRep=args.injRep, firstPassStartTime=1000*args.firstPassStartTime,
         firstPassEndTime=1000*args.firstPassEndTime, TE=args.TE, r2GdInBlood=args.r2GdInBlood, paramFilePath=args.paramFilePath)

