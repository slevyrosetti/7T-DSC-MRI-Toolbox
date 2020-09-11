#!/usr/bin/env python3

"""
This file contains DSC pipeline functions.


Created on Mon Apr 27 12:44:01 2020

@author: slevy
"""

import nibabel as nib
import dsc_utils
import numpy as np

#%%
def computeDeltaR2withinROI(iFname, maskFname, physioLogFname, TE):
    """
    1) Load data
    2) Extract mean in mask
    3) Extract physiological data (pulseOx, breathing) + acquisition times
    4) Normalize by effective TR
    5) Filter out missed trigger points and the following
    6) Compute ∆R2

    :param iFname:
    :param maskFname:
    :param physioLogFname:
    :param TE: in milliseconds
    :return:
    """

    # ----------------------------------------------------------------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------------------------------------------------------------
    img = nib.load(iFname).get_data()  # MRI image
    mask = nib.load(maskFname).get_data()  # masks

    # ----------------------------------------------------------------------------------------------------------------------
    # extract mean in mask
    # ----------------------------------------------------------------------------------------------------------------------
    mriSignal_along_reps, mriSignal_along_reps_by_slice = dsc_utils.extract_signal_within_roi(img, mask)
    print('\nInitial data\n------------')
    cov0 = dsc_utils.get_temporalCOV(mriSignal_along_reps)

    # ----------------------------------------------------------------------------------------------------------------------
    # Physio processing
    # ----------------------------------------------------------------------------------------------------------------------
    # ***** DOES NOT TAKE INTO ACCOUNT THE SHIFT OF ACQUISITION TIME BETWEEN THE DIFFERENT SLICES *****
    repsAcqTime_PulseOx, Time_PulseOx, values_PulseOx, repsAcqTime_Resp, Time_Resp, values_Resp = dsc_utils.extract_acqtime_and_physio(physioLogFname, img.shape[3], physioplot_out_fname='')

    # ----------------------------------------------------------------------------------------------------------------------
    # Normalize by 1 - exp(-TReffective/T1)
    # ----------------------------------------------------------------------------------------------------------------------
    TReff_dsc = np.append(np.diff(repsAcqTime_PulseOx)[0], np.diff(repsAcqTime_PulseOx))
    print('Mean RR = '+str(np.mean(TReff_dsc))+' ms')
    dsc_mean_slices = np.divide(mriSignal_along_reps, 1 - np.exp(-TReff_dsc/1251))
    print('\nAfter division by 1 - exp(-effective TR / T1)\n-------------------')
    cov_TRnorm = dsc_utils.get_temporalCOV(dsc_mean_slices)

    # ----------------------------------------------------------------------------------------------------------------------
    # Filter out points where TR was too long (missed a trigger)
    # ----------------------------------------------------------------------------------------------------------------------
    dsc_mean_slices_TRfiltered, repsAcqTime_PulseOx_TRfiltered, repsAcqTime_Resp_TRfiltered, __, cardiacPeriod = dsc_utils.discardWrongTRs(TReff_dsc, Time_PulseOx, values_PulseOx, dsc_mean_slices, repsAcqTime_PulseOx, repsAcqTime_Resp, outPlotFname='')
    print('\nAfter discarding missed triggers\n-------------------')
    cov_missedTrig = dsc_utils.get_temporalCOV(dsc_mean_slices_TRfiltered)

    # ----------------------------------------------------------------------------------------------------------------------
    # Regrid all signals with regular sampling (twice more sampling)
    # ----------------------------------------------------------------------------------------------------------------------
    # MRI
    dscTimeRegGrid = np.linspace(np.min(repsAcqTime_PulseOx), np.max(repsAcqTime_PulseOx), 2 * len(repsAcqTime_PulseOx))
    dscMeanSlicesRegGrid = np.interp(dscTimeRegGrid, repsAcqTime_PulseOx_TRfiltered, dsc_mean_slices_TRfiltered)
    # PulseOx
    dscTimePulseOxRegrid = np.linspace(np.min(Time_PulseOx), np.max(Time_PulseOx), 2 * len(Time_PulseOx))
    dscPulseOxSignalRegrid = np.interp(dscTimePulseOxRegrid, Time_PulseOx, values_PulseOx)
    # Respiration
    dscTimeRespRegrid = np.linspace(np.min(Time_Resp), np.max(Time_Resp), 2 * len(Time_Resp))
    dscRespSignalRegrid = np.interp(dscTimeRespRegrid, Time_Resp, values_Resp)

    # ----------------------------------------------------------------------------------------------------------------------
    # Remove all frequencies found when no injection (RVS acquisition) in Fourier domain
    # ----------------------------------------------------------------------------------------------------------------------
    # if rvsFname:
    #     samplingFreq = 1000/(dscTimeRegGrid[2] - dscTimeRegGrid[1])
    #     dscMeanSlicesRegGrid = dsc_utils.removeRVSfreq(dscMeanSlicesRegGrid, samplingFreq, rvsFname, rvsMaskname, rvsPhysioLogFname, 'filteringBasedOnRVSdata.png')

    # ----------------------------------------------------------------------------------------------------------------------
    # Filter only the baseline to remove high frequencies
    # ----------------------------------------------------------------------------------------------------------------------
    # baselineLastRepRegrid = np.abs(dscTimeRegGrid - baselineEndTime).argmin()
    # print(
    #     '\n\t>> Baseline ends at repetition #%i which happens at %f ms and corresponds to repetition #%i on the regridded time axis.\n\n' % (
    #     injectionRep, baselineEndTime, baselineLastRepRegrid))
    # baseline_filtered = dsc_utils.filterHighFreq(dscMeanSlicesRegGrid[0:baselineLastRepRegrid],
    #                                              dscTimeRegGrid[0:baselineLastRepRegrid], dscRespSignalRegrid,
    #                                              dscTimeRespRegrid, oFname + '_baseline_filtering.png')
    # dscMeanSlicesRegGrid[0:baselineLastRepRegrid] = baseline_filtered

    # ----------------------------------------------------------------------------------------------------------------------
    # Filter signal so as to remove frequencies in the range of the respiration frequency
    # ----------------------------------------------------------------------------------------------------------------------
    dsc_mean_slices_filtered = dsc_utils.filterResp(dscMeanSlicesRegGrid, dscTimeRegGrid, dscRespSignalRegrid, dscTimeRespRegrid, outPlotFname='', cardiacPeriod=cardiacPeriod)
    # dsc_mean_slices_filtered = dsc_utils.filterHighFreq(dscMeanSlicesRegGrid, dscTimeRegGrid, dscRespSignalRegrid, dscTimeRespRegrid, oFname+'_highFreq_filtering_results.png')
    # dsc_mean_slices_filtered = dscMeanSlicesRegGrid
    print('\nAfter breathing frequencies filtering\n-------------------')
    cov_breathFilt = dsc_utils.get_temporalCOV(dsc_mean_slices_filtered)

    # ----------------------------------------------------------------------------------------------------------------------
    # Smooth signal
    # ----------------------------------------------------------------------------------------------------------------------
    dsc_mean_slices_smoothed = dsc_utils.smooth_signal(dsc_mean_slices_filtered, outPlotFname='')
    print('\nAfter final smoothing\n-------------------')
    cov_finalSmooth = dsc_utils.get_temporalCOV(dsc_mean_slices_smoothed)

    # ----------------------------------------------------------------------------------------------------------------------
    # Smoothly crop signal
    # ----------------------------------------------------------------------------------------------------------------------
    # firstPassEndTime = repsAcqTime_PulseOx[firstPassEndRep]
    # firstPassEndRepRegrid = np.abs(dscTimeRegGrid - firstPassEndTime).argmin()
    # if firstPassEndRep:
    #     dsc_mean_slices_crop = dsc_utils.smoothlyCropSignal(dsc_mean_slices_smoothed, baselineLastRepRegrid,
    #                                                         firstPassEndRepRegrid, 'smoothCrop_results.png')
    # else:
    #     dsc_mean_slices_crop = dsc_mean_slices_smoothed

    # ----------------------------------------------------------------------------------------------------------------------
    # Convert signal to ∆R2: ∆R2(t) = -1/TE*log(S(t)/S0)
    # ----------------------------------------------------------------------------------------------------------------------
    DeltaR2, tSD = dsc_utils.calculateDeltaR2(dsc_mean_slices_TRfiltered, TE)
    DeltaR2_afterFilters, tSD_afterFilters = dsc_utils.calculateDeltaR2(dsc_mean_slices_smoothed, TE)


    return DeltaR2, DeltaR2_afterFilters

#%%
def processSignal_bySlice(iFname, maskFname, physioLogFname, TE):

    # load data
    img = nib.load(iFname).get_data()  # MRI image
    mask = nib.load(maskFname).get_data()  # masks

    # extract mean in mask
    meanSignal, signalBySlice = dsc_utils.extract_signal_within_roi(img, mask)
    signals = np.vstack((meanSignal, signalBySlice.T))

    # Physio processing
    # -----------------
    # repsAcqTime: ((SC+all slices) x Nacq x (PulseOx, Resp)
    # timePhysio: N_pulseOx_points x ((PulseOx, Resp)
    # valuesPhysio: N_pulseOx_points x ((PulseOx, Resp)
    repsAcqTime, timePhysio, valuesPhysio = dsc_utils.extract_acqtime_and_physio_by_slice(physioLogFname, img.shape[2], img.shape[3])

    # reorder the acquisition times of each slice according to the acquisition scheme ("interleaved", "ascending",
    # "descending") [ONLY IF 3 SLICES ACQUIRED]
    if img.shape[2] == 3:
        acqScheme = "interleaved"
        if acqScheme == "interleaved":
            actualAcqTime_idx = [0, 2, 1]
        elif acqScheme == "ascending":
            actualAcqTime_idx = [0, 1, 2]
        elif acqScheme == "descending":
            actualAcqTime_idx = [2, 1, 0]
        slicesAcqTime = repsAcqTime[1:, :, :]
        repsAcqTime[1:, :, :] = slicesAcqTime[actualAcqTime_idx, :, :]

    # Normalize all signals by 1 - exp(-TReffective/T1)
    # ------------------------------------------------------------------------------------------------------------------
    TReff = np.append(np.diff(repsAcqTime[0, :, 0])[0], np.diff(repsAcqTime[0, :, 0]))
    signals_TRnorm = np.divide(signals, np.tile(1 - np.exp(-TReff / 1251), (signals.shape[0], 1)))

    # Missed triggers
    # ------------------------------------------------------------------------------------------------------------------
    signals_TRfilt, repsAcqTime_PulseOx_TRfilt, repsAcqTime_Resp_TRfilt, idxAcqToDiscard, cardiacPeriod = dsc_utils.discardWrongTRs(TReff, timePhysio[:, 0], valuesPhysio[:, 0], signals_TRnorm, repsAcqTime[:, :, 0], repsAcqTime[:, :, 1], outPlotFname='')
    repsAcqTime_TRfilt = np.stack((repsAcqTime_PulseOx_TRfilt, repsAcqTime_Resp_TRfilt), axis=2)

    # Breathing frequency filtering + final smoothing
    # ----------------------------------------------------------------------------------------------------------------------
    # Regrid physio signals with regular sampling (twice more sampling) except the MRI signal (will be done in the
    # following for loop)
    # ----------------------------------------------------------------------------------------------------------------------
    timePhysioRegrid = np.zeros((2 * timePhysio.shape[0], timePhysio.shape[1]))
    valuesPhysioRegrid = np.zeros((2 * timePhysio.shape[0], timePhysio.shape[1]))
    # PulseOx
    timePhysioRegrid[:, 0] = np.linspace(np.min(timePhysio[:, 0]), np.max(timePhysio[:, 0]), 2 * timePhysio.shape[0])
    valuesPhysioRegrid[:, 0] = np.interp(timePhysioRegrid[:, 0], timePhysio[:, 0], valuesPhysio[:, 0])
    # Respiration
    timePhysioRegrid[:, 1] = timePhysioRegrid[:, 0]
    valuesPhysioRegrid[:, 1] = np.interp(timePhysioRegrid[:, 1], timePhysio[:, 1], valuesPhysio[:, 1])

    # apply last two step to signals individually and then merge them back
    # signal in whole SC
    acqTimeRegrid_wholeSC, signal_breath_wholeSC, signal_smooth_wholeSC, freqResp_cut = filterSignal(signals_TRfilt[0, :], repsAcqTime_TRfilt[0, :, 0], timePhysioRegrid, valuesPhysioRegrid, cardiacPeriod=cardiacPeriod)

    signals_breath = np.zeros((img.shape[2] + 1, signal_breath_wholeSC.size))  # (SC+all slices) x time
    signals_smooth = np.zeros((img.shape[2] + 1, signal_smooth_wholeSC.size))  # (SC+all slices) x time
    acqTimeRegrid = np.zeros((img.shape[2] + 1, acqTimeRegrid_wholeSC.size))  # (SC+all slices) x time
    # store first signal (already processed)
    signals_breath[0, :] = signal_breath_wholeSC
    signals_smooth[0, :] = signal_smooth_wholeSC
    acqTimeRegrid[0, :] = acqTimeRegrid_wholeSC
    # signal in individual slices
    for i_slice in range(img.shape[2]):
        acqTimeRegrid[i_slice + 1, :], signals_breath[i_slice + 1, :], signals_smooth[i_slice + 1, :], freqResp_cut = filterSignal(signals_TRfilt[i_slice + 1, :], repsAcqTime_TRfilt[i_slice + 1, :, 0], timePhysioRegrid, valuesPhysioRegrid, cardiacPeriod=cardiacPeriod)

    return repsAcqTime, signals, repsAcqTime_TRfilt, signals_TRfilt, idxAcqToDiscard, signals_breath, signals_smooth, timePhysioRegrid, valuesPhysioRegrid


#%%
def filterSignal(mriSignal, acqTime, timePhysioRegrid, valuesPhysioRegrid, cardiacPeriod, freqDetection='temporal'):
    """

    Define function to apply last 2 steps (breathing frequencies filtering and final smoothing) in one single call

    :param mriSignal:
    :param acqTime:
    :param timePhysioRegrid:
    :param valuesPhysioRegrid:
    :return:
    """

    # Acquisition times
    acqTimeRegrid = np.arange(start=np.min(acqTime), stop=np.max(acqTime)+cardiacPeriod, step=cardiacPeriod)
    # regrid MRI signal on regular sampling
    mriSignalRegrid = np.interp(acqTimeRegrid, acqTime, mriSignal)

    # # injection rep
    # injRepRegrid = np.abs(acqTimeRegrid - injTime).argmin()
    # print('\n\t>> Injection occurred at t=%.1fms <=> repetition #%i on regridded time axis.\n\n' % (injTime, injRepRegrid))
    # # first pass start rep
    # firstPassStartRepRegrid = np.abs(acqTimeRegrid - firstPassStartTime).argmin()
    # print('\n\t>> First pass starts at %.1f ms, which corresponds to repetition #%i on the resampled time axis.' % (firstPassStartTime, firstPassStartRepRegrid))
    # # first pass last rep
    # firstPassEndRepRegrid = np.abs(acqTimeRegrid - firstPassEndEndTime).argmin()
    # print('\t>> First pass ends at %.1f ms, which corresponds to repetition #%i on the resampled time axis.\n\n' % (firstPassEndEndTime, firstPassEndRepRegrid))

    # # Filter only the baseline to remove high frequencies
    # # ---------------------------------------------------
    # if injectionRep:
    #     mriSignalRegrid[0:firstPassStartRepRegrid] = dsc_utils.filterHighFreq(mriSignalRegrid[0:firstPassStartRepRegrid], acqTimeRegrid[0:firstPassStartRepRegrid], valuesPhysioRegrid[:, 1], timePhysioRegrid[:, 1], outPlotFname='')

    # Filter signal so as to remove frequencies in the range of the respiration frequency
    # -----------------------------------------------------------------------------------
    mriSignal_filtered, freqResp_cut = dsc_utils.filterResp(mriSignalRegrid, acqTimeRegrid, valuesPhysioRegrid[:, 1], timePhysioRegrid[:, 1], outPlotFname='', cardiacPeriod=cardiacPeriod, freqDetection=freqDetection)

    # Smooth signal
    # -------------
    mriSignal_smoothed = dsc_utils.smooth_signal(mriSignal_filtered, outPlotFname='')
    # mriSignal_smoothed = dsc_utils.smooth_signal(mriSignalRegrid, outPlotFname='')

    # # Smoothly crop signal
    # # --------------------
    # mriSignal_crop, _ = dsc_utils.smoothlyCropSignal(mriSignal_smoothed, firstPassStartRepRegrid, firstPassEndRepRegrid, injRepRegrid)

    return acqTimeRegrid, mriSignal_filtered, mriSignal_smoothed, freqResp_cut  # mriSignal_crop

