#!/usr/bin/env python3

"""
Investigate DSC data.


Created on Fri Sep 13 12:44:01 2019

@author: slevy
"""

import dsc_extract_physio
import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import pydicom
from matplotlib import cm
from lmfit.models import GaussianModel
from datetime import datetime
import warnings



def extract_signal_within_roi(image, mask):

    if len(image.shape) > 3:

        nrep = image.shape[3]
        s_along_reps = np.zeros((nrep))
        s_along_reps_by_slice = np.zeros((nrep, image.shape[2]))
        for i_rep in range(nrep):
            img_rep_i = image[:, :, :, i_rep]
            s_along_reps[i_rep] = np.mean(img_rep_i[mask > 0])
            for i_z in range(image.shape[2]):
                s_along_reps_by_slice[i_rep, i_z] = np.mean(img_rep_i[mask[:, :, i_z] > 0, i_z])

        return s_along_reps, s_along_reps_by_slice

    else:

        s_whole_mask = np.mean(image[mask > 0])
        s_by_slice = np.zeros((image.shape[2]))
        for i_z in range(image.shape[2]):
            s_by_slice[i_z] = np.mean(image[mask[:, :, i_z] > 0, i_z])

        return s_whole_mask, s_by_slice


# def detect_outliers(signal, time):
#
#     # thresholds for detection
#     sd_t = np.std(signal[1:])  # first point is always outlier
#     mean_baseline = np.mean(signal[0, 1:12])
#
#
#     # find outliers =================================================================================
#     signal_reptimes = np.vstack((s_along_reps, reps_acqtime))
#     signal_reptimes_outliers = np.zeros((2, 1))
#     signal_reptimes_outliers[:, 0] = signal_reptimes[:, 0]  # save the first point as outlier because it is always corrupted in those data
#     signal_reptimes_without_outliers = signal_reptimes[:, 1:]  # remove the first point which is always corrupted with this sequence
#
#     # if above 3 standard-deviation it is an outlier
#     idx_outliers = np.where(np.abs(signal_reptimes_without_outliers[0, :] - mean_baseline) >= 3*sd_t)  # find indexes of outliers
#     signal_reptimes_outliers = np.hstack((signal_reptimes_outliers, signal_reptimes_without_outliers[:, idx_outliers[0]]))  # save the detected outliers
#     signal_reptimes_without_outliers = np.delete(signal_reptimes_without_outliers, idx_outliers, axis=1)  # remove the outliers
#     # by slice
#     s_along_reps_by_slice = np.delete(s_along_reps_by_slice, 0, axis=0)  # first point is always outlier
#     sd_t_by_slice = np.std(s_along_reps_by_slice, axis=0)  # temporal SD for each slice
#     s_along_reps_by_slice_without_outliers = []  # [[signal, acqtimes], [,], [,] ]
#     for i_z in range(dsc.shape[2]):
#         idx_outliers_z_i = np.where(np.abs(s_along_reps_by_slice[:, i_z] - np.mean(s_along_reps_by_slice[0:11, i_z])) >= 3 * sd_t_by_slice[i_z])  # find indexes of outliers
#         s_along_reps_by_slice_without_outliers.append([np.delete(s_along_reps_by_slice[:, i_z], idx_outliers_z_i), np.delete(signal_reptimes[1, 1:], idx_outliers_z_i)])
#
#     return idx_outliers, signal_without_outliers, signal_outliers, time_without_outliers_time_outliers


def smooth_signal(signal, baseline_nb=10, windowLength=23, outPlotFname=''):
    """
    Smooth signal.
    :param signal: MRI signal, already regridded to a regular sampling
    :param time:
    :param baseline_nb:
    :param increase_res_factor:
    :return:
    """

    # first point is always an outlier (and a NaN actually because of the TReff normalization)
    # --> replace it by the mean signal at baseline
    signal[0] = np.mean(signal[1:baseline_nb])

    # # interpolate signal on regular grid
    # t_regular_sampling = np.linspace(np.min(time), np.max(time), increase_res_factor * len(time))
    # signal_interp = np.interp(t_regular_sampling, time, signal)

    # replace

    # signal_interp_smoothed = scipy.signal.savgol_filter(signal_interp, window_length=25, polyorder=3)
    signal_smoothed = scipy.signal.savgol_filter(signal, window_length=windowLength, polyorder=5, mode='constant', cval=signal[0])

    if outPlotFname:
        # plot results
        fig, ((ax1)) = plt.subplots(1, 1, figsize=(20, 9.5))

        ax1.set_title('Final signal smoothing')
        ax1.set_xlabel('Points')
        ax1.plot(np.arange(signal.size), signal, label='original signal', color='black', lw=0.3, marker='+')
        ax1.plot(np.arange(signal.size), signal_smoothed, label='smoothed signal', color='tab:blue', lw=0.3, marker='o', fillstyle='none')
        ax1.legend()
        ax1.grid()

        fig.savefig(outPlotFname)
        plt.close()

    return signal_smoothed


def smoothlyCropSignal(mriSignalRegrid, firstPassStartRepRegrid, firstPassEndRepRegrid, injRepRegrid, outPlotFname=''):
    """

    :param mriSignalRegrid:
    :param baselineLastRepRegrid:
    :param firstPassEndRepRegrid:
    :param outPlotFname:
    :return: mriSignalCropSmooth: signal cropped before first pass start and after first pass end with smooth transitions
             mriSignalCropEndSmooth_forAIF: signal cropped only after half time of first pass (start time + (end time -
             start time)/2) with smooth transition, to be used for AIF detection
    """

    # calculate the baseline before and after contrast agent first pass
    baselineBefore = np.mean(mriSignalRegrid[0:firstPassStartRepRegrid])
    baselineAfter = np.mean(mriSignalRegrid[firstPassEndRepRegrid:-1])

    # replace them in original signal
    mriSignalCrop = np.copy(mriSignalRegrid)
    mriSignalCrop[0:firstPassStartRepRegrid] = baselineBefore
    mriSignalCrop[firstPassEndRepRegrid:-1] = baselineAfter

    # crop larger for AIF detection
    mriSignalCropEnd_forAIF = np.copy(mriSignalRegrid)
    firstPassMiddleRep = int(np.ceil(firstPassStartRepRegrid + (firstPassEndRepRegrid - firstPassStartRepRegrid)/2))
    mriSignalCropEnd_forAIF[0:injRepRegrid] = baselineBefore
    mriSignalCropEnd_forAIF[firstPassMiddleRep:-1] = baselineAfter

    # smooth whole signal to avoid sharp transitions
    mriSignalCropSmooth = scipy.signal.savgol_filter(mriSignalCrop, window_length=25, polyorder=3, mode='nearest')
    mriSignalCropEndSmooth_forAIF = scipy.signal.savgol_filter(mriSignalCropEnd_forAIF, window_length=25, polyorder=3, mode='nearest')

    if outPlotFname:
        # plot results
        fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(20, 9.5))

        ax1.set_title('Final smooth & crop of signal')
        ax1.set_xlabel('Points')
        ax1.plot(np.arange(mriSignalRegrid.size), mriSignalRegrid, label='original signal', color='black', lw=0.7, marker='+')
        ax1.plot(np.arange(mriSignalRegrid.size), mriSignalCrop, label='cropped signal', color='tab:blue', lw=0.7, marker='.')
        ax1.plot(np.arange(mriSignalRegrid.size), mriSignalCropSmooth, label='smoothly cropped signal', color='tab:red', lw=0.7, marker='.')
        ax1.axvline(x=firstPassStartRepRegrid, label='first pass start', color='green', lw=1)
        ax1.axvline(x=firstPassEndRepRegrid, label='first pass end', color='red', lw=1)
        ax1.legend()
        ax1.grid()

        ax2.set_title('Final smooth & crop of signal for AIF detection')
        ax2.set_xlabel('Points')
        ax2.plot(np.arange(mriSignalRegrid.size), mriSignalRegrid, label='original signal', color='black', lw=0.7, marker='+')
        ax2.plot(np.arange(mriSignalRegrid.size), mriSignalCropEnd_forAIF, label='cropped signal', color='tab:blue', lw=0.7, marker='.')
        ax2.plot(np.arange(mriSignalRegrid.size), mriSignalCropEndSmooth_forAIF, label='smoothly cropped signal', color='tab:red', lw=0.7, marker='.')
        ax2.axvline(x=firstPassStartRepRegrid, label='first pass start', color='green', lw=1)
        ax2.axvline(x=firstPassEndRepRegrid, label='first pass end', color='red', lw=1)
        ax2.axvline(x=firstPassMiddleRep, label='first pass middle', color='orange', lw=1)
        ax2.legend()
        ax2.grid()

        fig.savefig(outPlotFname)
        plt.close()

    return mriSignalCropSmooth, mriSignalCropEndSmooth_forAIF



def plot_signal_vs_TReff(effTR, signal, time, ofname=''):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 9.7))

    ax1.set_xlabel('Effective TR (ms)')
    ax1.set_ylabel('Signal')
    ax1.set_title("Signal vs effective TR: (Pearson\'s R, p-value)={}".format(tuple(np.round(scipy.stats.pearsonr(effTR[1:], signal[1:]), decimals=4))))
    ax1.grid(which='both')
    ax1.plot(effTR, signal, linewidth=0, marker='+', markersize=5.0)

    ax2.set_xlabel('$1 - e^{-TR_{eff}/T_{1}} (TR_{eff}\\ in\\ ms)$')
    ax2.set_ylabel('Signal')
    pearsonr, pval = scipy.stats.pearsonr(1 - np.exp(-effTR[1:]/1251.0), signal[1:])
    ax2.set_title("Signal vs $1 - e^{-TR_{eff}/T_{1}}$: (Pearson\'s R, p-value)=(%.4f, %.4f)" % (pearsonr, pval))
    ax2.grid(which='both')
    ax2.plot(1 - np.exp(-effTR/1251.0), signal, linewidth=0, marker='+', markersize=5.0)

    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Signal')
    ax3.set_title("Signal and effective TR vs time")
    ax3.grid(which='both')
    ax3.plot(time/1000, signal, linewidth=1.0, marker='+', markersize=7.0)
    ax3_effTR = ax3.twinx()
    ax3_effTR.plot(time/1000, effTR, linewidth=1.0, marker='+', markersize=7.0, color='orange')
    ax3_effTR.tick_params(axis='y', labelcolor='orange')
    ax3_exp_effTR = ax3.twinx()
    ax3_exp_effTR.plot(time/1000, 1 - np.exp(-effTR/1251.0), linewidth=1.0, marker='+', markersize=7.0, color='green')
    ax3_exp_effTR.tick_params(axis='y', labelcolor='green')

    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Signal', color='green')
    ax4.set_title("Signal vs time")
    ax4.grid(which='both')
    signal_norm_exp = np.divide(signal, (1 - np.exp(-effTR/1251.0)))
    ax4.plot(time/1000, signal_norm_exp, linewidth=1, marker='+', markersize=7.0, color='green', label='norm by $1 - e^{-TR_{eff}/T_{1}}$: COV='+str(round(100*np.std(signal_norm_exp[1:])/np.mean(signal_norm_exp[1:]), 2))+'%')
    ax4.tick_params(axis='y', color='green')
    ax4.legend(loc="lower left")

    ax4_effTR = ax4.twinx()
    signal_norm_TR = np.divide(signal, effTR)
    ax4_effTR.plot(time/1000, signal_norm_TR, linewidth=1, marker='+', markersize=7.0, color='orange', label='norm by $TR_{eff}$: COV='+str(round(100*np.std(signal_norm_TR[1:])/np.mean(signal_norm_TR[1:]), 2))+'%')
    ax4_effTR.set_ylabel('Signal', color='orange')
    ax4_effTR.tick_params(axis='y', color='orange')
    ax4_effTR.legend(loc="lower right")

    ax4_rawsignal = ax4.twinx()
    ax4_rawsignal.plot(time/1000, signal, linewidth=1, marker='+', markersize=7.0, color='blue', label='raw signal: COV='+str(round(100*np.std(signal[1:])/np.mean(signal[1:]), 2))+'%')
    ax4_rawsignal.set_ylabel('Signal', color='blue')
    ax4_rawsignal.tick_params(axis='y', color='blue')
    ax4_rawsignal.legend(loc="upper right")

    plt.show(block=True)
    if ofname:
        fig.savefig(ofname+'_signal_vs_TReff.png')


def plot_signal_vs_time(time, signal, time_interp, signal_interp, signal_smoothed, signal_by_slice_smoothed, ofname='', baseline_nb=50):

    fig, (axtime, axtime_norm) = plt.subplots(1, 2, figsize=(20, 9.5))
    plt.subplots_adjust(wspace=0.2, left=0.05, right=0.95)

    axtime.set_xlabel('Time (s)')
    axtime.set_ylabel('Signal')
    axtime.grid(which='both')
    axtime.plot(time/1000., signal, marker='+', linewidth=0.2, color='black', label='raw signal')
    # baseline and injection
    axtime.axvline(x=time[baseline_nb+1]/1000, linestyle='-', label='injection', color='red', linewidth=1.0)
    axtime.axhline(y=np.mean(signal[0:baseline_nb]), linestyle='-', label='baseline', color='gray', alpha=0.7, linewidth=3.0)
    axtime.legend()

    # display repetition numbers on top x-axis
    time_ticks_locs = axtime.get_xticks()
    reps_nb_interp_to_time_ticks_locs = np.interp(time_ticks_locs, time / 1000, range(len(time)))
    axreps = axtime.twiny()
    axreps.plot(time/1000., signal, marker='', linewidth=0)  # fake plot to get the same x-axis
    axreps.set_xticklabels(np.round(reps_nb_interp_to_time_ticks_locs, decimals=0).astype(int))
    axreps.set_xlabel('Repetition number')

    # add effective TR
    axTR = axtime.twinx()
    axTR.plot(time / 1000, np.append(np.nan, np.diff(time)), color='orange', linewidth=0.8, label='effective TR')
    axTR.set_ylabel('Effective TR (ms)', color='orange')
    axTR.tick_params(axis='y', labelcolor='orange')

    # plot normalized signal
    axtime_norm.set_xlabel('Time (s)')
    axtime_norm.set_ylabel('Signal')
    axtime_norm.grid(which='both')
    axtime_norm.plot(time_interp/1000., signal_interp, linestyle='-', color='gray', alpha=0.7, label='S$_{inter}$')
    axtime_norm.plot(time_interp/1000., signal_smoothed, linestyle='-', color='green', label='S$_{smoothed}$')
    # plt.plot(signal_reptimes_outliers[1, :]/1000., signal_reptimes_outliers[0, :], marker='+', linewidth=0., color='red', label='outliers')
    # plot by slice
    colors = [cm.jet(slc) for slc in np.linspace(0.0, 1.0, signal_by_slice_smoothed.shape[1])]
    for iz, color in enumerate(colors):
        axtime_norm.plot(time/1000., signal_by_slice_smoothed[:, iz], label="z="+str(iz), color=color, lw=1, marker='.', ms=2.)
    axtime_norm.axvline(x=time[baseline_nb+1]/1000, linestyle='-', label='injection', color='red', linewidth=1.0)
    axtime_norm.legend()

    # # plot physio on the same graph but different scale
    # ax_physio = plt.gca().twinx()
    # ax_physio.plot(time/1000., physio_values, marker=None, color='cyan', label='pulseOx')

    if ofname:
        fig.savefig(ofname+'_signal_vs_time.png')


def extract_acqtime_and_physio(log_fname, nrep_nii, physioplot_out_fname=''):

    # pulseOx ----------------------------
    if os.path.exists(log_fname+'.puls'):

        time_puls, puls_values, epi_acqtime_puls, epi_event_puls, acq_window_puls = dsc_extract_physio.read_physiolog(log_fname+'.puls', sampling_period=20)  # extract physio signal
        reps_table_puls, slice_table_puls = dsc_extract_physio.sort_event_times(epi_acqtime_puls, epi_event_puls)  # sort event times
        if physioplot_out_fname:
            dsc_extract_physio.plot_physio(time_puls, puls_values, epi_acqtime_puls, reps_table_puls, acq_window_puls, physioplot_out_fname+'_pulseOx')  # plot physio signal
        # calculate acquisition time for each rep
        nrep_pulseOxLog = np.sum(reps_table_puls[:, 1])
        if nrep_nii != nrep_pulseOxLog:
            os.error('Number of repetitions in image is different from the number of repetitions recorded in pulseOx physiolog.')
        reps_acqtime_pulseOx = np.squeeze(np.mean(slice_table_puls[np.where(reps_table_puls[:, 1] == 1), :], axis=2))
    else:
        reps_acqtime_pulseOx = 900*np.arange(0, nrep_nii)
        time_puls = np.linspace(np.min(reps_acqtime_pulseOx), np.max(reps_acqtime_pulseOx), int((np.max(reps_acqtime_pulseOx) - np.min(reps_acqtime_pulseOx))/20000))
        puls_values = None

    # respiration ----------------------------
    if os.path.exists(log_fname+'.resp'):

        time_resp, resp_values, epi_acqtime_resp, epi_event_resp, acq_window_resp = dsc_extract_physio.read_physiolog(log_fname+'.resp', sampling_period=20)  # extract physio signal
        reps_table_resp, slice_table_resp = dsc_extract_physio.sort_event_times(epi_acqtime_resp, epi_event_resp)  # sort event times
        if physioplot_out_fname:
            dsc_extract_physio.plot_physio(time_resp, resp_values, epi_acqtime_resp, reps_table_resp, acq_window_resp, physioplot_out_fname+'_resp')  # plot physio signal
        # calculate acquisition time for each rep
        nrep_respLog = np.sum(reps_table_resp[:, 1])
        if nrep_nii != nrep_respLog:
            os.error('Number of repetitions in image is different from the number of repetitions recorded in respiration physiolog.')
        reps_acqtime_resp = np.squeeze(np.mean(slice_table_resp[np.where(reps_table_resp[:, 1] == 1), :], axis=2))
    else:
        reps_acqtime_resp = 900*np.arange(0, nrep_nii)
        time_resp = np.linspace(np.min(reps_acqtime_resp), np.max(reps_acqtime_resp), int((np.max(reps_acqtime_resp) - np.min(reps_acqtime_resp))/20000))
        resp_values = None

    return reps_acqtime_pulseOx, time_puls, puls_values, reps_acqtime_resp, time_resp, resp_values


def extract_acqtime_and_physio_by_slice(log_fname, nSlices, nAcqs, acqTime_firstImg, TR=1000):
    """

    :param log_fname:
    :param nSlices:
    :param nAcqs:
    :return:    repsAcqTime: ((SC+all slices) x Nacq x (PulseOx, Resp)
                timePhysio: N_pulseOx_points x ((PulseOx, Resp)
                valuesPhysio: N_pulseOx_points x ((PulseOx, Resp)
    """

    # repsAcqTime: ((SC+all slices) x Nacq x (PulseOx, Resp)
    # timePhysio: N_pulseOx_points x ((PulseOx, Resp)
    # valuesPhysio: N_pulseOx_points x ((PulseOx, Resp)
    repsAcqTime = np.zeros((1+nSlices, nAcqs, 2))

    # pulseOx ----------------------------
    if os.path.exists(log_fname+'.puls'):
        print('Processing pulseOx log: '+log_fname+'.puls')

        if 'slr' in os.path.basename(log_fname):
            print('\t[\'slr\'-type physiolog]')
            time_puls, puls_values, epi_acqtime_puls, epi_event_puls, acq_window_puls = dsc_extract_physio.read_physiolog(log_fname+'.puls', sampling_period=20)  # extract physio signal
            reps_table_puls, slices_table_puls = dsc_extract_physio.sort_event_times(epi_acqtime_puls, epi_event_puls)  # sort event times

            nrep_pulseOxLog = np.sum(reps_table_puls[:, 1])
            if nAcqs != nrep_pulseOxLog:
                os.error('Number of repetitions in image is different from the number of repetitions recorded in pulseOx physiolog.')
            # get acquisition time for each slice
            repsAcqTime[1:, :, 0] = np.squeeze(slices_table_puls[np.where(reps_table_puls[:, 1] == 1), :]).T
        else:
            print('\t[\'CMRR\'-type physiolog]')
            time_puls, trigger_start_times_puls, trigger_end_times_puls, puls_values, acq_window_puls, acqStartTime_puls = dsc_extract_physio.read_physiolog_cmrr(log_fname+'.puls')
            triggerStartTimes_imgOnly_puls = dsc_extract_physio.extract_acqTimes_cmrr(trigger_start_times_puls, acqTime_firstImg, acqStartTime_puls, trigger_end_times_puls)
            repsAcqTime[1:, :, 0] = np.tile(triggerStartTimes_imgOnly_puls, (nSlices, 1)) + np.tile(TR/nSlices * np.arange(0, nSlices), (nAcqs, 1)).T

    else:
        print('\nNo log found for pulseOx.')
        repsAcqTime[1:, :, 0] = TR*np.tile(np.arange(0, nAcqs), (nSlices, 1)) + np.tile(TR/nSlices*np.arange(0, nSlices), (nAcqs, 1)).T
        time_puls = np.arange(np.min(repsAcqTime), np.max(repsAcqTime), step=20)
        puls_values = None

    # take the mean acquisition time across slices for the whole rep (SC)
    repsAcqTime[0, :, 0] = np.mean(repsAcqTime[1:nSlices, :, 0], axis=0)

    # respiration ----------------------------
    if os.path.exists(log_fname+'.resp'):
        print('Processing respiration log: '+log_fname+'.resp')

        if 'slr' in os.path.basename(log_fname):
            print('\t[\'slr\'-type physiolog]')
            time_resp, resp_values, epi_acqtime_resp, epi_event_resp, acq_window_resp = dsc_extract_physio.read_physiolog(log_fname+'.resp', sampling_period=20)  # extract physio signal
            reps_table_resp, slices_table_resp = dsc_extract_physio.sort_event_times(epi_acqtime_resp, epi_event_resp)  # sort event times

            nrep_respLog = np.sum(reps_table_resp[:, 1])
            if nAcqs != nrep_respLog:
                os.error('Number of repetitions in image is different from the number of repetitions recorded in respiration physiolog.')
            # get acquisition time for each slice
            repsAcqTime[1:, :, 1] = np.squeeze(slices_table_resp[np.where(reps_table_resp[:, 1] == 1), :]).T
        else:
            print('\t[\'CMRR\'-type physiolog]')
            time_resp, trigger_start_times_resp, trigger_end_times_resp, resp_values, acq_window_resp, acqStartTime_resp = dsc_extract_physio.read_physiolog_cmrr(log_fname+'.resp')

    else:
        print('\nNo log found for respiration.\n')
        repsAcqTime[1:, :, 1] = TR*np.tile(np.arange(0, nAcqs), (nSlices, 1)) + np.tile(TR/nSlices*np.arange(0, nSlices), (nAcqs, 1)).T
        time_resp = np.arange(np.min(repsAcqTime), np.max(repsAcqTime), step=20)
        resp_values = None

    # take the mean acquisition time across slices for the whole rep (SC)
    repsAcqTime[0, :, 1] = np.mean(repsAcqTime[1:nSlices, :, 1], axis=0)

    # merge the two physiological signal into one array each (for time and physio values)
    if time_puls.size > time_resp.size:
        time_resp = np.hstack((time_resp, time_puls[time_resp.size:]))
        resp_values = np.pad(resp_values, (0, puls_values.size - resp_values.size), 'reflect')
    elif time_puls.size < time_resp.size:
        time_puls = np.hstack((time_puls, time_resp[time_puls.size:]))
        puls_values = np.pad(puls_values, (0, resp_values.size - puls_values.size), 'reflect')

    timePhysio = np.vstack((time_puls, time_resp)).T
    valuesPhysio = np.vstack((puls_values, resp_values)).T

    return repsAcqTime, timePhysio, valuesPhysio


def plot_pulseOx_and_resp(pulseTime, pulseVal, pulseAcqTimes, respTime, respVal, respAcqTime, ofname=''):

    fig, ((ax1)) = plt.subplots(1, 1, figsize=(20, 9.5))

    ax1.plot(pulseTime, pulseVal, color='red', label='PulseOx signal')
    ax1.plot(respTime, respVal, color='blue', label='Respiration signal')
    for acqtime in pulseAcqTimes:
        ax1.axvline(x=acqtime, ymin=0, ymax=.5, color='red', lw=0.8, label='reps' if np.where(pulseAcqTimes==acqtime)[0][0] == 0 else "_nolegend_")
    for acqtime in respAcqTime:
        ax1.axvline(x=acqtime, ymin=.5, ymax=1, color='blue', lw=0.8, label='reps' if np.where(respAcqTime==acqtime)[0][0] == 0 else "_nolegend_")

    ax1.legend()
    ax1.grid()

    fig.show()
    if ofname:
        ax1.set_title('Saved to: ' + ofname + '.png')
        fig.savefig(ofname+'.png')
    plt.close()


def plot_signal_vs_resp(respTime, respSignal, mriTime, mriSignal, ofname=''):

    # interpolate respiration signal to MRI signal sampling
    respSignalSampledToMRISignal = np.interp(mriTime, respTime, respSignal)
    # remove points where respiration signal is saturated
    mriSignal_noRespSat = np.delete(mriSignal, np.where((respSignalSampledToMRISignal == 0) | (respSignalSampledToMRISignal == 4095)))
    respSignal_noRespSat = np.delete(respSignalSampledToMRISignal, np.where((respSignalSampledToMRISignal == 0) | (respSignalSampledToMRISignal == 4095)))
    mriTime_noRespSat = np.delete(mriTime, np.where((respSignalSampledToMRISignal == 0) | (respSignalSampledToMRISignal == 4095)))

    # interpolate MRI signal to respiration signal sampling
    mriSignalSampledToRespSignal = np.interp(respTime, mriTime, mriSignal)
    mriSignalSampledToRespSignal = mriSignalSampledToRespSignal[np.abs(respTime - np.min(mriTime)).argmin():np.abs(respTime - np.max(mriTime)).argmin()]
    respTimeCropToMRI = respTime[np.abs(respTime - np.min(mriTime)).argmin():np.abs(respTime - np.max(mriTime)).argmin()]
    respSignalCropToMRI = respSignal[np.abs(respTime - np.min(mriTime)).argmin():np.abs(respTime - np.max(mriTime)).argmin()]

    # remove points where respiration signal is saturated
    mriSignalOverSampled_noRespSat = np.delete(mriSignalSampledToRespSignal, np.where((respSignalCropToMRI == 0) | (respSignalCropToMRI == 4095)))
    respSignalCropToMRI_noRespSat = np.delete(respSignalCropToMRI, np.where((respSignalCropToMRI == 0) | (respSignalCropToMRI == 4095)))
    respTimeCropToMRI_noRespSat = np.delete(respTimeCropToMRI, np.where((respSignalCropToMRI == 0) | (respSignalCropToMRI == 4095)))


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 9.7))
    plt.subplots_adjust(wspace=0.3, left=0.05, right=0.95, hspace=0.3, bottom=0.05, top=0.95)

    ax1.set_title("Signal vs time")
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Signal', color='green')
    ax1.grid(which='both')
    ax1.plot(mriTime/1000, mriSignal, linewidth=1, marker='+', markersize=7.0, color='green', label='$S_{MRI}$: COV='+str(round(100*np.std(mriSignal)/np.mean(mriSignal), 2))+'%')
    ax1.plot(respTimeCropToMRI/1000, mriSignalSampledToRespSignal, linewidth=1.2, marker=None, color='gray', label='$S_{MRI} interp$: COV='+str(round(100*np.std(mriSignalSampledToRespSignal)/np.mean(mriSignalSampledToRespSignal), 2))+'%')
    ax1.tick_params(axis='y', labelcolor='green')
    ax1.legend(loc="lower left")

    ax1_resp = ax1.twinx()
    ax1_resp.set_ylabel('Signal')
    ax1_resp.grid(which='both')
    ax1_resp.plot(respTime/1000, respSignal, linewidth=1, marker=None, color='blue', label='$S_{resp}$: COV=' + str(round(100 * np.std(respSignal) / np.mean(respSignal), 2)) + '%')
    ax1_resp.plot(mriTime/1000, respSignalSampledToMRISignal, linewidth=0, marker='+', color='red', label='$S_{resp}$: COV=' + str(round(100 * np.std(respSignalSampledToMRISignal) / np.mean(respSignalSampledToMRISignal), 2)) + '%')
    ax1_resp.plot(mriTime_noRespSat/1000, respSignal_noRespSat, linewidth=0, marker='+', color='blue', label='$S_{resp}$ no sat')
    ax1_resp.tick_params(axis='y', labelcolor='blue')
    ax1_resp.legend(loc="lower right")

    ax2.set_title("MRI signal vs Respiration signal: (Pearson\'s R, p-value)={}".format(tuple(np.round(scipy.stats.pearsonr(mriSignalOverSampled_noRespSat, respSignalCropToMRI_noRespSat), decimals=4))))
    ax2.set_xlabel('Respiration signal')
    ax2.set_ylabel('MRI signal (interpolated to respiration sampling)')
    ax2.grid(which='both')
    # ax2.plot(respSignalSampledToMRISignal, mriSignal, linewidth=0, marker='+', markersize=7.0, color='tab:red', label='all points')
    # ax2.plot(respSignal_noRespSat, mriSignal_noRespSat, linewidth=0, marker='+', markersize=7.0, color='tab:blue', label='without respiration signal saturation')
    # ax2.plot(respSignalCropToMRI, mriSignalSampledToRespSignal, linewidth=0, marker='+', markersize=7.0, color='tab:orange', label='all points')
    ax2.plot(respSignalCropToMRI_noRespSat, mriSignalOverSampled_noRespSat, linewidth=0, marker='+', markersize=7.0, color='tab:green', label='without respiration signal saturation')
    ax2.legend()

    ax3.set_title("Signal vs time interpolated to respiration sampling")  # --------------------------------------------
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Signal', color='green')
    ax3.grid(which='both')
    ax3.plot(respTimeCropToMRI/1000, mriSignalSampledToRespSignal, linewidth=0, marker='.', markersize=3.0, color='tab:red', label='$S_{MRI} interp to resp$')
    ax3.plot(respTimeCropToMRI_noRespSat/1000, mriSignalOverSampled_noRespSat, linewidth=0, marker='.', markersize=3.0, color='green', label='$S_{MRI} interp to resp NO RESP SAT$')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.legend(loc="lower left")

    ax3_resp = ax3.twinx()
    ax3_resp.set_ylabel('Signal')
    ax3_resp.plot(respTimeCropToMRI/1000, respSignalCropToMRI, linewidth=0, marker='.', markersize=3.0, color='tab:red', label='$S_{resp}$ crop')
    ax3_resp.plot(respTimeCropToMRI_noRespSat/1000, respSignalCropToMRI_noRespSat, linewidth=0, marker='.', markersize=3.0, color='blue', label='$S_{resp}$ NO RESP SAT')
    ax3_resp.tick_params(axis='y', labelcolor='blue')
    ax3_resp.legend(loc="lower right")

    ax3_respPeriod = ax3.twinx()
    respSignalMax, respSignalMin = peakdet(respSignalCropToMRI, 300)
    respPeriod = np.append(np.nan, np.diff(respTimeCropToMRI[respSignalMax[:, 0]]))/1000
    ax3_respPeriod.plot(respTimeCropToMRI[respSignalMax[:, 0]]/1000, respPeriod, linewidth=3.0, marker='+', markersize=10, color='tab:pink', label='Resp period')
    ax3_respPeriod.tick_params(axis='y', labelcolor='tab:pink')
    ax3_respPeriod.set_ylabel('Resp period is s (mean = '+str(round(np.mean(respPeriod[1:]), 2))+' ['+str(np.min(respPeriod[1:]))+', '+str(np.max(respPeriod[1:]))+']', color='tab:pink')
    for tPeak in respTimeCropToMRI[respSignalMax[:, 0]]/1000:
        ax3_respPeriod.axvline(x=tPeak, linestyle='-', color='tab:pink', linewidth=1.0)

    ax3_corr = ax3.twinx()
    ax3_corr.plot(respTimeCropToMRI_noRespSat/1000, scipy.signal.correlate(mriSignalOverSampled_noRespSat, respSignalCropToMRI_noRespSat, mode='same', method='direct'), linewidth=1, marker=None, markersize=0, color='tab:orange', label='Cross-corr')
    ax3_corr.legend(loc="upper right")

    ax4.set_title("FFT")  # --------------------------------------------------------------------------------------------

    # respSignal_FFT = np.fft.fft((respSignalCropToMRI - np.mean(respSignalCropToMRI))/np.std(respSignalCropToMRI))
    # mriSignal_FFT = np.fft.fft((mriSignalSampledToRespSignal - np.mean(mriSignalSampledToRespSignal))/np.std(mriSignalSampledToRespSignal))
    # freq = np.fft.fftfreq(respTimeCropToMRI.size, d=respTimeCropToMRI[1]-respTimeCropToMRI[0])  # in MHz
    # idx_f0 = np.where(freq == 0)[0]
    # idx_ascending_freq = np.argsort(freq)
    freqResMRI, respSignalResMRI_FFT = fft_warpper(mriTime, respSignalSampledToMRISignal, increase_res_factor=5)
    freqResMRI, mriSignalResMRI_FFT = fft_warpper(mriTime, mriSignal, increase_res_factor=5)

    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Signal')
    ax4.grid(which='both')
    # ax4.plot(freq[idx_ascending_freq]*1000, np.abs(respSignal_FFT[idx_ascending_freq]), linewidth=0.9, marker='.', markersize=0, color='black', label='$S_{resp}$')
    # ax4.plot(freq[idx_ascending_freq]*1000, np.abs(mriSignal_FFT[idx_ascending_freq]), linewidth=0.9, marker='.', markersize=0, color='green', label='$S_{MRI}\ interp\ to\ resp$')
    ax4.plot(freqResMRI*1000, respSignalResMRI_FFT, label='$S_{resp}\ res\ MRI$', linewidth=0.9, marker='+', markersize=0, color='black')
    ax4.plot(freqResMRI*1000, mriSignalResMRI_FFT, label='$S_{MRI}\ res\ MRI$', linewidth=0.9, marker='+', markersize=0, color='green')

    ax4.axvspan(xmin=1/np.min(respPeriod[1:]), xmax=1/np.max(respPeriod[1:]), label='respiration frequency range', color='tab:pink', alpha=0.2)
    ax4.legend(loc="upper right")
    ax4.set_xlim(left=0, right=1.5)

    ax4_corr = ax4.twinx()
    ax4_corr.plot(freqResMRI*1000, scipy.signal.correlate(respSignalResMRI_FFT, mriSignalResMRI_FFT, mode='same', method='direct'), label='Cross-corr', linewidth=1, marker=None, markersize=0, color='tab:orange')
    ax4_corr.legend(loc="lower right")

    plt.show(block=True)
    if ofname:
        fig.suptitle('Saved to: '+ofname+'_signal_vs_resp.png')
        fig.savefig(ofname+'_signal_vs_resp.png')

def calculateB1Factor(timePulseOx, signalPulseOx, measuredFAB1map, b1mapVoltage, DSCvoltage, selectedFlipAngle, T1=1251):

    # calculate subject's cardiac cycle
    pulseOxSignalMax, pulseOxSignalMin = peakdet(signalPulseOx, 600)
    cardiacPeriods = np.diff(timePulseOx[pulseOxSignalMax[:, 0].astype(int)])  # in milliseconds
    cardiacPeriodMean = np.mean(cardiacPeriods)

    # calculate required excitation flip angle
    excFArequired = 180 - np.arccos(np.exp(-cardiacPeriodMean/T1))*180/np.pi  # scalar

    # actual B1
    B1actual = (measuredFAB1map * DSCvoltage/b1mapVoltage)/45.0  # matrix

    # actual excitation flip angle
    excFAactual = selectedFlipAngle * B1actual  # matrix

    # actual refocusing flip angle
    refocFAactual = 180 * B1actual  # matrix

    # final factor of used signal
    usedSignalFactor = (excFAactual/excFArequired) * (refocFAactual/180)  # matrix

    return usedSignalFactor


def discardWrongTRs(TReff, timePulseOx, signalPulseOx, mriSignal, repsAcqTime_PulseOx, repsAcqTime_Resp, outPlotFname=''):
    """
    Detect points where sequence missed a cardiac window, loosing steady state.
    Normal cardiac beat varies between 700 and 1400 ms, anything above 1400 ms is probably due to a missed trigger.
    :param TReff:
    :param mriSignal:
    :return:
    """

    # calculate subject's cardiac cycle
    pulseOxSignalMax, pulseOxSignalMin = peakdet(signalPulseOx, 599, outPlotFname=outPlotFname)
    cardiacPeriods = np.diff(timePulseOx[pulseOxSignalMax[:, 0].astype(int)])  # in milliseconds
    # find a threshold to detect misssed triggers
    cardiacPeriodMean_withOutliers = np.mean(cardiacPeriods)
    cardiacPeriodStd_withOutliers = np.std(cardiacPeriods)
    print('\nMean +/- SD cardiac cycle WITH outliers (ms) = %d +/- %d' % (cardiacPeriodMean_withOutliers, cardiacPeriodStd_withOutliers))
    cardiacPeriods_withoutOutliers = cardiacPeriods[(cardiacPeriods < cardiacPeriodMean_withOutliers + 2*cardiacPeriodStd_withOutliers) & (cardiacPeriods > cardiacPeriodMean_withOutliers - 2*cardiacPeriodStd_withOutliers)]
    cardiacPeriodMean_withoutOutliers = np.mean(cardiacPeriods_withoutOutliers)
    cardiacPeriodStd_withoutOutliers = np.std(cardiacPeriods_withoutOutliers)
    print('Mean +/- SD cardiac cycle WITHOUT outliers (ms) = %d +/- %d' % (cardiacPeriodMean_withoutOutliers, cardiacPeriodStd_withoutOutliers))

    # discard acquisitions with effective TR outside mean cardiac cylce +/- 3 true SD (without outliers)
    idxAcqWithBadTR = np.argwhere((TReff >= cardiacPeriodMean_withoutOutliers+4*cardiacPeriodStd_withoutOutliers) | (TReff <= cardiacPeriodMean_withoutOutliers-4*cardiacPeriodStd_withoutOutliers))
    # also discard the repetition following a missed trigger AND the first two repetitions of the set
    idxAcqToDiscard = np.concatenate((np.array([[0], [1]]), idxAcqWithBadTR, idxAcqWithBadTR[idxAcqWithBadTR[:,0] < (TReff.size-1), :]+1))
    idxAcqToDiscard = np.unique(idxAcqToDiscard)
    # discard data
    mriSignal_TRfiltered = np.delete(mriSignal, idxAcqToDiscard, axis=-1)
    repsAcqTime_PulseOx_TRfiltered = np.delete(repsAcqTime_PulseOx, idxAcqToDiscard, axis=-1)
    repsAcqTime_Resp_TRfiltered = np.delete(repsAcqTime_Resp, idxAcqToDiscard, axis=-1)
    print('\nDiscarded '+str(len(idxAcqToDiscard))+' points due to inconsistent effective TR.')

    # plot filtering results if asked
    if outPlotFname and len(mriSignal.shape) == 1:

        fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(20, 9.7))
        plt.subplots_adjust(left=0.05, right=0.95, hspace=0.25, bottom=0.05, top=0.9)

        ax2.set_title("Effective TR")
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Effective TR (ms)')
        ax2.grid(which='both')
        ax2.plot(repsAcqTime_PulseOx/1000, TReff, linewidth=0, marker='+', markersize=7.0, color='red', label='Discarded points')
        ax2.plot(repsAcqTime_PulseOx_TRfiltered/1000, np.delete(TReff, idxAcqToDiscard, axis=-1), linewidth=0, marker='+', markersize=7.0, color='black', label='Kept points')
        ax2.axhline(y=cardiacPeriodMean_withoutOutliers+4*cardiacPeriodStd_withoutOutliers, linewidth=3, color='red', label='Threshold (mean RR = '+str(round(cardiacPeriodMean_withoutOutliers,2))+'+/-'+str(round(cardiacPeriodStd_withoutOutliers,2))+' ms)')
        ax2.axhline(y=cardiacPeriodMean_withoutOutliers-4*cardiacPeriodStd_withoutOutliers, linewidth=3, color='red')
        ax2.legend(loc="lower left")

        ax1.set_title("Effective TR filtering")
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal')
        ax1.grid(which='both')
        ax1.plot(repsAcqTime_PulseOx/1000, mriSignal, linewidth=0, marker='+', markersize=7.0, color='red', label='Discarded points')
        ax1.plot(repsAcqTime_PulseOx_TRfiltered/1000, mriSignal_TRfiltered, linewidth=0, marker='+', markersize=7.0, color='black', label='Kept points')
        ax1.legend(loc="lower left")

        fig.suptitle('Effective TR filtering\nSaved to: '+outPlotFname)
        fig.savefig(outPlotFname)
        plt.show()

    return mriSignal_TRfiltered, repsAcqTime_PulseOx_TRfiltered, repsAcqTime_Resp_TRfiltered, idxAcqToDiscard, cardiacPeriodMean_withoutOutliers


# def deduce_wrongTR_3Tdata(TReff, timePulseOx, signalPulseOx, mriSignal, repsAcqTime_PulseOx, repsAcqTime_Resp, outPlotFname=''):
#     """
#     Detect points where sequence missed a cardiac window, loosing steady state.
#     Normal cardiac beat varies between 700 and 1400 ms, anything above 1400 ms is probably due to a missed trigger.
#     :param TReff:
#     :param mriSignal:
#     :return:
#     """
#
#
#     # sliding window
#
#     # calculate subject's cardiac cycle
#     pulseOxSignalMax, pulseOxSignalMin = peakdet(signalPulseOx, 599, outPlotFname=outPlotFname)
#     cardiacPeriods = np.diff(timePulseOx[pulseOxSignalMax[:, 0].astype(int)])  # in milliseconds
#     cardiacPeriodMean = np.mean(cardiacPeriods)
#     cardiacPeriodMin = np.min(cardiacPeriods)
#     print('\nMean +/- SD cardiac cycle (ms) = %d +/- %d' % (cardiacPeriodMean, np.std(cardiacPeriods)))
#
#     # discard acquisitions with effective TR >= 1.8 times the minimum cardiac cycle of the subject
#     idxAcqWithBadTR = np.argwhere((TReff >= 1.5*cardiacPeriodMean) | (TReff <= 0.5*cardiacPeriodMean))
#     # also discard the repetition following a missed trigger AND the first two repetitions of the set
#     idxAcqToDiscard = np.concatenate((np.array([[0], [1]]), idxAcqWithBadTR, idxAcqWithBadTR+1))
#     # discard data
#     mriSignal_TRfiltered = np.delete(mriSignal, idxAcqToDiscard, axis=-1)
#     repsAcqTime_PulseOx_TRfiltered = np.delete(repsAcqTime_PulseOx, idxAcqToDiscard, axis=-1)
#     repsAcqTime_Resp_TRfiltered = np.delete(repsAcqTime_Resp, idxAcqToDiscard, axis=-1)
#     print('\nDiscarded '+str(len(idxAcqToDiscard))+' points due to inconsistent effective TR.')
#
#     # plot filtering results if asked
#     if outPlotFname and len(mriSignal.shape) == 1:
#
#         fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(20, 9.7))
#         plt.subplots_adjust(left=0.05, right=0.95, hspace=0.25, bottom=0.05, top=0.9)
#
#         ax2.set_title("Effective TR")
#         ax2.set_xlabel('Time (s)')
#         ax2.set_ylabel('Effective TR (ms)')
#         ax2.grid(which='both')
#         ax2.plot(repsAcqTime_PulseOx/1000, TReff, linewidth=0, marker='+', markersize=7.0, color='red', label='Discarded points')
#         ax2.plot(repsAcqTime_PulseOx_TRfiltered/1000, np.delete(TReff, idxAcqToDiscard, axis=-1), linewidth=0, marker='+', markersize=7.0, color='black', label='Kept points')
#         ax2.axhline(y=1.5*cardiacPeriodMean, linewidth=3, color='red', label='Threshold (mean RR = '+str(cardiacPeriodMean)+'ms)')
#         ax2.axhline(y=0.5*cardiacPeriodMean, linewidth=3, color='red')
#         ax2.legend(loc="lower left")
#
#         ax1.set_title("Effective TR filtering")
#         ax1.set_xlabel('Time (s)')
#         ax1.set_ylabel('Signal')
#         ax1.grid(which='both')
#         ax1.plot(repsAcqTime_PulseOx/1000, mriSignal, linewidth=0, marker='+', markersize=7.0, color='red', label='Discarded points')
#         ax1.plot(repsAcqTime_PulseOx_TRfiltered/1000, mriSignal_TRfiltered, linewidth=0, marker='+', markersize=7.0, color='black', label='Kept points')
#         ax1.legend(loc="lower left")
#
#         fig.suptitle('Effective TR filtering\nSaved to: '+outPlotFname)
#         fig.savefig(outPlotFname)
#         plt.show()
#
#     return mriSignal_TRfiltered, repsAcqTime_PulseOx_TRfiltered, repsAcqTime_Resp_TRfiltered, idxAcqWithBadTR



#%% Functions related to breathing frequencies filtering
def filterResp(mriSignal, mriTime, respSignal, respTime, outPlotFname, cardiacPeriod=0.5, freqDetection='temporal'):

    """Becareful that mriTime and respTime are synchronized correctly AND that mriSignal is sampled regularly."""

    # crop respiration signal to the same time window as MRI signal
    respTimeCrop = respTime[np.abs(respTime - np.min(mriTime)).argmin():np.abs(respTime - np.max(mriTime)).argmin()]
    respSignalCrop = respSignal[np.abs(respTime - np.min(mriTime)).argmin():np.abs(respTime - np.max(mriTime)).argmin()]

    # calculate respiration frequencies and determine cutoffs in Hz
    lowFreqCut, highFreqCut = calculate_respFreq_cutoff(respTimeCrop/1000, respSignalCrop, freqDetection, cardiacPeriod=cardiacPeriod/1000, outPlotFname='')

    # remove the frequency range (in Hz) of respiration signal from MRI signal
    mriSampling_rate = 1000/(mriTime[1]-mriTime[0])
    if highFreqCut <= mriSampling_rate/2:
        mriSignalFiltered = butter_bandstop_filter(mriSignal, lowFreqCut, highFreqCut, mriSampling_rate, outPlotFname=outPlotFname)
    else:
        warnings.warn("***** WARNING *****\nMRI data sampling rate is lower than the maximum breathing frequency"
                      " detected\n==> cannot filter breathing frequencies\n==> MRI signal not filtered")
        mriSignalFiltered = mriSignal

    return mriSignalFiltered, np.array([lowFreqCut, highFreqCut])


def calculate_respFreq_cutoff(respTime, respSignal, freqDetectionMethod, cardiacPeriod=0.5, outPlotFname=''):

    """

    :param respTime: in seconds
    :param respSignal:
    :param freqDetectionMethod:
    :param outPlotFname:
    :return:
    """

    # find peaks of respiration signal
    respSignalMax, respSignalMin = peakdet(respSignal, 975, outPlotFname='')
    respPeriod = np.diff(respTime[respSignalMax[:, 0].astype(int)])  # in seconds
    # remove too short periods which are outliers due to aorta beat in the respiratory bellows during apnea
    respPeriod = respPeriod[respPeriod > 2*cardiacPeriod]

    respFreq = 1 / respPeriod
    print('\nMean +/- SD breathing cycle (seconds) = %.2f +/- %.2f' % (np.mean(respPeriod), np.std(respPeriod)))

    if freqDetectionMethod == 'fourier':

        # Fourier transform
        frequencies, fftAbs = fft_warpper(respTime, respSignal, increase_res_factor=2)

        # fit gaussian model only on frequencies > 0 (symmetry)
        fit_domain = (frequencies >= 0) & (frequencies <= 1)
        gmodel = GaussianModel()
        params = gmodel.guess(fftAbs[fit_domain], x=frequencies[fit_domain])
        gfit = gmodel.fit(fftAbs[fit_domain], params, x=frequencies[fit_domain])

        # take -2sigmas and +2sigmas as low and high frequency cutoffs but if low cutoff frequency is lower than 0.1 Hz,
        # increase it (and decrease the high cutoff) until reaching this threshold
        rangeFact = 2
        lowFreqCut, highFreqCut = gfit.values['center'] - rangeFact*gfit.values['sigma'], gfit.values['center'] + rangeFact*gfit.values['sigma']
        while lowFreqCut < 0.1:
            rangeFact -= 0.1
            lowFreqCut, highFreqCut = gfit.values['center'] - rangeFact * gfit.values['sigma'], gfit.values['center'] + rangeFact * gfit.values['sigma']

        if outPlotFname:
            figResFreqFit, axis = plt.subplots()
            axis.plot(frequencies[fit_domain], fftAbs[fit_domain], 'bo')
            axis.plot(frequencies[fit_domain], gfit.init_fit, 'k--', label='initial fit')
            axis.plot(frequencies[fit_domain], gfit.best_fit, 'r-', label='best fit')
            axis.axvspan(xmin=np.min(respFreq), xmax=np.max(respFreq), label='respiration freq range from temporal analysis', color='tab:olive', alpha=0.2)
            axis.axvspan(xmin=lowFreqCut, xmax=highFreqCut, label='frequency cutoffs', color='tab:pink', alpha=0.2)
            axis.legend(loc='best')
            figResFreqFit.savefig(outPlotFname, transparent=True)
            plt.show()

    elif freqDetectionMethod == 'temporal':

        # take min and max
        lowFreqCut, highFreqCut = np.min(respFreq), np.max(respFreq)

    print('\nRespiration cutoff frequencies: '+str(round(lowFreqCut,4))+' Hz to '+str(round(highFreqCut,4))+' Hz.\n')

    return lowFreqCut, highFreqCut


def butter_bandstop_filter(data, lowcut, highcut, fs, order=3, outPlotFname=''):

    # remove NaN from data (otherwise filter will output only NaN)
    dataNoNan = data[~np.isnan(data)]
    fft = np.fft.fft(dataNoNan, norm='ortho')
    fftFreq = np.fft.fftfreq(dataNoNan.size, d=1/fs)  # in MHz
    idx_ascending_freq = np.argsort(fftFreq)

    # create a bandpass Butterworth filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='bandstop')
    w, h = scipy.signal.freqz(b, a, worN=len(dataNoNan))  # double number of points
    # add also for negative frequencies
    filter = np.concatenate((np.flip(h), h[1:]))
    filterFreq = np.concatenate((-(fs * 0.5 / np.pi) * np.flip(w), (fs * 0.5 / np.pi) * w[1:]))
    # interpolate to initial resolution
    filterInterp = np.interp(fftFreq, filterFreq, filter)

    # apply filter by multiplying FFT by filter
    fftFiltered = fft * filterInterp

    # come back to temporal domain
    dataFiltered = np.fft.ifft(fftFiltered, norm='ortho')

    # add NaN back
    dataFilteredNan = np.repeat(np.nan, data.shape)  #.astype(dataFiltered.dtype)
    dataFilteredNan[~np.isnan(data)] = dataFiltered

    if outPlotFname:
        # plot results
        fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(20, 9.5))
        plt.subplots_adjust(hspace=0.3, bottom=0.05, top=0.92)

        # calculate normalized FFT of both signals for plotting
        freqIdx, fftOriginalSignalAbs = fft_warpper(np.linspace(0, 1/fs*len(dataNoNan), len(dataNoNan)), dataNoNan, increase_res_factor=2)
        freqIdx, fftFilteredSignalAbs = fft_warpper(np.linspace(0, 1/fs*len(dataNoNan), len(dataNoNan)), dataFiltered.astype(float), increase_res_factor=2)

        ax1.set_title('Frequency domain')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.plot(freqIdx, fftOriginalSignalAbs, label='original signal', color='black', lw=0.3, marker='+')
        ax1.plot(freqIdx, fftFilteredSignalAbs, label='filtered signal', color='tab:blue', lw=0.3, marker='o', fillstyle='none')
        ax1.axvspan(xmin=lowcut, xmax=highcut, label='respiration frequency range', color='tab:pink', alpha=0.2)
        ax1.axvspan(xmin=-highcut, xmax=-lowcut, label='_nolegend_', color='tab:pink', alpha=0.2)
        ax1.plot(fftFreq[idx_ascending_freq], np.abs(filterInterp[idx_ascending_freq]), label='filter frequency response', color='tab:pink', lw=2)
        ax1.legend()
        ax1.grid()

        ax2.set_title('Time domain')
        ax2.set_xlabel('Time (s)')
        ax2.plot(np.linspace(0, 1/fs*len(dataNoNan), len(dataNoNan)), dataNoNan, label='original signal: COV='+str(round(100*np.std(dataNoNan)/np.mean(dataNoNan), 2))+'%', color='black', lw=1, marker='+')
        ax2.plot(np.linspace(0, 1/fs*len(dataNoNan), len(dataNoNan)), dataFiltered.astype(float), label='filtered signal: COV='+str(round(100*np.std(dataFiltered.astype(float))/np.mean(dataFiltered.astype(float)), 2))+'%', color='tab:blue', lw=1, marker='o', fillstyle='none')
        ax2.legend()
        ax2.grid()

        fig.suptitle('Saved to: '+outPlotFname)
        fig.savefig(outPlotFname)

        plt.close()

    return dataFilteredNan

# def butter_bandpass(lowcut, highcut, fs, order=3):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = scipy.signal.butter(order, [low, high], btype='bandpass')
#     return b, a


def filterHighFreq(mriSignal, mriTime, respSignal, respTime, outPlotFname):

    """Filter high breathing frequencies from MRI signal (made for baseline filtering).
    Be careful that mriTime and respTime are synchronized correctly AND that mriSignal is sampled regularly.
    """

    # crop respiration signal to the same time window as MRI signal
    respTimeCrop = respTime[np.abs(respTime - np.min(mriTime)).argmin():np.abs(respTime - np.max(mriTime)).argmin()]
    respSignalCrop = respSignal[np.abs(respTime - np.min(mriTime)).argmin():np.abs(respTime - np.max(mriTime)).argmin()]

    # calculate respiration frequencies
    respSignalMax, respSignalMin = peakdet(respSignalCrop, 850, outPlotFname='')
    respPeriod = np.diff(respTimeCrop[respSignalMax[:, 0].astype(int)]) / 1000  # in seconds
    respFreq = 1 / respPeriod

    # remove the frequency range of respiration signal from MRI signal
    mriSignalFiltered = butter_lowpass_filter(mriSignal, np.min(respFreq), 1000/(mriTime[1]-mriTime[0]), outPlotFname=outPlotFname)
    # mriSignalFiltered = butter_lowpass_filter_conventional(mriSignal, np.min(respFreq), 1000/(mriTime[1]-mriTime[0]), outPlotFname=outPlotFname)

    return mriSignalFiltered


def butter_lowpass_filter(data, lowcut, fs, order=5, outPlotFname=''):

    # remove NaN from data (otherwise filter will output only NaN)
    dataNoNan = data[~np.isnan(data)]
    fft = np.fft.fft(dataNoNan, norm='ortho')
    fftFreq = np.fft.fftfreq(dataNoNan.size, d=1/fs)  # in MHz
    idx_ascending_freq = np.argsort(fftFreq)

    # create a bandpass Butterworth filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = scipy.signal.butter(order, low, btype='lowpass', analog=False)
    w, h = scipy.signal.freqz(b, a, worN=len(dataNoNan))  # double number of points
    # add also for negative frequencies
    filter = np.concatenate((np.flip(h), h[1:]))
    filterFreq = np.concatenate((-(fs * 0.5 / np.pi) * np.flip(w), (fs * 0.5 / np.pi) * w[1:]))
    # interpolate to initial resolution
    filterInterp = np.interp(fftFreq, filterFreq, filter)

    # apply filter by multiplying FFT by filter
    fftFiltered = fft * filterInterp

    # come back to temporal domain
    dataFiltered = np.fft.ifft(fftFiltered, norm='ortho')

    # add NaN back
    dataFilteredNan = np.repeat(np.nan, data.shape)  #.astype(dataFiltered.dtype)
    dataFilteredNan[~np.isnan(data)] = dataFiltered

    if outPlotFname:
        # plot results
        fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(20, 9.5))
        plt.subplots_adjust(hspace=0.3, bottom=0.05, top=0.92)

        # calculate normalized FFT of both signals for plotting
        freqAxis, fftOriginalSignalAbs = fft_warpper(np.linspace(0, 1/fs*len(dataNoNan), len(dataNoNan)), dataNoNan, increase_res_factor=1)
        freqAxis, fftFilteredSignalAbs = fft_warpper(np.linspace(0, 1/fs*len(dataNoNan), len(dataNoNan)), dataFiltered.astype(float), increase_res_factor=1)

        ax1.set_title('Frequency domain')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.plot(freqAxis, fftOriginalSignalAbs, label='original signal', color='black', lw=0.3, marker='+')
        ax1.plot(freqAxis, fftFilteredSignalAbs, label='filtered signal', color='tab:blue', lw=0.3, marker='o', fillstyle='none')
        ax1.axvspan(xmin=lowcut, xmax=fs/2-fs/len(dataNoNan), label='filtered frequencies', color='tab:pink', alpha=0.2)
        ax1.axvspan(xmin=-fs/2, xmax=-lowcut, label='_nolegend_', color='tab:pink', alpha=0.2)
        ax1.plot(fftFreq[idx_ascending_freq], np.abs(filterInterp[idx_ascending_freq]), label='filter frequency response', color='tab:pink', lw=2)
        ax1.legend()
        ax1.grid()

        ax2.set_title('Time domain')
        ax2.set_xlabel('Time (s)')
        ax2.plot(np.linspace(0, 1/fs*len(dataNoNan), len(dataNoNan)), dataNoNan, label='original signal: COV='+str(round(100*np.std(dataNoNan)/np.mean(dataNoNan), 2))+'%', color='black', lw=1, marker='+')
        ax2.plot(np.linspace(0, 1/fs*len(dataNoNan), len(dataNoNan)), dataFiltered.astype(float), label='filtered signal: COV='+str(round(100*np.std(dataFiltered.astype(float))/np.mean(dataFiltered.astype(float)), 2))+'%', color='tab:blue', lw=1, marker='o', fillstyle='none')
        ax2.legend()
        ax2.grid()

        fig.suptitle('Saved to: ' + outPlotFname)
        fig.savefig(outPlotFname)

        plt.close()

    return dataFilteredNan


def butter_lowpass_filter_conventional(data, lowcut, fs, order=5, outPlotFname=''):
    """
    Code from https://medium.com/analytics-vidhya/how-to-filter-noise-with-a-low-pass-filter-python-885223e5e9b7
    (Dec 27 2019)
    :param data:
    :param cutoff:
    :param fs:
    :param order:
    :return:
    """

    # remove NaN from data (otherwise filter will output only NaN)
    dataNoNan = data[~np.isnan(data)]

    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = lowcut / nyq
    # Get the filter coefficients
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    dataFiltered = scipy.signal.filtfilt(b, a, dataNoNan)

    if outPlotFname:
        # plot results
        fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(20, 9.5))
        plt.subplots_adjust(hspace=0.3, bottom=0.05, top=0.92)

        # calculate normalized FFT of both signals for plotting
        freqAxis, fftOriginalSignalAbs = fft_warpper(np.linspace(0, 1/fs*len(dataNoNan), len(dataNoNan)), dataNoNan, increase_res_factor=1)
        freqAxis, fftFilteredSignalAbs = fft_warpper(np.linspace(0, 1/fs*len(dataNoNan), len(dataNoNan)), dataFiltered.astype(float), increase_res_factor=1)

        ax1.set_title('Frequency domain')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.plot(freqAxis, fftOriginalSignalAbs, label='original signal', color='black', lw=0.3, marker='+')
        ax1.plot(freqAxis, fftFilteredSignalAbs, label='filtered signal', color='tab:blue', lw=0.3, marker='o', fillstyle='none')
        ax1.axvspan(xmin=lowcut, xmax=fs/2-fs/len(dataNoNan), label='filtered frequencies', color='tab:pink', alpha=0.2)
        ax1.axvspan(xmin=-fs/2, xmax=-lowcut, label='_nolegend_', color='tab:pink', alpha=0.2)
        # ax1.plot(fftFreq[idx_ascending_freq], np.abs(filterInterp[idx_ascending_freq]), label='filter frequency response', color='tab:pink', lw=2)
        ax1.legend()
        ax1.grid()

        ax2.set_title('Time domain')
        ax2.set_xlabel('Time (s)')
        ax2.plot(np.linspace(0, 1/fs*len(dataNoNan), len(dataNoNan)), dataNoNan, label='original signal: COV='+str(round(100*np.std(dataNoNan)/np.mean(dataNoNan), 2))+'%', color='black', lw=1, marker='+')
        ax2.plot(np.linspace(0, 1/fs*len(dataNoNan), len(dataNoNan)), dataFiltered.astype(float), label='filtered signal: COV='+str(round(100*np.std(dataFiltered.astype(float))/np.mean(dataFiltered.astype(float)), 2))+'%', color='tab:blue', lw=1, marker='o', fillstyle='none')
        ax2.legend()
        ax2.grid()

        fig.suptitle('Saved to: ' + outPlotFname)
        fig.savefig(outPlotFname)

        plt.close()

    return dataFiltered


def fft_warpper(time, signal, increase_res_factor=2):

    # resample to regular grid
    t_regular_sampling = np.linspace(np.min(time), np.max(time), increase_res_factor * len(time))
    signal_resampled = np.interp(t_regular_sampling, time, signal)

    # normalize (to remove central frequency and amplitude difference between signals)
    signal_norm = (signal_resampled - np.mean(signal_resampled))/np.std(signal_resampled)

    # calculate FFT and frequency axis
    fft = np.fft.fft(signal_norm, norm='ortho')
    freq = np.fft.fftfreq(t_regular_sampling.size, d=np.mean(np.diff(t_regular_sampling)))  # in MHz
    idx_ascending_freq = np.argsort(freq)

    return freq[idx_ascending_freq], np.abs(fft[idx_ascending_freq])


def peakdet(v, delta, x=None, outPlotFname=''):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html

    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.

    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        os.error('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        os.error('Input argument delta must be a scalar')

    if delta <= 0:
        os.error('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    if outPlotFname:
        fig, ((ax1)) = plt.subplots(1, 1, figsize=(20, 9.5))
        ax1.set_title('Pulse detection')
        ax1.set_xlabel('Time (s)')
        time = np.arange(v.size)*20*1e-3
        ax1.plot(time, v, label='Pulse Ox signal', color='blue', lw=0.3, marker='+')
        for xc in np.array(maxtab)[:, 0]:
            plt.axvline(x=time[xc.astype(int)], color='r', label='peak' if np.where(np.array(maxtab)[:, 0] == xc)[0][0] == 0 else "_nolegend_")
        ax1.legend()
        ax1.grid()
        # plt.show()
        fig.savefig(outPlotFname)

    return np.array(maxtab), np.array(mintab)


def removeRVSfreq(mriSignalInjectRegrid, samplingFreq, rvsDataFname, rvsMaskname, rvsPhysioLogFname, outPlotFname=''):

    # ----------------------------------------------------------------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------------------------------------------------------------
    rvsImg = nib.load(rvsDataFname).get_data()  # MRI image
    rvsMask = nib.load(rvsMaskname).get_data()  # masks

    # ----------------------------------------------------------------------------------------------------------------------
    # extract mean signal in mask within RVS data (no injection)
    # ----------------------------------------------------------------------------------------------------------------------
    rvsMriSignal, _ = extract_signal_within_roi(rvsImg, rvsMask)

    # ----------------------------------------------------------------------------------------------------------------------
    # Physio processing
    # ----------------------------------------------------------------------------------------------------------------------
    rvsRepsAcqTime_PulseOx, rvsTime_PulseOx, rvsValues_PulseOx, rvsRepsAcqTime_Resp, rvsTime_Resp, rvsValues_Resp = extract_acqtime_and_physio(rvsPhysioLogFname, rvsImg.shape[3])

    # ----------------------------------------------------------------------------------------------------------------------
    # Normalize by 1 - exp(-TReffective/T1)
    # ----------------------------------------------------------------------------------------------------------------------
    rvsTReff = np.append(np.diff(rvsRepsAcqTime_PulseOx)[0], np.diff(rvsRepsAcqTime_PulseOx))
    rvsMriSignal_normTR = np.divide(rvsMriSignal, rvsTReff)

    # ----------------------------------------------------------------------------------------------------------------------
    # Filter out points where TR was too long (missed a trigger)
    # ----------------------------------------------------------------------------------------------------------------------
    rvsMriSignal_TRfiltered, rvsRepsAcqTime_PulseOx_TRfiltered, rvsRepsAcqTime_Resp_TRfiltered = discardWrongTRs(rvsTReff, rvsTime_PulseOx, rvsValues_PulseOx, rvsMriSignal_normTR, rvsRepsAcqTime_PulseOx, rvsRepsAcqTime_Resp, '')

    # ----------------------------------------------------------------------------------------------------------------------
    # Regrid RVS data to the same sampling as the INJECT data were resampled
    # ----------------------------------------------------------------------------------------------------------------------
    # MRI
    rvsTimeRegGrid = np.linspace(np.min(rvsRepsAcqTime_PulseOx), np.max(rvsRepsAcqTime_PulseOx), (np.max(rvsRepsAcqTime_PulseOx) - np.min(rvsRepsAcqTime_PulseOx))*samplingFreq/1000)
    rvsSignalRegGrid = np.interp(rvsTimeRegGrid, rvsRepsAcqTime_PulseOx_TRfiltered, rvsMriSignal_TRfiltered)

    # ----------------------------------------------------------------------------------------------------------------------
    # FFT manipulation
    # ----------------------------------------------------------------------------------------------------------------------
    # remove NaN from data (otherwise filter will output only NaN)
    rvsDataNoNan = rvsSignalRegGrid[~np.isnan(rvsSignalRegGrid)]
    rvsFFT = np.fft.fft(rvsDataNoNan, norm='ortho')
    rvsFFTFreq = np.fft.fftfreq(rvsDataNoNan.size, d=1/samplingFreq)  # in MHz
    rvs_idx_ascending_freq = np.argsort(rvsFFTFreq)

    # same for inject data
    injectDataNoNan = mriSignalInjectRegrid[~np.isnan(mriSignalInjectRegrid)]
    injectFFT = np.fft.fft(injectDataNoNan, norm='ortho')
    injectFFTFreq = np.fft.fftfreq(injectDataNoNan.size, d=1/samplingFreq)  # in MHz
    inject_idx_ascending_freq = np.argsort(injectFFTFreq)

    # design filter
    filterFreqResp = np.ones(len(injectFFT))
    rvsFFT_interp = np.interp(injectFFTFreq[inject_idx_ascending_freq], rvsFFTFreq[rvs_idx_ascending_freq], rvsFFT[rvs_idx_ascending_freq])
    rvsFFT_interp_ishift = np.fft.ifftshift(rvsFFT_interp)
    filterFreqResp[np.abs(np.abs(rvsFFT_interp_ishift) - np.abs(injectFFT)) < 0.015] = 0
    filterFreqResp_smoothed = scipy.signal.savgol_filter(filterFreqResp, window_length=7, polyorder=5)
    filterFreqResp_smoothed[filterFreqResp_smoothed > 1] = 1
    filterFreqResp_smoothed[filterFreqResp_smoothed < 0] = 0

    # apply filter
    injectFFTfiltered = injectFFT * filterFreqResp
    injectFFTfilteredSmooth = injectFFT * filterFreqResp_smoothed
    # be sure that we don't change f0
    inject_idxF0 = inject_idx_ascending_freq[np.argwhere(injectFFTFreq == 0)][0]
    injectFFTfiltered[inject_idxF0] = injectFFT[inject_idxF0]
    injectFFTfilteredSmooth[inject_idxF0] = injectFFT[inject_idxF0]

    # come back to temporal domain
    dataFiltered = np.fft.ifft(injectFFTfiltered, norm='ortho')
    dataFilteredSmoothed = np.fft.ifft(injectFFTfilteredSmooth, norm='ortho')

    # add NaN back
    dataFilteredNan = np.repeat(np.nan, mriSignalInjectRegrid.shape)  #.astype(dataFiltered.dtype)
    dataFilteredNan[~np.isnan(mriSignalInjectRegrid)] = dataFiltered
    dataFilteredSmoothedNan = np.repeat(np.nan, mriSignalInjectRegrid.shape)  #.astype(dataFiltered.dtype)
    dataFilteredSmoothedNan[~np.isnan(mriSignalInjectRegrid)] = dataFilteredSmoothed

    if outPlotFname:
        # plot results
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 9.5))
        plt.subplots_adjust(hspace=0.3, bottom=0.05, top=0.92)

        # # calculate normalized FFT of both signals for plotting
        # freqIdxInject, fftInjectSignalAbs = fft_warpper(np.linspace(0, 1/samplingFreq*len(injectDataNoNan), len(injectDataNoNan)), injectDataNoNan, increase_res_factor=1)
        # freqIdxRvs, fftRvsSignalAbs = fft_warpper(np.linspace(0, 1/samplingFreq*len(rvsDataNoNan), len(rvsDataNoNan)), rvsDataNoNan.astype(float), increase_res_factor=1)
        # freqIdxRvsInterp, fftRvsInterpSignalAbs = fft_warpper(np.linspace(0, 1/samplingFreq*len(rvsDataNoNan), len(rvsDataNoNan)), np.fft.ifft(rvsFFT_smoothed, norm='ortho').astype(float), increase_res_factor=1)

        ax1.set_title('Frequency domain')
        ax1.set_xlabel('Frequency (Hz)')
        # ax1.plot(freqIdxInject, fftInjectSignalAbs, label='signal with injection', color='black', lw=0.3, marker='+')
        # ax1.plot(freqIdxRvs, fftRvsSignalAbs, label='signal without injection', color='tab:blue', lw=0.3, marker='+')
        # ax1.plot(freqIdxRvsInterp, fftRvsInterpSignalAbs, label='signal without injection interp', color='tab:green', lw=0.3, marker='+')
        ax1.plot(rvsFFTFreq[rvs_idx_ascending_freq], np.abs(rvsFFT[rvs_idx_ascending_freq]), label='signal without injection', color='tab:blue', lw=0.3, marker='+')
        ax1.plot(injectFFTFreq[inject_idx_ascending_freq], filterFreqResp_smoothed[inject_idx_ascending_freq], label='filter frequency response smoothed', color='tab:orange', lw=0.3, marker='+')
        ax1.plot(injectFFTFreq[inject_idx_ascending_freq], filterFreqResp[inject_idx_ascending_freq], label='filter frequency response', color='tab:green', lw=0.3, marker='+')
        ax1.plot(injectFFTFreq[inject_idx_ascending_freq], np.abs(injectFFT[inject_idx_ascending_freq]), label='signal with injection', color='black', lw=0.3, marker='+')
        ax1.legend()
        ax1.grid()

        ax2.set_title('Time domain')
        ax2.set_xlabel('Time (s)')
        ax2.plot(np.linspace(0, 1/samplingFreq*len(injectDataNoNan), len(injectDataNoNan)), injectDataNoNan, label='signal with injection', color='black', lw=0.3, marker='+')
        ax2.plot(np.linspace(0, 1/samplingFreq*len(rvsDataNoNan), len(rvsDataNoNan)), rvsDataNoNan.astype(float), label='signal without injection', color='tab:blue', lw=0.3, marker='+')
        ax2.legend()
        ax2.grid()

        freqIdxInjectFiltered, fftInjectSignalFilteredAbs = fft_warpper(np.linspace(0, 1/samplingFreq*len(injectDataNoNan), len(injectDataNoNan)), dataFilteredNan, increase_res_factor=1)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Filtered signal')
        # ax3.plot(freqIdxInjectFiltered, fftInjectSignalFilteredAbs, label='signal with injection filtered', color='black', lw=0.3, marker='+')
        ax3.plot(injectFFTFreq[inject_idx_ascending_freq], np.abs(injectFFT[inject_idx_ascending_freq]), label='original signal with injection', color='black', lw=0.3, marker='+')
        ax3.plot(injectFFTFreq[inject_idx_ascending_freq], np.abs(injectFFTfiltered[inject_idx_ascending_freq]), label='signal with injection filtered', color='tab:blue', lw=0.3, marker='+')
        ax3.plot(injectFFTFreq[inject_idx_ascending_freq], np.abs(injectFFTfilteredSmooth[inject_idx_ascending_freq]), label='signal with injection filtered with smooth filter', color='red', lw=0.3, marker='+')
        ax3.legend()
        ax3.grid()

        ax4.set_xlabel('Time (s)')
        ax4.plot(np.linspace(0, 1/samplingFreq*len(injectDataNoNan), len(injectDataNoNan)), injectDataNoNan, label='original signal with injection', color='black', lw=0.3, marker='+')
        ax4.plot(np.linspace(0, 1/samplingFreq*len(injectDataNoNan), len(injectDataNoNan)), dataFilteredNan, label='signal with injection filtered', color='tab:blue', lw=0.3, marker='+')
        ax4.plot(np.linspace(0, 1/samplingFreq*len(injectDataNoNan), len(injectDataNoNan)), dataFilteredSmoothed, label='signal with injection filtered', color='red', lw=0.3, marker='+')
        ax4.legend()
        ax4.grid()

        fig.suptitle('Saved to: '+outPlotFname)
        fig.savefig(outPlotFname)

        plt.close()

    return dataFilteredSmoothed

#%%
def get_physiologFname_TE_injRep(subjID, filename='', baseDir='/Users/slevy/data/cei'):
    """

    :param subjID:
    :return:
    """

    if filename:
        shFile = open(baseDir+'/'+subjID+'/'+filename, 'r')
    else:
        shFile = open(baseDir+'/'+subjID+'/epi/'+subjID+'_process_inject.sh', 'r')

    injRep = 0
    firstPassStart, firstPassEnd = 0.0, 0.0
    content = shFile.readlines()
    for line in content:
        if 'physiologFolder=' in line:
            physiologFolder = line.split('physiologFolder=')[1].strip()
            physiologFolder_absPath = os.path.abspath(os.path.dirname(shFile.name)+'/'+physiologFolder)
        if line[0:4] == 'seq_':
            dcmFileName = line.split('=')[1].strip()
        if 'dcm_path=' in line:
            dcmDir = line.split('dcm_path=')[1].strip()
        if 'injRepNb=' in line:
            injRep = int(line.split('injRepNb=')[1].strip())
        if 'CApass_start=' in line:
            firstPassStart = 1000*float(line.split('CApass_start=')[1].strip())
        if 'CApass_end=' in line:
            firstPassEnd = 1000*float(line.split('CApass_end=')[1].strip())

    # get absolute path of physiolog file and dcm file
    dcm_absPath = os.path.abspath(os.path.dirname(shFile.name)+'/' + dcmDir + '/' + dcmFileName)
    shFile.close()

    # # return absolute path of physiolog file and dcm file
    # if filename:
    #     shFile_dir = os.path.dirname(subjID+'/'+filename)
    #     physiolog_absPath = baseDir + '/' + shFile_dir + '/' + physiologFname
    #     dcm_absPath = baseDir + '/' + shFile_dir + '/' + dcmDir + '/' + dcmFileName
    # else:
    #     physiolog_absPath = baseDir+'/'+subjID+'/epi/'+physiologFname
    #     dcm_absPath = baseDir+'/'+subjID+'/epi/' + dcmDir + '/' + dcmFileName


    # read dcm and extract TE and gap between slices
    # dcm = pydicom.dcmread(dcm_absPath+'/'+os.listdir(dcm_absPath)[-1])
    # TE = float(dcm.EchoTime)
    # gap = float(dcm.SpacingBetweenSlices)
    # TR = float(dcm.RepetitionTime)
    # print('\t>>> Echo time: '+ str(TE) +' ms')
    # print('\t>>> Spacing between slices: '+ str(gap) + ' mm\n')
    physiolog_absPath, TE, gap, TR, acqTime_firstImg, resolution = get_physiologFname_from_dcm(dcm_absPath, physiologFolder_absPath)

    print('\nDicom file path for subject #'+subjID+': '+dcm_absPath+'\n')

    return physiolog_absPath, TE, injRep, gap, TR, acqTime_firstImg, firstPassStart, firstPassEnd, resolution


#%%
def get_physiologFname_from_dcm(dcmFolder, physiologFolder):
    """

    :param:
    :return:
    """

    # read dcm and extract TE and gap between slices
    dcmFiles = sorted([filename for filename in os.listdir(dcmFolder) if filename.endswith('.dcm')])
    dcm = pydicom.dcmread(dcmFolder+'/'+dcmFiles[0])
    TE = float(dcm.EchoTime)
    gap = float(dcm.SpacingBetweenSlices)
    TR = float(dcm.RepetitionTime)
    acqTime_firstImg = datetime.strptime(dcm.AcquisitionDate+'-'+dcm.AcquisitionTime, "%Y%m%d-%H%M%S.%f")
    resolution = np.array(dcm.PixelSpacing)
    print('\t>>> Echo time: '+ str(TE) +' ms')
    print('\t>>> Repetition time: '+ str(TR) + ' ms\n')
    print('\t>>> Spacing between slices: '+ str(gap) + ' mm\n')
    print('\t>>> Acquisition time: '+ str(acqTime_firstImg) + '\n')
    print('\t>>> Resolution: '+ str(resolution) + '\n')

    # find physiolog filename
    physiologFname_list = [filename for filename in os.listdir(physiologFolder) if filename.endswith('.puls')]
    physiologAcqTime_list = []
    for filename in physiologFname_list:
        # get acquisition time of physiolog
        if 'T' in filename:
            physio_time = filename.split('T')[1].split('.')[0][0:6]
            physio_date = filename.split('T')[0][-8:]
        else:
            physio_time = filename.split('_')[3]
            physio_date = filename.split('_')[2]
        physiologAcqTime_list.append(datetime.strptime(physio_date+'-'+physio_time, "%Y%m%d-%H%M%S"))
    # find the closest from the dicom tag
    timeDiff = np.abs(np.subtract(np.array(physiologAcqTime_list), acqTime_firstImg))
    physiologFname = physiologFname_list[timeDiff.argmin()]
    physiolog_absPath = os.path.abspath(physiologFolder+'/'+physiologFname).split('.puls')[0]
    print('\nPhysiolog file path: '+physiolog_absPath+'\n')

    return physiolog_absPath, TE, gap, TR, acqTime_firstImg, resolution

#%%
def get_temporalCOV(signal):
    """

    :param signal:
    :return:
    """

    cov = 100*np.std(signal, axis=-1)/np.mean(signal, axis=-1)
    print('Temporal COV = '+str(np.round(cov, 2))+' %')

    return cov

#%%
def plot_DeltaR2_perSlice(time,
                          signal,
                          TE,
                          axis,
                          lateralTitle='',
                          superiorTitle='',
                          xlabel='',
                          injTime=0,
                          ylims=[],
                          signalLabels=['whole SC', 'Inferior slice', 'Middle slice', 'Superior slice'],
                          timeAcqToDiscard=[],
                          cardiacPeriod=0,
                          stats=True,
                          colorSlices = ['tab:blue', 'tab:orange', 'tab:brown', 'tab:purple', 'tab:green', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']):
    """
    Define function to plot ∆R2 along time on a given axis.

    :param time:
    :param signal: (SC cord, slice 0, slice 1, slice 2) X (repetitions)
    :param TE:
    :param injectionRep:
    :return:
    """

    # convert signal to ∆R2
    if injTime:
        injRep = np.abs(time - injTime).argmin(axis=-1)
    else:
        injRep = 0

    DeltaR2, tSD, _ = calculateDeltaR2(signal, TE, injRep=injRep)

    # plot each slice
    if lateralTitle:
        ylabel = axis.set_ylabel('$\Delta{}R_2^{(*)}\ (s^{-1})$', rotation=90, labelpad=0.5, fontsize=15)
        axis.text(ylabel.get_position()[0]-0.31, ylabel.get_position()[1], lateralTitle, fontsize=17, transform=axis.transAxes)
    axis.set_title(superiorTitle, fontsize=18, pad=30)
    axis.set_xlabel(xlabel, fontsize=18)
    # axes[i_subj].set_xlim([0, np.min([np.max(data['acqTimeRegrid'] - data['timeOffset']) for data in sliceWiseSignal])/1000])
    axis.axhline(y=0, color='tab:gray', lw=0.8, alpha=0.5, label='_nolegend_')
    if injTime:
        axis.axvline(x=injTime, label='injection', color='red', lw=3)
        if signalLabels: signalLabels.insert(0, 'injection')
        # axes[i_subj].axvspan(xmin=CApassTimes[i_subj][0], xmax=CApassTimes[i_subj][1], label='contrast agent pass', color='r', lw=1, alpha=0.2)

    for i_slice in range(signal.shape[0]):
        axis.plot(time[i_slice, :], DeltaR2[i_slice, :], color=colorSlices[i_slice], lw=1.5)

    if signalLabels: axis.legend(signalLabels, loc='center', bbox_to_anchor=(0.5, 1.17), ncol=4, fancybox=True, shadow=True, fontsize=17)
    if stats: axis.text(0.01, 0.01, 'tSD for '+', '.join(signalLabels[1:])+' = '+str(np.array2string(tSD, precision=2, separator=' | ')), transform=axis.transAxes, fontsize=9, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), zorder=11)
    if ylims: axis.set_ylim(ylims)

    # plot discarded acquisitions if asked
    for t in timeAcqToDiscard:
        if cardiacPeriod:
            axis.axvspan(xmin=t-cardiacPeriod/2, xmax=t+cardiacPeriod/2, ymin=0.01, ymax=0.99, color='white', zorder=10)
        else:
            axis.axvline(x=t, color='tab:gray', alpha=0.3, ls=':')

    axis.tick_params(labelsize=16)
    if not xlabel:
        axis.tick_params(axis='x',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False)  # labels along the bottom edge are off

    return axis.get_ylim()


#%%
def calculateDeltaR2(signal, TE, injRep=0, r2=1):

    """

    Convert signal to ∆R2.

    :param signal:
    :param TE: in ms
    :param injRep:
    :return:
    """

    # convert signal to ∆R2
    if np.array(injRep).any():
        S0 = np.array([np.mean(signal[i_signal, 0:injRep[i_signal]+1]) for i_signal in range(signal.shape[0])])
    else:
        S0 = np.mean(signal, axis=-1)
    S_over_S0 = np.divide(signal, np.tile(S0, (signal.shape[1], 1)).T)
    DeltaR2 = - np.log( S_over_S0 ) / ( r2 * TE / 1000 )
    # calculate tSD (and not tCOV because in ∆R2 the mean is 0)
    if np.array([injRep]).any():
        tSD = np.array([np.std(DeltaR2[i_signal, 0:injRep[i_signal]+1]) for i_signal in range(signal.shape[0])])  #cov_baseline = dsc_utils.get_temporalCOV(DeltaR2[:, 0:injRep])
    else:
        tSD = np.std(DeltaR2, axis=-1)  #cov_baseline = dsc_utils.get_temporalCOV(DeltaR2)

    return DeltaR2, tSD, S0


def saveAsNifti(OldNii, data, oFname, dataType=np.float32):

    # if nifty1
    if OldNii.header['sizeof_hdr'] == 348:
        newNii = nib.Nifti1Image(data, OldNii.affine, header=OldNii.header)
    # if nifty2
    elif OldNii.header['sizeof_hdr'] == 540:
        newNii = nib.Nifti2Image(data, OldNii.affine, header=OldNii.header)
    else:
        raise IOError('Input image header problem')
    # save file
    newNii.set_data_dtype(dataType)  # set header data type to float
    nib.save(newNii, oFname+'.nii.gz')

