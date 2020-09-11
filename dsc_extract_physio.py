#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot physiological signal vs. time and return acquisition time for each repetition.


Created on Tue Jul  4 17:45:43 2017

@author: slevy
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import os
from datetime import datetime

class Param:
    def __init__(self):
        self.out_fname = 'physio'
        self.sampling_period = 20  # ms
        self.physiolog_type = 'cmrr'  # type of physiolog, either "slr" (as coded by Simon Levy-Rosetti) or "cmrr" (as coded in CMRR sequences)


def main(log_fname, out_fname):
    """Main."""

    # different processing depending on the physiolog type
    if param_default.physiolog_type == 'slr':

        # extract physio signal
        time, physio_values, epi_acqtime, epi_event, acq_window = read_physiolog(log_fname)

        # sort event times
        reps_table = sort_event_times(epi_acqtime, epi_event)

        # plot physio signal
        plot_physio(time, physio_values, epi_acqtime, reps_table, acq_window, out_fname)

        # write acquisition time of each measurement
        pickle.dump([time, epi_acqtime, reps_table, physio_values, acq_window], open(out_fname+"_trigtimes.pickle", "wb"))

    elif param_default.physiolog_type == 'cmrr':

        # extract physio signal
        time, trigger_start_time, trigger_end_time, physio_values, acq_window = read_physiolog_cmrr(log_fname)

        # plot physio signal along with trigger start and end time
        plot_physio_cmrr(time, trigger_start_time, trigger_end_time, physio_values, acq_window, out_fname)

        # write acquisition time of each measurement
        pickle.dump([time, trigger_start_time, trigger_end_time, physio_values, acq_window], open(out_fname + "_trigtimes.pickle", "wb"))

    print('****Done.****')

def read_physiolog(path, sampling_period=20):
    """
    Read physio logfile and parse it.
    :param path:
           sampling_period: in ms
    :return:
    """

    file = open(path, 'r')
    text = file.readlines()
    physio_sig = np.array(text[2].strip().split(' '), dtype=str)

    # get stats in file footer
    acq_window_idx_line = [text.index(line) for line in text if "AcqWin" in line]
    acq_window = float(text[acq_window_idx_line[0]].strip().split(' ')[-1])

   # get time axis, time of trigger start and time of trigger end
    time, epi_acqtime, epi_event, physio_values = [], [], [], []
    sampling_count = 0
    for i_meas in range(len(physio_sig)):
        if (physio_sig[i_meas] not in ['5000', '6000', '5002', '6002']) and ("[" not in physio_sig[i_meas]):
            time.append(sampling_count*sampling_period)
            physio_values.append(int(physio_sig[i_meas]))
            sampling_count += 1
        elif ("[" in physio_sig[i_meas]) and ("]" in physio_sig[i_meas]):
            epi_acqtime.append((sampling_count-1)*sampling_period)
            epi_event.append(physio_sig[i_meas])

    return np.array(time), np.array(physio_values), np.array(epi_acqtime), np.array(epi_event), acq_window



def read_physiolog_cmrr(path, sampling_period=20):
    """
    Read physio logfile and parse it.
    :param path:
    :return:
    """

    file = open(path, 'r')
    text = file.readlines()
    physio_sig = np.array(text[0].strip().split(' '), dtype=str)

    # get useful data in file footer
    # Acquisition window
    acq_window_idx_line = [text.index(line) for line in text if "AcqWin" in line]
    acq_window = float(text[acq_window_idx_line[0]].strip().split(' ')[-1])
    # Start time of acquisition
    acqStartTime_idx_line = [text.index(line) for line in text if "LogStartMDHTime:" in line]
    # acqStartTime = datetime.strptime(acqStartTime_date+'-'+str(int(acqStartTime_seconds/1000))+'.'+str(acqStartTime_seconds)[-5:], "%Y%m%d-%S.%f")
    # acqStartTime = datetime.timedelta(milliseconds=float(text[acqStartTime_idx_line[0]].strip().split('LogStartMDHTime:')[-1]))
    acqStartTime_time = datetime.utcfromtimestamp(float(text[acqStartTime_idx_line[0]].strip().split('LogStartMDHTime:')[-1]) / 1000.0)
    acqStartTime_date = datetime.strptime(os.path.basename(path).split('_')[2], "%Y%m%d")
    acqStartTime = datetime.combine(acqStartTime_date.date(), acqStartTime_time.time())

    # # remove first (6002=end of info added by sequence, which was at the opening of the logfile) and last (last 5002=section added to indicate acquisition end) elements
    # idx_start, idx_end = np.min(np.where(physio_sig == '6002')), np.max(np.where(physio_sig == '5002'))
    # physio_sig = physio_sig[idx_start+1:idx_end]

    # get time axis, time of trigger start, time of trigger end and physiological values
    time, trigger_start_times, trigger_end_times, physio_values = [], [], [], []
    sampling_count = 0
    logInfo_ON = False
    for i_meas in range(len(physio_sig)):
        if (physio_sig[i_meas] not in ['5000', '6000', '5002', '6002']) and ("[" not in physio_sig[i_meas]) and (not logInfo_ON):
            time.append(sampling_count*sampling_period)
            physio_values.append(int(physio_sig[i_meas]))
            sampling_count += 1
        elif physio_sig[i_meas] == '5000':
            trigger_start_times.append(sampling_count*sampling_period)
        elif physio_sig[i_meas] == '6000':
            trigger_end_times.append(sampling_count*sampling_period)
        elif physio_sig[i_meas] == '5002':
            logInfo_ON = True
        elif physio_sig[i_meas] == '6002':
            logInfo_ON = False

    return np.array(time), np.array(trigger_start_times), np.array(trigger_end_times), np.array(physio_values), acq_window, acqStartTime


def sort_event_times(epi_acqtimes, epi_events):
    """

    :param epi_acqtimes:
    :param epi_events:
    :return:
    """

    if len(epi_acqtimes) != len(epi_events):
        os.error("ERROR: Number of times and events are different.")

    # extract rep and slice numbers of each scan event
    reps, slices = [], []
    for i_event in range(len(epi_acqtimes)):
        reps.append(int(epi_events[i_event].strip().split('Rep#=')[1].split(',')[0]))
        slices.append(int(epi_events[i_event].strip().split('Slice#=')[1].split(',')[0]))
    reps = np.array(reps)
    slices = np.array(slices)

    # get repetitions numbers, acquisition time for each slice of each rep, real or dummy scan
    n_slices = max(slices)+1
    n_reps = int(epi_events.size/n_slices)

    # get acquisition time of each slice for each rep
    slices_table = np.zeros((int(n_reps), n_slices))
    for i_slice in range(n_slices):
        slices_table[:, i_slice] = epi_acqtimes[np.where(slices == i_slice)[0]]

    # get rep number and dummy or real scan
    reps_table = np.zeros((int(n_reps), 2))  # rep number, dummy/real scan (0: dummy scan, 1: real scan)
    idx_new_rep = np.where(slices == min(slices))[0]
    reps_table[:, 0] = reps[idx_new_rep]  # get rep number
    # define for each rep if it is a dummy scan or not
    for i_new_rep in range(0, len(idx_new_rep)):
        if reps_table[i_new_rep, 0] == 1 and reps_table[i_new_rep-1, 0] == 0:
            # if we start the second real scan, the previous one was also a real scan
            reps_table[i_new_rep-1, 1] = 1
            reps_table[i_new_rep, 1] = 1
        elif reps_table[i_new_rep, 0] > 1:
            # if the rep number is more than 1, it was also a real scan
            reps_table[i_new_rep, 1] = 1

    return reps_table, slices_table


def plot_physio(time, physio_sig, epi_acqtime, reps_table, acq_window, out_fname):

    fig = plt.figure(figsize=(20, 9.5))
    plt.title('Saved to: '+out_fname+'_plot.pdf')
    plt.plot(time/1000., physio_sig, '+-', label='physio signal', color='b')
    # add vertical rectangle for each repetition period
    legend_label_counter = [0, 0]
    nSlices = int(len(epi_acqtime)/reps_table.shape[0])
    for i_rep in range(reps_table.shape[0]):
        if reps_table[i_rep, 1] == 1:
            plt.axvspan(epi_acqtime[i_rep*nSlices]/1000., (epi_acqtime[i_rep*nSlices] + acq_window)/1000., facecolor='orange', alpha=0.15, label='acquisition window' if sum(legend_label_counter) == 0 else "_nolegend_")
            plt.axvspan(epi_acqtime[i_rep*nSlices]/1000., epi_acqtime[i_rep*nSlices+nSlices-1]/1000., facecolor='r', alpha=0.25, label='repetitions' if legend_label_counter[0] == 0 else '_nolegend_')
            legend_label_counter[0] += 1
        else:
            plt.axvspan(epi_acqtime[i_rep*nSlices]/1000., (epi_acqtime[i_rep*nSlices] + acq_window)/1000., facecolor='orange', alpha=0.15, label='acquisition window' if sum(legend_label_counter) == 0 else "_nolegend_")
            plt.axvspan(epi_acqtime[i_rep*nSlices]/1000., epi_acqtime[i_rep*nSlices+nSlices-1]/1000., facecolor='gray', alpha=0.25, label='dummy scans' if legend_label_counter[1] == 0 else '_nolegend_')
            legend_label_counter[1] += 1

    # add vertical lines for each slice (each epi event actually)
    for xc in epi_acqtime:
        plt.axvline(x=xc/1000., color='g', label='slices' if np.where(epi_acqtime==xc)[0][0] == 0 else "_nolegend_")
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Physio signal')

    plt.show(block=True)
    fig.savefig(out_fname+'_plot.pdf')
    plt.close()


def extract_acqTimes_cmrr(triggerStartTime, acqTime_firstImg, acqStartTime, triggerEndTime):

    """

    :param triggerStartTime: in milliseconds
    :param acqTime_firstImg: datetime object
    :param acqStartTime: datetime object (
    :return: acquisition times in milliseconds
    """

    # remove all triggers start time without paired trigger end time
    idxTrigToKeep = []
    for i_trig in range(len(triggerStartTime)):
        if (i_trig == len(triggerStartTime)-1) and (triggerEndTime > triggerStartTime[i_trig]).any():
            idxTrigToKeep.append(i_trig)
        elif ((triggerEndTime > triggerStartTime[i_trig]) & (triggerEndTime < triggerStartTime[i_trig+1])).any():
            idxTrigToKeep.append(i_trig)
    triggerStartTime_notStopped = triggerStartTime[idxTrigToKeep]

    # get the duration of dummy or auto-calibration scans in microseconds
    seqInitDuration = acqTime_firstImg - acqStartTime

    # only keep trigger times after the sequence initialization period
    triggerFirstImg_idx = np.abs(triggerStartTime_notStopped - seqInitDuration.total_seconds()*1000).argmin()
    acqTimes = triggerStartTime_notStopped[triggerFirstImg_idx:]

    return acqTimes


def plot_physio_cmrr(time, trigger_start_times, trigger_end_times, physio_sig, acq_window, out_fname):

    if trigger_start_times.shape[0] != trigger_end_times.shape[0]:
        os.error("ERROR: Number of start and end times are different.")

    fig = plt.figure("CMRR physiolog signal", figsize=(30, 20))

    plt.plot(time, physio_sig, '+-', label='physio signal', color='b')

    # add vertical rectangle for each trigger signal and add acquisition window
    for i_trig in range(trigger_start_times.shape[0]):
        plt.axvspan(trigger_start_times[i_trig], trigger_start_times[i_trig] + acq_window, facecolor='orange', alpha=0.15, label='acquisition window' if i_trig == 0 else "_nolegend_")
        plt.axvspan(trigger_start_times[i_trig], trigger_end_times[i_trig], facecolor='g', alpha=0.25, label='trigger signal' if i_trig == 0 else "_nolegend_")

    # # add vertical lines for trigger start and end
    # for t_start, t_end in zip(trigger_start_times, trigger_end_times):
    #     plt.axvline(x=t_start, color='g')
    #     plt.axvline(x=t_end, color='r')
    #     plt.text(t_start, 1000, 'Trigger period = '+str(t_end-t_start)+'ms', rotation=90)

    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Physio signal')

    plt.show(block=False)

    fig.savefig(out_fname+'_plot.pdf')


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Plot physiological signal vs. time and return acquisition time for each repetition.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-i', dest='ifname', help='Path to physio log file.', type=str, required=True)
    requiredArgs.add_argument('-o', dest='ofname', help='Output file name for plot and file storing acquisition times.', type=str, required=True)

    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()

    # params
    param_default = Param()

    # run main
    main(log_fname=args.ifname, out_fname=args.ofname)
