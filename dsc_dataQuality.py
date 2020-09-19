#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program computes the mean image during baseline, from injection to end of first pass and the difference between those two mean images.


Created on Mon Oct  14 19:21:50 2019

@author: slevy
"""

import dsc_utils
import nibabel as nib
import numpy as np
import argparse
from scipy.io import loadmat
import os
import dsc_pipelines
import matplotlib.pyplot as plt


def main(imgFname, maskFname, dcmFolder, physiologFolder, dataLabel, baselineEndRep):
    """Main.
    """

    # ----------------------------------------------------------------------------------------------------------------------
    # Get physiolog file and TE
    # ----------------------------------------------------------------------------------------------------------------------
    physioLogFname, TE, gap, TR, acqTime_firstImg, resolution = dsc_utils.get_physiologFname_from_dcm(dcmFolder, physiologFolder)

    # ----------------------------------------------------------------------------------------------------------------------
    # Process signal (SC, slices)
    # ----------------------------------------------------------------------------------------------------------------------
    acqTime, signals, acqTime_TRfilt, signals_TRfilt, idxAcqToDiscard, signals_breath, signals_smooth, timePhysio, valuesPhysio = dsc_pipelines.processSignal_bySlice(imgFname, maskFname, physioLogFname, acqTime_firstImg)

    # ----------------------------------------------------------------------------------------------------------------------
    # Plot result figure
    # ----------------------------------------------------------------------------------------------------------------------
    fig = plt.figure(constrained_layout=False, figsize=(17, 4))
    gs = fig.add_gridspec(2, 9, wspace=0, hspace=0.2, left=0.03, right=0.98, bottom=0.11, top=0.84)

    # Signal stability
    ax0 = fig.add_subplot(gs[:, 0:6])

    # if apnea, get the ending time
    baselineEndTime = 0
    if baselineEndRep:
        baselineEndTime = acqTime_TRfilt[0, baselineEndRep, 0] / 1000

    ylims = dsc_utils.plot_DeltaR2_perSlice(acqTime_TRfilt[:, :, 0] / 1000, signals_TRfilt, TE, ax0, lateralTitle="", superiorTitle=dataLabel, injTime=baselineEndTime, ylims=[-8, 8], xlabel='Time (s)', timeAcqToDiscard=acqTime[0, idxAcqToDiscard, 0]/1000)
    # add respiratory bellows signal
    ax0physio = ax0.twinx()
    curveBreath = ax0physio.plot(timePhysio[:, 1]/1000, valuesPhysio[:, 1]/1000, color='gray', alpha=0.3, lw=0.7, label='respiratory belt signal')
    ax0physio.axis("off")

    # Mean image
    img = nib.load(imgFname).get_data()
    meanImage = np.mean(img, axis=3)

    axes1 = [fig.add_subplot(gs[0, 6]), fig.add_subplot(gs[0, 7]), fig.add_subplot(gs[0, 8])]
    for i_z in range(meanImage.shape[2]):
        axes1[i_z].set_title('Z='+str(i_z))
        c = axes1[i_z].imshow(np.rot90(meanImage[:, :, i_z]), cmap='gray', clim=(0, 100))
        axes1[i_z].axis("off")
    cbar = fig.colorbar(c, ax=axes1[meanImage.shape[2]-1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    # for temporal SNR
    if baselineEndRep:
        tSNR = meanImage/np.std(img[:,:,:,0:baselineEndRep], axis=3)
    else:
        tSNR = meanImage/np.std(img, axis=3)
    mask = nib.load(maskFname).get_data()
    mean_tSNR = np.round(np.mean(tSNR[mask>0]), 1)

    axes2 = [fig.add_subplot(gs[1, 6]), fig.add_subplot(gs[1, 7]), fig.add_subplot(gs[1, 8])]
    for i_z in range(meanImage.shape[2]):
        c = axes2[i_z].imshow(np.rot90(tSNR[:, :, i_z]), cmap='hot', clim=(0, 10))
        axes2[i_z].axis("off")
    axes2[0].set_title('Mean tSNR='+str(mean_tSNR))
    cbar = fig.colorbar(c, ax=axes2[meanImage.shape[2]-1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    # turn off axis of third slice if only 2 slices were acquired
    for i_z in range(meanImage.shape[2], 3):
        axes1[i_z].axis("off")
        axes2[i_z].axis("off")

    oFolder = '/'.join(os.path.abspath(dcmFolder).split('/')[0:-2])
    fig.savefig(oFolder + '/fig_qc_'+dataLabel+'.pdf', transparent=True)


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program plot ∆R2 along time, mean image and temporal SNR maps.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-i', dest='imgFname', help='Path to NII data file.', type=str, required=True)
    requiredArgs.add_argument('-m', dest='maskFname', help='Path to NII mask file.', type=str, required=True)
    requiredArgs.add_argument('-dcm', dest='dcmFolder', help='Dicom folder path.', type=str, required=True)
    requiredArgs.add_argument('-p', dest='physiologFolder', help='Physiologs folder path.', type=str, required=True)
    requiredArgs.add_argument('-l', dest='dataLabel', help='Label or title for the plot.', type=str, required=True)

    optionalArgs.add_argument('-b', dest='baselineEndRep', help='Last repetition of the baseline.', type=int, required=False, default=0)

    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()

    # run main
    main(imgFname=args.imgFname, maskFname=args.maskFname, dcmFolder=args.dcmFolder, physiologFolder=args.physiologFolder, dataLabel=args.dataLabel, baselineEndRep=args.baselineEndRep)

