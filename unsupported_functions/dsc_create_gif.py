#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This program creates a GIF animation for a 4D volume.


Created on Mon Oct  14 19:21:50 2019

@author: slevy
"""

import nibabel as nib
import argparse
import imageio
import numpy as np
import time
import os
import shutil
import dsc_utils
import matplotlib.pyplot as plt
import dsc_pipelines
import scipy.misc

def main(iFname, oFname, duration, cropX, cropY, cropT, processSHFilename, maskFname, subjID):

    # load data
    data4d = nib.load(iFname).get_data()

    # define cropping boundaries
    if not cropX: cropX = [0, data4d.shape[0]-1]
    if not cropY: cropY = [0, data4d.shape[1]-1]
    if not cropT: cropT = [0, data4d.shape[3]-1]

    if not processSHFilename:
        # directly create GIF (without generating PNG)
        with imageio.get_writer(oFname+'.gif', mode='I', duration=duration) as writer:
            for i_vol in range(cropT[0], cropT[1]+1):
                vol_3slices_view = np.concatenate((np.rot90(data4d[cropX[0]:cropX[1], cropY[0]:cropY[1], 0, i_vol]),
                                                   np.rot90(data4d[cropX[0]:cropX[1], cropY[0]:cropY[1], 1, i_vol]),
                                                   np.rot90(data4d[cropX[0]:cropX[1], cropY[0]:cropY[1], 2, i_vol])), axis=0)
            writer.append_data(vol_3slices_view)

    else:
        # extract volumes acquisition time and TR
        physioLogFname, TE, injRep, gap, TR, acqTime_firstImg, firstPassStart, firstPassEnd, resolution = dsc_utils.get_physiologFname_TE_injRep(subjID, filename=processSHFilename)
        repsAcqTime, timePhysio, valuesPhysio = dsc_utils.extract_acqtime_and_physio_by_slice(physioLogFname, data4d.shape[2], data4d.shape[3], acqTime_firstImg, TR=TR)

        if not maskFname:
            # save all frames to PNG in a temporary directory
            tmpDirPath = oFname+"_%s" % time.strftime("%y%m%d%H%M%S")
            os.makedirs(tmpDirPath)
            for i_vol in range(cropT[0], cropT[1]+1):
                vol_3slices_view = np.concatenate((np.rot90(data4d[cropX[0]:cropX[1], cropY[0]:cropY[1], 0, i_vol]),
                                                   np.rot90(data4d[cropX[0]:cropX[1], cropY[0]:cropY[1], 1, i_vol]),
                                                   np.rot90(data4d[cropX[0]:cropX[1], cropY[0]:cropY[1], 2, i_vol])), axis=0)
                imageio.imwrite(tmpDirPath + '/frame' + str(i_vol) + '.jpeg', vol_3slices_view, cmin=0.0, cmax=500.0)

        else:

            # apply signal processing pipeline
            acqTime, signals, acqTime_TRfilt, signals_TRfilt, idxAcqToDiscard, signals_breath, signals_smooth, timePhysio, valuesPhysio = dsc_pipelines.processSignal_bySlice(iFname, maskFname, physioLogFname, acqTime_firstImg)

            # convert to ∆R2(*)
            DeltaR2, _, _ = dsc_utils.calculateDeltaR2(signals[np.newaxis, 0, :], TE, injRep=0)

            # discard wrong TRs to compute temporal SD
            DeltaR2_TRfilt, tSD_TRfilt, _ = dsc_utils.calculateDeltaR2(signals_TRfilt[np.newaxis, 0, cropT[0]:cropT[1]], TE, injRep=0)
            print('\n>>> Temporal ∆R2(*) SD on signal without inconsistent TRs in s-1 = '+str(np.round(tSD_TRfilt,3)))
            # pulseOxSignalMax, pulseOxSignalMin = dsc_utils.peakdet(values_PulseOx, 700)
            # cardiacPeriods = np.diff(Time_PulseOx[pulseOxSignalMax[:, 0].astype(int)])  # in milliseconds
            # cardiacPeriodMean = np.mean(cardiacPeriods)
            # idxAcqWithBadTR = np.argwhere((TReff >= 1.5 * cardiacPeriodMean) | (TReff <= 0.5 * cardiacPeriodMean))

            # save all frames to PNG in a temporary directory
            tmpDirPath = oFname + "_%s" % time.strftime("%y%m%d%H%M%S")
            os.makedirs(tmpDirPath)

            # # arrange view of the 3 slices for the mask
            # mask_noBackground = np.ma.masked_where(mask == 0.0, mask)
            # mask_3slices_view = np.concatenate((np.rot90(mask_noBackground[cropX[0]:cropX[1], cropY[0]:cropY[1], 0]),
            #                                    np.rot90(mask_noBackground[cropX[0]:cropX[1], cropY[0]:cropY[1], 1]),
            #                                    np.rot90(mask_noBackground[cropX[0]:cropX[1], cropY[0]:cropY[1], 2])), axis=1)

            # set the time starting at 0
            repsAcqTime[0, :, 0] = repsAcqTime[0, :, 0] - repsAcqTime[0, cropT[0], 0]
            for i_vol in range(cropT[0], cropT[1] + 1):

                if i_vol not in idxAcqToDiscard:  # remove inconsistent TRs

                    vol_3slices_view = np.concatenate((np.rot90(data4d[cropX[0]:cropX[1], cropY[0]:cropY[1], 0, i_vol]),
                                                       np.rot90(data4d[cropX[0]:cropX[1], cropY[0]:cropY[1], 1, i_vol]),
                                                       np.rot90(data4d[cropX[0]:cropX[1], cropY[0]:cropY[1], 2, i_vol])), axis=1)
                    fig_i_vol, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 3))
                    plt.subplots_adjust(wspace=0.1, left=0.1, right=0.99, hspace=0.0, bottom=0.2, top=0.95)

                    ax1.imshow(vol_3slices_view, cmap='gray', vmin=10, vmax=600)
                    # ax1.imshow(mask_3slices_view, cmap='Blues', alpha=0.5, clim=(.5, 1))
                    # ax1.set_axis_off()
                    ax1.set_xlabel('Rep #'+str(i_vol))
                    ax1.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
                    ax1.patch.set_visible(False)

                    # ax2.plot(repsAcqTime[0, :, 0]/1000, DeltaR2[0, :], color='tab:blue', lw=2.0)
                    # Below: for the figure on the effects of swallowing to show breathing instructions
                    if i_vol in [70, 71, 72, 73, 120, 121, 122, 123, 160, 161, 162, 163, 200, 201, 202, 203, 240, 241, 242, 243]:
                        ax2.set_facecolor('tab:olive')
                    ax2.plot(acqTime[0, :, 0]/1000, DeltaR2[0, :], color='tab:blue', lw=2.0)
                    ax2.axvline(x=acqTime[0, i_vol, 0]/1000, color='red')
                    ax2.axhline(y=DeltaR2[0, i_vol], color='red')
                    # ax2.set_xlim([repsAcqTime[0, cropT[0], 0]/1000, repsAcqTime[0, cropT[1], 0]/1000])
                    ax2.set_ylim([-6.0, 7.0])  #ax2.set_ylim([200, 290])
                    ax2.set_xlabel('Time (s)')
                    ax2.set_ylabel('$\Delta{}R_2\ (s^{-1})$')
                    ax2.grid()

                    # ax2physio = ax2.twinx()
                    # ax2physio.plot(Time_Resp/1000, values_Resp/1000, color='black', alpha=0.5, lw=0.7, label='respiratory signal')
                    # ax2physio.tick_params(right=False, labelright=False)
                    # ax2physio.legend(loc='upper right')

                    fig_i_vol.savefig(tmpDirPath + '/frame' + str(i_vol) + '.jpeg', transparent=False)
                    plt.close(fig_i_vol)

        # prepare and run the imageMagick command with variable effective TR
        shellCmd = 'convert '
        TReff = np.append(np.diff(repsAcqTime[0, :, 0])[0], np.diff(repsAcqTime[0, :, 0]))
        for i_vol in range(data4d.shape[3]):
            shellCmd += '-delay '+str(TReff[i_vol]/100)+' '+ tmpDirPath + '/frame' + str(i_vol) + '.jpeg '
        shellCmd += '-loop 0 '+oFname+'.gif'
        print('Run command: '+shellCmd)
        os.system(shellCmd)
        # time.sleep(5)
        shutil.rmtree(tmpDirPath)

    print('\nSaved to: '+oFname+'.gif')


# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program creates a GIF animation from a 4D volume.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-i', dest='iFname', help='Path to 4D MRI data.', type=str, required=True)
    requiredArgs.add_argument('-o', dest='oFname', help='Filename for output GIF (do not include the extension .gif).', type=str, required=True)
    requiredArgs.add_argument('-subj', dest='subjID', help='Subject ID.', type=str, required=True)

    optionalArgs.add_argument('-d', dest='duration', help='Duration between each frame (in seconds).', type=float, required=False, default=0.5)
    optionalArgs.add_argument('-cx', dest='cropX', help='Cropping boundaries along X as x1,x2.', type=str, required=False, default='')
    optionalArgs.add_argument('-cy', dest='cropY', help='Cropping boundaries along Y as y1,y2.', type=str, required=False, default='')
    optionalArgs.add_argument('-ct', dest='cropT', help='Cropping boundaries along T as t1,t2.', type=str, required=False, default='')
    optionalArgs.add_argument('-sh', dest='processSHFilename', help='Filename of the process .sh file to get the path to physiolog folder.', type=str, required=False, default='')
    optionalArgs.add_argument('-m', dest='maskFname', help='Path to mask for the 4D MRI data in input.', type=str, required=False, default='')

    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()

    # run main
    main(iFname=args.iFname, oFname=args.oFname, duration=args.duration, cropX=list(map(int, args.cropX.split(','))),
         cropY=list(map(int, args.cropY.split(','))), cropT=list(map(int, args.cropT.split(','))), processSHFilename=args.processSHFilename, maskFname=args.maskFname, subjID=args.subjID)

