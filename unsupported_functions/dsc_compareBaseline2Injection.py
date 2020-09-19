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

def main(imgFname, matFname, injRep):
    """Main.
    """

    # ----------------------------------------------------------------------------------------------------------------------
    # Load matfile to get injection rep, first pass start and end reps
    # ----------------------------------------------------------------------------------------------------------------------
    data = loadmat(matFname)

    # ----------------------------------------------------------------------------------------------------------------------
    # Compute images
    # ----------------------------------------------------------------------------------------------------------------------
    img_nii = nib.load(imgFname)  # MRI image
    img = np.copy(img_nii.get_data())
    # baseline and injection
    baseline = np.mean(img[:, :, :, 0:injRep+1], axis=3)
    injection = np.mean(img[:, :, :, int(data["firstPassStartRep"]):int(data["firstPassEndRep"])+1], axis=3)
    # difference
    difference = injection - baseline

    # ----------------------------------------------------------------------------------------------------------------------
    # Save as nii
    # ----------------------------------------------------------------------------------------------------------------------
    # extract filename and extension
    oDir, oFilenameWithExt = os.path.split(imgFname)
    # oFilename, oExtension = os.path.splitext(oFilenameWithExt)
    oFilenameWithExtSplit = oFilenameWithExt.split('.')
    # save nii
    dsc_utils.saveAsNifti(img_nii, baseline, oDir+'/'+oFilenameWithExtSplit[0]+'_baseline', dataType=np.float64)
    dsc_utils.saveAsNifti(img_nii, injection, oDir+'/'+oFilenameWithExtSplit[0]+'_injection', dataType=np.float64)
    dsc_utils.saveAsNifti(img_nii, difference, oDir+'/'+oFilenameWithExtSplit[0]+'_difference', dataType=np.float64)

# ==========================================================================================
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='This program computes the mean image during baseline, from injection to end of first pass and the difference between those two mean images.')

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-i', dest='imgFname', help='Path to MRI data file.', type=str, required=True)
    requiredArgs.add_argument('-m', dest='matFname', help='Path to .mat file containing repetition numbers.', type=str, required=True)
    requiredArgs.add_argument('-inj', dest='injRep', help='Repetition number of the injection.', type=int, required=True)

    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()

    # run main
    main(imgFname=args.imgFname, matFname=args.matFname, injRep=args.injRep)

