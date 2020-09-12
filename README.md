# :mag_right: :bulb: 7T-DSC-MRI-Toolbox :flashlight: :wrench: </font>

<p align="center">
This Python-based toolbox provides functions and pipeline to process Dynamic Susceptibility Contrast (DSC) MRI data acquired at 7T. A dataset of DSC MRI acquired in the spinal cord of a healthy volunteer as published in the reference below (healthy volunteer HC1) is also provided. More particularly, this toolbox provides functions to extract acquisition times from physiologs (Siemens format) for cardiac-gated acquisitions, correct signal by effective TR and discard inconsistent TRs as well as to filter breathing frequencies in the signal based on the signal measured by any respiratory device in a repiration physiolog (Siemens format) are available.</p>

We thank you for choosing our toolbox! :heart: According to the Apache license 2.0, please cite the following reference:
> **Lévy S, Roche P-H, Callot V. Dynamic Susceptibility Contrast imaging at 7T for spinal cord perfusion mapping in Cervical Spondylotic Myelopathy patients, In: *Proc. Intl. Soc. Mag. Reson. Med. 28*. 2019;3195.**

# Code description

This toolbox includes Python scripts that be run directly from a bash environment as well as Python functions that are called by the scripts. Here is an overview of those functions:
  - `dsc_image2concentration.py`: this script converts a time-series 4D volume to contrast agent concentration values after voxel-wise signal processing
  - `dsc_process_signal_by_slice.py`: this script performs similar processing as `dsc_image2concentration` but on the mean signal in a given region of interest and by slice
  - `dsc_dataQuality.py`: this script provides temporal SNR maps and signal-time profile of a given region of interest for a 4D volume
  - `dsc_utils.py`: this function provides the main methods for DSC signal processing and filtering that are called by the scripts above
  - `dsc_pipelines.py`: this function proposes methods calling methods for signal processing from `dsc_utils`
  - `dsc_extract_physio.py`: this function provides methods to extract acquisition times of each repetition in times series acquired on Siemens systems provided a physiolog file.


### dsc_image2concentration

This function takes an Nifti 4D image as input and converts it to the value of variation of the contrast agent concentration value (Delta_C). Below are the available options:
required arguments:
  -i IFNAME             Path to MRI data file.
  -m MASKFNAME          NIFTI volume defining the region of interest.
  -physio PHYSIOLOGFOLDER
                        Path to folder including the physiologs.
  -o OFNAME             Filename for output image and mat file.

optional arguments:
  -h, --help            show this help message and exit
  -param PARAMFILEPATH  Path to file giving specific parameters (injection
                        repetition, dicom path).
  -r2 R2GDINBLOOD       Transverve relaxivity (in s-1.mmol-1.L = s-1.mM-1) of
                        Gadolinium in blood. Default = 3.55 s-1.mmol-1.L [from
                        Proc. Intl. Soc. Mag. Reson. Med. 16 (2008) 1457]

Note that if you use `-r2 1`, the obtained value will correspond to the variation of relaxation rate (R2 or R2* depending on whether acquired data are spin-echo or gradient-echo)

