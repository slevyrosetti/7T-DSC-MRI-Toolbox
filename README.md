# :mag_right: :bulb: 7T-DSC-MRI-Toolbox :flashlight: :wrench: </font>

<p align="center">
This Python-based toolbox provides functions and pipeline to process Dynamic Susceptibility Contrast MRI data acquired at 7T. In particular, functions to extract acquisition times from physiologs (Siemens format) for cardiac-gated acquisitions, correct signal by effective TR and discard inconsistent TRs as well as to filter breathing frequencies in the signal based on the signal measured by any respiratory device in a repiration physiolog (Siemens format) are available.</p>

We thank you for choosing our toolbox! :heart: According to the Apache license 2.0, please cite the following reference:
> **Lévy S, Roche P-H, Callot V. Dynamic Susceptibility Contrast imaging at 7T for spinal cord perfusion mapping in Cervical Spondylotic Myelopathy patients, In: *Proc. Intl. Soc. Mag. Reson. Med. 28*. 2019;3195.**

# Code description

This toolbox so far provides the following functions:
  - `dsc_image2concentration.py`: fit IVIM biexponential signal representation to NIFTI data according to specified fitting approach
  - `ivim_view_fits.py`: display an IVIM parameter map and enable user to inspect fitting by clicking on any voxel and display corresponding fit plot
  - `ivim_simu_compute_error_nonoise.py`: compute error of a given fitting approach according to true IVIM values
  - `ivim_simu_plot_error_nonoise.py`: plot results from previous tool
  - `ivim_simu_compute_error_noise.py`: compute error of a given fitting approach according to true IVIM values for a given SNR (Monte Carlo simulations)
  - `ivim_simu_plot_error_noise.py`: plot results from previous tool
  - `ivim_simu_compute_required_snr.py`: compute required SNR to estimate parameters within 10% error margins for a given fitting approach and according to true IVIM values
  - `ivim_simu_plot_required_snr.py`: plot results from previous tool
  - `ivim_toolbox.py`: launch the graphical user interface

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

