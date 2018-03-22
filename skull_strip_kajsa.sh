#!/bin/bash

module load fsl/5.0.9
module load afni/openmp.latest

# Fit structural to standard space
fslreorient2std anat.nii anat_2std.nii.gz

# Remove everything below brain  
robustfov -v -b 140 -i anat_2std.nii.gz -r anat_2std_no_neck.nii.gz
# if cerebellum gets cropped, increase the value for -b (size of brain in z dimension; default is 170 mm but that's pretty large) I crop it just below the cerebellum

# Help improve problems resulting from non-uniform intensity
3dUnifize -prefix anat_2std_no_neck_unifize.nii.gz -input anat_2std_no_neck.nii.gz -GM

# Skullstrip
@NoisySkullStrip -input anat_2std_no_neck_unifize.nii.gz -3dSkullStrip_opts -use_skull
# If you get superior uncovered gyri you can try -blur_fwhm
# -use_skull usually is an improvement but not always

# Convert back to NIFTI from AFNI's output files 
3dAFNItoNIFTI -prefix anat_brain.nii.gz anat_2std_no_neck_unifize.nii.gz.ns+orig; rm -f *orig* __*
