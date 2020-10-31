import numpy as np
import nibabel as nib

parcelNum = 100
thr = '01'
roi_name = 'left_AG'
idx = 43

def save_nifti(data,affine,savedir):
    minval = np.min(data)
    maxval = np.max(data)
    img = nib.Nifti1Image(data,affine)
    img.header["cal_min"] = minval
    img.header["cal_max"] = maxval
    img.header.set_data_dtype(np.float64)
    nib.save(img, savedir)
 
datadir = '/jukebox/norman/jamalw/MES/'
savedir = datadir + "/data/fdr_" + thr + "_human_bounds_" + roi_name + "_bin.nii.gz"

full_mask_img = nib.load(datadir + 'data/mask_nonan.nii')

stats_mask = nib.load(datadir + 'prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_fit_to_all/ttest_results/masks/qstats_no_motion_' + thr + '_mask.nii.gz')

parcels = nib.load(datadir + "data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_" + str(parcelNum) + "Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

# create brain-like object to store data into
roi = np.zeros_like(full_mask_img.get_data(),dtype=int) 

# get indices where mask and parcels overlap
#indices = np.where((full_mask_img.get_data() > 0) & (parcels == 39) & (parcels == 35) & (parcels == 16) & (parcels == 36) & (parcels == 24) & (parcels == 37) & (parcels == 48) & (stats_mask.get_data() > 0))

indices = np.where((full_mask_img.get_data() > 0) & (parcels == idx) & (stats_mask.get_data() > 0))
 
# fit match score and pvalue into brain
roi[indices] = 1
 
save_nifti(roi, full_mask_img.affine, savedir) 

