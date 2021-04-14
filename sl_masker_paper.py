import numpy as np
import nibabel as nib

parcelNum = 100
thr = '01'
roi_name = 'bil_mPFC'

def save_nifti(data,affine,savedir):
    minval = np.min(data)
    maxval = np.max(data)
    img = nib.Nifti1Image(data,affine)
    img.header["cal_min"] = minval
    img.header["cal_max"] = maxval
    img.header.set_data_dtype(np.float64)
    nib.save(img, savedir)
 
datadir = '/jukebox/norman/jamalw/MES/'
savedir = datadir + "/data/fdr_" + thr + "_human_bounds_split_merge_" + roi_name + "_no_srm_bin.nii.gz"

full_mask_img = nib.load(datadir + 'data/mask_nonan.nii')

# load in q-stats mask which is made with "fslmaths input_file -uthr thr -bin output_file"
stats_mask = nib.load(datadir + 'prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_fit_to_all/ttest_results/masks/qstats_split_merge_' + thr + 'no_srm_mask.nii.gz')

parcels = nib.load(datadir + "data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_" + str(parcelNum) + "Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

# create brain-like object to store data into
roi = np.zeros_like(full_mask_img.get_data(),dtype=int) 

# get indices where mask and parcels overlap
#
#left PCC
#indices = np.where((full_mask_img.get_data() > 0) & (parcels == 39) | (parcels == 35) | (parcels == 36) | (parcels == 48) & (stats_mask.get_data() > 0))

# bilateral mPFC
indices = np.where((full_mask_img.get_data() > 0) & (parcels == 40) | (parcels == 92) & (stats_mask.get_data() > 0))

# bilateral PHC
#indices = np.where((full_mask_img.get_data() > 0) & (parcels == 49) | (parcels == 97) & (stats_mask.get_data() > 0))

#indices = np.where((full_mask_img.get_data() > 0) & (parcels == idx) & (stats_mask.get_data() > 0))
 
# fit match score and pvalue into brain
roi[indices] = 1
 
save_nifti(roi, full_mask_img.affine, savedir) 

