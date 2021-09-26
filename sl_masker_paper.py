import numpy as np
import nibabel as nib

parcelNum = 100
thr = ['01','05']
roi_name = ['bil_precuneus','bil_mPFC','bil_AG','bil_A1']

datadir = '/jukebox/norman/jamalw/MES/'

parcels = nib.load(datadir + "data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_" + str(parcelNum) + "Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

full_mask_img = nib.load(datadir + 'data/mask_nonan.nii')

def save_nifti(data,affine,savedir):
    minval = np.min(data)
    maxval = np.max(data)
    img = nib.Nifti1Image(data,affine)
    img.header["cal_min"] = minval
    img.header["cal_max"] = maxval
    img.header.set_data_dtype(np.float64)
    nib.save(img, savedir)
 

for r in range(len(roi_name)):
    for t in range(len(thr)):
        # create roi + mask image
        roi = np.zeros_like(full_mask_img.get_data(),dtype=int)

        # load in q-stats mask which can be found in scripts/data/searchlight_output/HMM_searchlight_human_bounds_fit_to_all/ttest_results/masks/
        stats_mask = nib.load(datadir + 'prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_fit_to_all/ttest_results/masks/qstats_map_both_runs_regress_all_tempo_12PC_singles_spect_no_srm_mask_' + thr[t] + '.nii.gz')

        savedir = datadir + "/data/fdr_" + thr[t] + "_human_bounds_both_runs_split_merge_regress_" + roi_name[r] + "_no_srm_bin.nii.gz"

        # get indices where mask and parcels overlap
        if r == 0:
            # bil PCC
            indices = np.where((full_mask_img.get_data() > 0) & (parcels == 39) | (parcels == 91) & (stats_mask.get_data() > 0))
        elif r == 1:
            # bilateral mPFC
            indices = np.where((full_mask_img.get_data() > 0) & (parcels == 40) | (parcels == 92) & (stats_mask.get_data() > 0))
        elif r == 2:
            # bil AG
            indices = np.where((full_mask_img.get_data() > 0) & (parcels == 89) | (parcels == 43) & (stats_mask.get_data() > 0))
        elif r == 3:
            # bilA1
            indices = np.where((full_mask_img.get_data() > 0) & (parcels == 61) | (parcels == 10) & (stats_mask.get_data() > 0))

        # fit match score and pvalue into brain
        roi[indices] = 1
 
        save_nifti(roi, full_mask_img.affine, savedir) 



# bilateral PHC
#indices = np.where((full_mask_img.get_data() > 0) & (parcels == 49) | (parcels == 97) & (stats_mask.get_data() > 0))

# left PHC
#indices = np.where((full_mask_img.get_data() > 0) & (parcels == 49) & (stats_mask.get_data() > 0))

# rA1
#indices = np.where((full_mask_img.get_data() > 0) & (parcels == 61) & (stats_mask.get_data() > 0))

# rAG
#indices = np.where((full_mask_img.get_data() > 0) & (parcels == 89) & (stats_mask.get_data() > 0))

#indices = np.where((full_mask_img.get_data() > 0) & (parcels == idx) & (stats_mask.get_data() > 0))

# left PCC
#indices = np.where((full_mask_img.get_data() > 0) & (parcels == 39) | (parcels == 35) | (parcels == 36) | (parcels == 48) & (stats_mask.get_data() > 0))


