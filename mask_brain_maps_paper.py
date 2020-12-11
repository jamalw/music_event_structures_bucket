import numpy as np
import sys
import nibabel as nib
from save_nifti import save_nifti
from nilearn.masking import compute_epi_mask
from nilearn.plotting import plot_epi, show

maskdir = '/jukebox/norman/jamalw/MES/data/'
#datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/parcels/Schaefer300/ttest_results/'
#tfn = 'tstats_map_both_runs_srm_v1_all_pure_random_split_merge_original_match_score.nii.gz'
#qfn = 'qstats_map_both_runs_srm_v1_all_pure_random_split_merge_original_match_score.nii.gz'
datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_fit_to_all/ttest_results/'

tfn = 'tstats_map_both_runs_split_merge_regress_all_features_tempo_12PC_singles.nii.gz'
qfn = 'qstats_map_both_runs_split_merge_regress_all_features_tempo_12PC_singles.nii.gz'

mask_img = nib.load(maskdir + 'mask_nonan.nii')

stat_data = nib.load(datadir + tfn).get_data()
thresh_data = nib.load(datadir + qfn)
save_data = np.zeros_like(mask_img.get_data(),dtype=float)
qthresh = 0.01

#thresh_mask = compute_epi_mask(thresh_data, lower_cutoff=0,upper_cutoff=qthresh)

indices = np.where(thresh_data.get_data() < qthresh)

save_data[indices] = stat_data[indices]

save_nifti(save_data,mask_img.affine,datadir + 'tstats_masked_regressed')



