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
tfn = 'tstats_map_run2_split_merge_no_srm.nii.gz'
qfn = 'qstats_map_run2_split_merge_no_srm.nii.gz'

mask_img = nib.load(maskdir + 'mask_nonan.nii')

stat_data = nib.load(datadir + tfn).get_data()
thresh_data = nib.load(datadir + qfn).get_data()
save_data = np.zeros_like(mask_img.get_data(),dtype=float)
qthresh = 0.05
qthresh_fn = str(qthresh).split('.')[1]

indices = np.where((thresh_data < qthresh) & (thresh_data > 0))

save_data[indices] = stat_data[indices]

save_data[save_data > 0] = 1

save_nifti(save_data,mask_img.affine,datadir + 'masks/' + str(qfn).split('.')[0] + '_mask_' + str(qthresh_fn) + '.nii.gz')




