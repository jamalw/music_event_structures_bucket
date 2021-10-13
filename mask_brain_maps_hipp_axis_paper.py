import numpy as np
import nibabel as nib
from save_nifti import save_nifti

maskdir = '/jukebox/norman/jamalw/MES/data/'

# load roi data
stats_mask = nib.load(maskdir + 'fdr_01_human_bounds_both_runs_split_merge_bil_hipp_no_srm_bin.nii.gz')

# load rectangular 3D mask
rect_mask = nib.load(maskdir + 'hipp_point_posterior.nii.gz').get_data()

# create brain map for putting new masked voxels into
save_data = np.zeros_like(stats_mask.get_data(),dtype=float)

# find voxels at intersection of rectangular mask and full hippocampal mask
indices = np.where((stats_mask.get_data() > 0) & (rect_mask > 0))

# set all voxels at intersection to 1
save_data[indices] = 1

# save mask
save_nifti(save_data,stats_mask.affine,maskdir + 'fdr_01_human_bounds_both_runs_split_merge_bil_hipp_posterior_no_srm_bin.nii.gz')




