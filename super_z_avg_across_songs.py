import numpy as np
import sys
import os
import nibabel as nib
import glob
import scipy.stats as st

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

nii_template = nib.load('/jukebox/norman/jamalw/MES/subjects/MES_022817_0/analysis/run1.feat/trans_filtered_func_data.nii')

# Collect searchlight files
k = ['20']

for i in range(len(k)):
    fn = datadir + 'avg_z_k'+k[i]+'_across_songs.nii.gz'
    global_outputs_all = nib.load(fn).get_data()

    # Reshape data
    z_scores_reshaped = np.nan_to_num(np.reshape(global_outputs_all,(91*109*91)))

    # Mask data with nonzeros
    mask = z_scores_reshaped != 0
    z_scores_reshaped[mask] = st.zscore(z_scores_reshaped[mask])
    #z_scores_reshaped[mask] = -np.log(st.norm.sf(z_scores_reshaped[mask]))

    # Reshape data back to original shape
    #neg_log_p_values = np.reshape(z_scores_reshaped,(91,109,91))
    super_z_avg_z = np.reshape(z_scores_reshaped,(91,109,91))
    # Plot and save searchlight results
    maxval = np.max(super_z_avg_z)
    minval = np.min(super_z_avg_z)
    img = nib.Nifti1Image(super_z_avg_z, affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,datadir + '/avg_superz_k'+k[i]+'_across_songs.nii.gz')


