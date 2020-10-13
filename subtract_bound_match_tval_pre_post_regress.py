import numpy as np
import nibabel as nib

datadir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/parcels/Schaefer300/ttest_results/"

nii_template = nib.load('/jukebox/norman/jamalw/MES/subjects/MES_022817_0/analysis/run1.feat/trans_filtered_func_data.nii')

fn = 'tstats_map_both_runs_srm_v1_all_pure_random_split_merge_original_match_score.nii.gz'
fn_regress = 'tstats_map_both_runs_srm_v1_all_pure_random_split_merge_original_match_score_regress_all_features_whole_brain.nii.gz'

data = nib.load(datadir + fn).get_data()
data_regressed = nib.load(datadir + fn_regress).get_data()
subtracted_data = data - data_regressed
maxval = np.max(subtracted_data)
minval = np.min(subtracted_data)
img = nib.Nifti1Image(subtracted_data,affine=nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img, datadir + 'subtracted_tval_data_all_features.nii.gz') 
