import numpy as np
import nibabel as nib

datadir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/"

nii_template = nib.load('/jukebox/norman/jamalw/MES/subjects/MES_022817_0/analysis/run1.feat/trans_filtered_func_data.nii')

songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

for i in range(len(songs)):
    data_run1 = np.load(datadir + 'HMM_searchlight_human_bounds_fit_to_all/' + songs[i] + '/zscores/globals_z_test_run1_split_merge_no_srm.npy')
    data_run2 = np.load(datadir + 'HMM_searchlight_human_bounds_fit_to_all/' + songs[i] + '/zscores/globals_z_test_run2_split_merge_no_srm.npy')
    subtracted_data = data_run2 - data_run1
    maxval = np.max(subtracted_data)
    minval = np.min(subtracted_data)
    img = nib.Nifti1Image(subtracted_data,affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img, datadir + 'bound_match_subtraction/' + songs[i] + '/zscores/z_subtracted_run2_run1_no_srm.nii.gz') 
