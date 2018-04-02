import numpy as np
import glob
import nibabel as nib

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_srm/'

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

nii_template = nib.load('/jukebox/norman/jamalw/MES/subjects/MES_022817_0/analysis/run1.feat/trans_filtered_func_data.nii')

for i in range(len(songs)):
    fn_raw = glob.glob(datadir + songs[i] + '/raw/*.npy')
    fn_z   = glob.glob(datadir + songs[i] + '/zscores/*.npy')
    avg_raw = np.zeros((91,109,91))
    avg_z   = np.zeros((91,109,91))
    for j in range(len(fn_raw)):
        subj_raw = np.load(fn_raw[j])
        subj_z   = np.load(fn_z[j])
        avg_raw += subj_raw/len(fn_raw)
        avg_z   += subj_z/len(fn_z)
    # Save raw averages
    maxval = np.max(avg_raw)
    minval = np.min(avg_raw)
    img = nib.Nifti1Image(avg_raw, affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,datadir + songs[i] + '/avg_data/globals_avg_raw_n25.nii.gz')  
    # Save zscore averages
    maxval = np.max(avg_z)
    minval = np.min(avg_z)
    img = nib.Nifti1Image(avg_z, affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,datadir + songs[i] + '/avg_data/globals_avg_z_n25.nii.gz')  
     
