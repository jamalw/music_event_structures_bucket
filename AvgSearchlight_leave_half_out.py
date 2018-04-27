import numpy as np
import sys
import os
import nibabel as nib
import glob
import scipy.stats as st

datadir = '/scratch/jamalw/HMM_searchlight_K_sweep_srm/'
nii_template = nib.load('/jukebox/norman/jamalw/MES/subjects/MES_022817_0/analysis/run1.feat/trans_filtered_func_data.nii')

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

# Collect searchlight files
k = ['3']

for i in range(len(k)):
    global_outputs_z = np.zeros((91,109,91))
    global_outputs_raw = np.zeros((91,109,91))
    for s in range(len(songs)):
        fn_z = glob.glob(datadir + songs[s] + '/zscores/' + '*' + k[i]+ '.npy')
        fn_raw = glob.glob(datadir + songs[s] + '/raw/' + '*' + k[i] + '.npy')
        data_z = np.load(fn_z[0])
        data_raw = np.load(fn_raw[0])
        
        # Plot and save searchlight results
        maxval = np.max(data_z)
        minval = np.min(data_z)
        img = nib.Nifti1Image(data_z, affine=nii_template.affine)
        img.header['cal_min'] = minval
        img.header['cal_max'] = maxval
        nib.save(img,datadir + songs[s] + '/avg_data/avg_z_n25_k'+k[i]+'.nii.gz')

        # Plot and save searchlight results
        maxval = np.max(data_raw)
        minval = np.min(data_raw)
        img = nib.Nifti1Image(data_raw, affine=nii_template.affine)
        img.header['cal_min'] = minval
        img.header['cal_max'] = maxval
        nib.save(img,datadir + songs[s] + '/avg_data/avg_raw_n25_k'+k[i]+'.nii.gz') 
        
        global_outputs_z[:,:,:] += data_z/(len(songs)) 
        global_outputs_raw[:,:,:] += data_raw/(len(songs))    

    # Save average results 
    np.save(datadir + 'avg_data/avg_z_n25_k'+k[i],global_outputs_z)

    np.save(datadir + 'avg_data/avg_raw_n25_k' + k[i],global_outputs_raw)

    # Plot and save searchlight results
    maxval = np.max(global_outputs_z)
    minval = np.min(global_outputs_z)
    img = nib.Nifti1Image(global_outputs_z, affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,datadir + 'avg_data/avg_z_n25_k'+k[i]+'.nii.gz')

    # Plot and save searchlight results
    maxval = np.max(global_outputs_raw)
    minval = np.min(global_outputs_raw)
    img = nib.Nifti1Image(global_outputs_raw, affine=nii_template.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img,datadir + 'avg_data/avg_raw_n25_k'+k[i]+'.nii.gz')


