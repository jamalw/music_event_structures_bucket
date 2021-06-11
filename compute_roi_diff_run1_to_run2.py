import numpy as np
import nibabel as nib
from scipy import stats
import glob

parcelNum = 100

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_fit_to_all/'

parcels = nib.load("/jukebox/norman/jamalw/MES/data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_" + str(parcelNum) + "Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

mask = nib.load('/jukebox/norman/jamalw/MES/data/mask_nonan.nii')

# Schaefer 100 bil mPFC
parcel_idx = [40,92]

brain_mask = np.zeros_like(mask.get_data())

for p in range(len(parcel_idx)):
    brain_mask[(parcels == parcel_idx[p]) & (mask.get_data() > 0)] = 1

songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

voxels_by_songs = np.zeros((brain_mask[brain_mask == 1].shape[0],len(songs)))

# create matrix which contains match scores for each song at every voxel
for i in range(len(songs)):
    # load data for each run separately
    data_run1 = np.load(datadir + songs[i] + '/zscores/globals_z_test_run1_split_merge_no_srm.npy')
    data_run2 = np.load(datadir + songs[i] + '/zscores/globals_z_test_run2_split_merge_no_srm.npy')
    
    # extract voxels from data for each run separately using brain mask
    masked_data_run1 = data_run1[brain_mask == 1]
    masked_data_run2 = data_run2[brain_mask == 1]

   

 






    
