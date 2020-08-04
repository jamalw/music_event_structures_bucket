import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn import plotting
import matplotlib.pyplot as plt
import pandas as pd
import sys
from nilearn.connectome import ConnectivityMeasure
from srm import SRM_V2,SRM_V3

idx = int(sys.argv[1])

n_rois = 200
net = 17
vox_size = 2

atlas = datasets.fetch_atlas_schaefer_2018(n_rois,net,vox_size)
atlas_filename = atlas.maps

atlas_pd = pd.DataFrame(atlas)

hrf = 4

# run 1 times
song_bounds1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

songs1 = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']


# get song start time and end time for run 1
start_idx_run1 = song_bounds1[idx]
end_idx_run1   = song_bounds1[idx+1]

## run 2 times
song_bounds2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])
#
songs2 = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']
#
## get song start time and end time for run 2
song_name = songs1[idx]
start_idx_run2 = song_bounds2[songs2.index(song_name)]
end_idx_run2   = song_bounds2[songs2.index(song_name) + 1]
#
datadir = '/jukebox/norman/jamalw/MES/'
savedir = datadir + 'prototype/link/scripts/data/functional_connectivity/'

mask_img = nib.load(datadir + 'data/mask_nonan.nii')

n_iter = 50
srm_k = 30

parcel_dir = datadir + "prototype/link/scripts/data/searchlight_input/parcels/Schaefer200/"

song_dur = end_idx_run2 - start_idx_run2

parcel_select = np.array(list(range(13,29)) + list(range(75,99)) + list(range(113,131)) + list(range(184,197)))

avg_parcels_run1 = np.zeros((len(parcel_select),song_dur))
avg_parcels_run2 = np.zeros((len(parcel_select),song_dur))
avg_parcels_srm = np.zeros((len(parcel_select),song_dur))

for i in range(len(parcel_select)):
    print("Parcel Num: ", str(parcel_select[i]))

    # initialize list for storing masked data across subjects
    run1 = np.load(parcel_dir + "parcel" + str(parcel_select[i]) + "_run1.npy")
    run2 = np.load(parcel_dir + "parcel" + str(parcel_select[i]) + "_run2.npy")

    # run SRM on masked data
    run1_SRM, run2_SRM = SRM_V2(run1,run2,srm_k,n_iter)

    # get song data from each run
    data_run1 = run1_SRM[:,start_idx_run1:end_idx_run1]
    data_run2 = run2_SRM[:,start_idx_run2:end_idx_run2]

    # average song data between two runs and store srm parcel
    data = (data_run1 + data_run2) / 2
    avg_parcels_srm[i,:] = np.mean(data,axis=0)    
    
    # average over subjects, slice song, then store run 1 parcel
    avg_over_subjs = np.mean(run1,axis=0)
    data_run1 = avg_over_subjs[:,start_idx_run1:end_idx_run1]
    avg_parcels_run1[i,:] = np.mean(data_run1,axis=0)

    # average over subjects, slice song, then store run 2 parcel
    avg_over_subjs = np.mean(run2,axis=0)
    data_run2 = avg_over_subjs[:,start_idx_run2:end_idx_run2]
    avg_parcels_run2[i,:] = np.mean(data_run2,axis=0)

# Set up the connectivity object
correlation_measure = ConnectivityMeasure(kind='correlation')

# Calculate the correlation of each parcel with every other parcel
corr_mat_srm = correlation_measure.fit_transform([avg_parcels_srm.T])[0]
corr_mat_run1 = correlation_measure.fit_transform([avg_parcels_run1.T])[0]
corr_mat_run2 = correlation_measure.fit_transform([avg_parcels_run2.T])[0]

# Remove the diagonal for visualization (guaranteed to be 1.0)
np.fill_diagonal(corr_mat_srm, np.nan)
np.fill_diagonal(corr_mat_run1,np.nan)
np.fill_diagonal(corr_mat_run2,np.nan)

# Plot and save correlation matrix for run 1 and save data
fig = plt.figure(figsize=(11,10))
plt.imshow(corr_mat_srm, interpolation='None', cmap='RdYlBu_r')
plt.yticks(range(len(atlas.labels[parcel_select])), atlas.labels[parcel_select - 1]);
plt.xticks(range(len(atlas.labels[parcel_select])), atlas.labels[parcel_select - 1], rotation=90);
plt.title('Parcellation FC SRM ' + songs1[idx])
plt.colorbar();
plt.tight_layout()
plt.savefig(savedir + 'srm/' + songs1[idx] + '_FC_srm')
np.save(savedir + 'srm/' +  songs1[idx] + '_FC_srm',corr_mat_srm) 
 
# Plot and save correlation matrix for run 1 and save data
fig = plt.figure(figsize=(11,10))
plt.imshow(corr_mat_run1, interpolation='None', cmap='RdYlBu_r')
plt.yticks(range(len(atlas.labels[parcel_select])), atlas.labels[parcel_select - 1]);
plt.xticks(range(len(atlas.labels[parcel_select])), atlas.labels[parcel_select - 1], rotation=90);
plt.title('Parcellation FC Run 1 ' + songs1[idx])
plt.colorbar();
plt.tight_layout()
plt.savefig(savedir + 'run1/' + songs1[idx] + '_FC_run1')
np.save(savedir + 'run1/' +  songs1[idx] + '_FC_run1',corr_mat_run1) 
 

# Plot and save correlation matrix for run 2 and save data
fig = plt.figure(figsize=(11,10))
plt.imshow(corr_mat_run2, interpolation='None', cmap='RdYlBu_r')
plt.yticks(range(len(atlas.labels[parcel_select])), atlas.labels[parcel_select - 1]);
plt.xticks(range(len(atlas.labels[parcel_select])), atlas.labels[parcel_select - 1], rotation=90);
plt.title('Parcellation FC Run 2 ' + songs1[idx])
plt.colorbar();
plt.tight_layout()
plt.savefig(savedir + 'run2/' + songs1[idx] + '_FC_run2')
np.save(savedir + 'run2/' +  songs1[idx] + '_FC_run2',corr_mat_run2) 
    

