import numpy as np
import nibabel as nib
from scipy import stats
import glob
import matplotlib.pyplot as plt
import save_nifti

def FDR_p(pvals):
    # Port of AFNI mri_fdrize.c
    assert np.all(pvals>=0) and np.all(pvals<=1)
    pvals[pvals < np.finfo(np.float_).eps] = np.finfo(np.float_).eps
    pvals[pvals == 1] = 1-np.finfo(np.float_).eps
    n = pvals.shape[0]

    qvals = np.zeros((n))
    sorted_ind = np.argsort(pvals)
    sorted_pvals = pvals[sorted_ind]
    qmin = 1.0
    for i in range(n-1,-1,-1):
        qval = (n * sorted_pvals[i])/(i+1)
        if qval > qmin:
            qval = qmin
        else:
            qmin = qval
        qvals[sorted_ind[i]] = qval

    # Estimate number of true positives m1 and adjust q
    if n >= 233:
        phist = np.histogram(pvals, bins=20, range=(0, 1))[0]
        sorted_phist = np.sort(phist[3:19])
        if np.sum(sorted_phist) >= 160:
            median4 = n - 20*np.dot(np.array([1, 2, 2, 1]), sorted_phist[6:10])/6
            median6 = n - 20*np.dot(np.array([1, 2, 2, 2, 2, 1]), sorted_phist[5:11])/10
            m1 = min(median4, median6)

            qfac = (n - m1)/n
            if qfac < 0.5:
                qfac = 0.25 + qfac**2
            qvals *= qfac

    return qvals


parcelNum = 200

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_fit_to_all/'

parcels = nib.load("/jukebox/norman/jamalw/MES/data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_" + str(parcelNum) + "Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

mask = nib.load('/jukebox/norman/jamalw/MES/data/mask_nonan.nii')

# Schaefer 200 bil mPFC
parcel_idx = [80,187]

brain_mask = np.zeros_like(mask.get_data())

for p in range(len(parcel_idx)):
    brain_mask[(parcels == parcel_idx[p]) & (mask.get_data() > 0)] = 1

# save mask (optional)
save_nifti.save_nifti(brain_mask,mask.affine,'plots/17Network_200_Default_mPFC1.nii.gz')

songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

# initialize matrix which will store match scores for each song at every voxel
voxels_by_songs_run1 = np.zeros((brain_mask[brain_mask == 1].shape[0],len(songs)))
voxels_by_songs_run2 = np.zeros((brain_mask[brain_mask == 1].shape[0],len(songs))) 

for i in range(len(songs)):
    # load data for each run separately
    data_run1 = np.load(datadir + songs[i] + '/zscores/globals_z_test_run1_split_merge_no_srm.npy')
    data_run2 = np.load(datadir + songs[i] + '/zscores/globals_z_test_run2_split_merge_no_srm.npy')
    
    # extract voxels from data for each run separately using brain mask
    masked_data_run1 = data_run1[brain_mask == 1]
    masked_data_run2 = data_run2[brain_mask == 1]

    # store match scores across all voxels for song n
    voxels_by_songs_run1[:,i] = masked_data_run1
    voxels_by_songs_run2[:,i] = masked_data_run2

# calculate song-specific match score differences between run 1 and run 2
run1_vs_run2_diffs = voxels_by_songs_run2 - voxels_by_songs_run1

# calculate tstats against zero for difference between match scores for 16 songs for run 2 minus run1 for each voxel separately and then do FDR
tmap1D = np.zeros((run1_vs_run2_diffs.shape[0]))
pmap1D = np.zeros((run1_vs_run2_diffs.shape[0]))
qmap1D = np.zeros((run1_vs_run2_diffs.shape[0]))

for j in range(run1_vs_run2_diffs.shape[0]):
        tmap1D[j],pmap1D[j] = stats.ttest_1samp(run1_vs_run2_diffs[j,:],0,axis=0)
        if run1_vs_run2_diffs[j,:].mean() > 0:
                pmap1D[j] = pmap1D[j]/2
        else:
                pmap1D[j] = 1-pmap1D[j]/2

qmap1D = FDR_p(pmap1D)
 

 






    
