import numpy as np
import nibabel as nib
from scipy import stats
import glob
import matplotlib.pyplot as plt
import save_nifti

parcelNum = 300

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/parcels/Schaefer' + str(parcelNum) + '/'

parcels = nib.load("/jukebox/norman/jamalw/MES/data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_" + str(parcelNum) + "Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

mask = nib.load('/jukebox/norman/jamalw/MES/data/mask_nonan.nii')

# Schaefer 200 bil mPFC
#parcel_idx = [80,187]

#Schaefer 300 DMN only
parcel_idx = np.concatenate([np.arange(111,146),np.arange(271,294)])

x = []
y = []
z = []

x_single = []
y_single = []
z_single = []

for i in range(len(parcel_idx)):
    mask_x = np.where((mask.get_data() > 0) & (parcels == parcel_idx[i]))[0]
    mask_y = np.where((mask.get_data() > 0) & (parcels == parcel_idx[i]))[1]
    mask_z = np.where((mask.get_data() > 0) & (parcels == parcel_idx[i]))[2]
    x.append(mask_x)
    y.append(mask_y)
    z.append(mask_z)
    # take first set of coordinates from mask for each parcel 
    x_single.append(mask_x[0])
    y_single.append(mask_y[0])
    z_single.append(mask_z[0])

x_stack = np.hstack(x)
y_stack = np.hstack(y)
z_stack = np.hstack(z)

indices = np.array((x_stack,y_stack,z_stack))

x_single_stack = np.hstack(x_single)
y_single_stack = np.hstack(y_single)
z_single_stack = np.hstack(z_single)

single_indices = np.array((x_single_stack,y_single_stack,z_single_stack))

mask_reshaped = np.reshape(mask.get_data(),(91*109*91))
tmap_final3D = np.zeros_like(mask.get_data(),dtype=float)
pmap_final3D = np.zeros_like(mask.get_data(),dtype=float)
qmap_final3D = np.zeros_like(mask.get_data(),dtype=float)

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


songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

all_songs1D = np.zeros((len(single_indices[0]),len(songs)))

for i in range(len(songs)):
    # load data for each run separately
    data_run1 = nib.load(datadir + songs[i] + '/zscores_srm_v1_test_run1_pure_random_split_merge_no_srm.nii').get_data()
    data_run2 = nib.load(datadir + songs[i] + '/zscores_srm_v1_test_run2_pure_random_split_merge_no_srm.nii').get_data()
    all_songs1D[:,i] = data_run2[single_indices[0],single_indices[1],single_indices[2]] - data_run1[single_indices[0],single_indices[1],single_indices[2]]
    
# calculate tstats against zero for difference between match scores for 16 songs for run 2 minus run1 for each voxel separately and then do FDR
tmap1D = np.zeros((len(all_songs1D[:,0])))
pmap1D = np.zeros((len(all_songs1D[:,0])))
qmap1D = np.zeros((len(all_songs1D[:,0])))

for j in range(len(all_songs1D[:,0])):
        tmap1D[j],pmap1D[j] = stats.ttest_1samp(all_songs1D[j,:],0,axis=0)
        if all_songs1D[j,:].mean() > 0:
                pmap1D[j] = pmap1D[j]/2
        else:
                pmap1D[j] = 1-pmap1D[j]/2


qmap1D = FDR_p(pmap1D)

# Fit data back into whole brain
for i in range(len(parcel_idx)):
    tmap_final3D[parcels==parcel_idx[i]] = tmap1D[i]
    pmap_final3D[parcels==parcel_idx[i]] = pmap1D[i]
    qmap_final3D[parcels==parcel_idx[i]] = qmap1D[i]

# save data
maxval = np.max(tmap_final3D)
minval = np.min(tmap_final3D)
img = nib.Nifti1Image(tmap_final3D, affine=mask.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'ttest_results/tstats_map_run2_minus_run1_pure_random_split_merge_original_match_score_no_srm_DMN.nii.gz')

maxval = np.max(pmap_final3D)
minval = np.min(pmap_final3D)
img = nib.Nifti1Image(pmap_final3D, affine=mask.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'ttest_results/pstats_map_run2_minus_run1_pure_random_split_merge_original_match_score_no_srm_DMN.nii.gz')

maxval = np.max(qmap_final3D)
minval = np.min(qmap_final3D)
img = nib.Nifti1Image(qmap_final3D, affine=mask.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'ttest_results/qstats_map_run2_minus_run1_pure_random_split_merge_original_match_score_no_srm_DMN.nii.gz')

 






    
