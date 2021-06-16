import numpy as np
import nibabel as nib
from scipy import stats
import glob
import matplotlib.pyplot as plt
from save_nifti import save_nifti

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


datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_fit_to_all/'
savedir = datadir + 'ttest_results/'

mask = nib.load('/jukebox/norman/jamalw/MES/data/mask_nonan.nii')
mask_reshape = np.reshape(mask.get_data(),(91*109*91))

# create coords matrix
x,y,z = np.mgrid[[slice(dm) for dm in tuple((91,109,91))]]
x = np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2]))
y = np.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2]))
z = np.reshape(z,(z.shape[0]*z.shape[1]*z.shape[2]))
coords = np.vstack((x,y,z)).T
coords_mask = coords[mask_reshape>0]

tmap1D = np.zeros((coords_mask.shape[0]))
pmap1D = np.zeros((coords_mask.shape[0]))
qmap1D = np.zeros((coords_mask.shape[0]))

tmap_final1D = np.zeros((len(mask_reshape)))
pmap_final1D = np.zeros((len(mask_reshape)))
qmap_final1D = np.zeros((len(mask_reshape)))

tmap3D = np.zeros_like(mask.get_data())
pmap3D = np.zeros_like(mask.get_data())
qmap3D = np.zeros_like(mask.get_data())

songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

all_songs1D_run1 = np.zeros((coords_mask.shape[0],len(songs)))
all_songs1D_run2 = np.zeros((coords_mask.shape[0],len(songs)))

for s in range(len(songs)):
    data_run1 = np.load(datadir + songs[s] + '/zscores/globals_z_test_run1_split_merge_no_srm.npy')
    data_run2 = np.load(datadir + songs[s] + '/zscores/globals_z_test_run2_split_merge_no_srm.npy')
    all_songs1D_run1[:,s] = data_run1[mask.get_data() != 0]
    all_songs1D_run2[:,s] = data_run2[mask.get_data() != 0]

# compute ttest between run 1 and run 2 song-specific match scores within a given voxel
for c in range(coords_mask.shape[0]):
    # compute ttest between run 1 and run 2 song-specific matches
    tmap1D[c], pmap1D[c]= stats.ttest_rel(all_songs1D_run2[c,:], all_songs1D_run1[c,:])    


# do FDR
qmap1D = FDR_p(pmap1D)
 
# Fit data back into whole brain, then save
tmap_final1D[mask_reshape==1] = tmap1D
tmap3D = np.reshape(tmap_final1D,(91,109,91))
save_nifti(tmap3D,mask.affine,savedir + 'tstats_map_run2_minus_run1_no_srm.nii.gz')

pmap_final1D[mask_reshape==1] = pmap1D
pmap3D = np.reshape(pmap_final1D,(91,109,91))
save_nifti(pmap3D,mask.affine,savedir + 'pstats_map_run2_minus_run1_no_srm.nii.gz')

qmap_final1D[mask_reshape==1] = qmap1D
qmap3D = np.reshape(qmap_final1D,(91,109,91))
save_nifti(qmap3D,mask.affine,savedir + 'qstats_map_run2_minus_run1_no_srm.nii.gz')



    
