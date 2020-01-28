import numpy as np
import nibabel as nib
from scipy import stats
import glob

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_bound_match_shuffle_event_lengths/'

nii_template = nib.load('/jukebox/norman/jamalw/MES/subjects/MES_022817_0/analysis/run1.feat/trans_filtered_func_data.nii')

mask = nib.load('/jukebox/norman/jamalw/MES/prototype/link/scripts/mask_nonan.nii.gz').get_data()
mask_reshaped = np.reshape(mask,(91*109*91))
tmap_final1D = np.zeros((len(mask_reshaped)))
pmap_final1D = np.zeros((len(mask_reshaped)))
qmap_final1D = np.zeros((len(mask_reshaped)))

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

all_songs3D = np.zeros((91,109,91,len(songs)))
all_songs1D = np.zeros((218477,len(songs)))

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

for i in range(len(songs)):
    fn_z = glob.glob(datadir + songs[i] + '/zscores/globals_avg_both_z_runs_n25_srmk_30.nii.gz')
    data = nib.load(fn_z[0]).get_data()
    all_songs3D[:,:,:,i] = data
    all_songs1D[:,i] = data[mask != 0]


inf_idx = np.where(np.isinf(zmap_final3D))

for i in range(len(inf_idx)):
    zmap_final_3D[inf_idx[0,i],inf_idx[1,i],inf_idx[2,i]] = np.nan    


zmap_final3D = np.nanmean(all_songs3D,axis=3)

tmap1D = np.zeros((len(all_songs1D[:,0])))
pmap1D = np.zeros((len(all_songs1D[:,0])))
qmap1D = np.zeros((len(all_songs1D[:,0])))

for j in range(len(all_songs1D[:,0])):
	tmap1D[j],pmap1D[j] = stats.ttest_1samp(all_songs1D[j,:][~np.isnan(all_songs1D[j,:])],0,axis=0)
	if all_songs1D[j,:].mean() > 0:
		pmap1D[j] = pmap1D[j]/2
	else:
		pmap1D[j] = 1-pmap1D[j]/2

qmap1D = FDR_p(pmap1D[~np.isnan(pmap1D)])

#return nans and real values to qmap
qmap1D_nans = np.zeros((len(all_songs1D[:,0])))
idx_nan = np.where(np.isnan(pmap1D))[0]
idx_non_nan = np.where(~np.isnan(pmap1D))[0]

for i in range(len(idx_non_nan)):
    qmap1D_nans[idx_non_nan[i]] = qmap1D[i]

for i in range(len(idx_nan)):
    qmap1D_nans[idx_nan[i]] = np.nan
    
# Fit data back into whole brain
tmap_final1D[mask_reshaped==1] = tmap1D
tmap_final3D = np.reshape(tmap_final1D,(91,109,91))

pmap_final1D[mask_reshaped==1] = pmap1D
pmap_final3D = np.reshape(pmap_final1D,(91,109,91))

qmap_final1D[mask_reshaped==1] = qmap1D_nans
qmap_final3D = np.reshape(qmap_final1D,(91,109,91))

# save data
maxval = np.max(tmap_final3D)
minval = np.min(tmap_final3D)
img = nib.Nifti1Image(tmap_final3D, affine=nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'ttest_results/tstats_map_both_runs.nii.gz')

maxval = np.max(pmap_final3D)
minval = np.min(pmap_final3D)
img = nib.Nifti1Image(pmap_final3D, affine=nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'ttest_results/pstats_map_both_runs.nii.gz')

maxval = np.max(qmap1D)
minval = np.min(qmap1D)
img = nib.Nifti1Image(qmap_final3D, affine=nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'ttest_results/qstats_map_both_runs.nii.gz')

maxval = np.max(zmap_final3D)
minval = np.min(zmap_final3D)
img = nib.Nifti1Image(zmap_final3D, affine=nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'ttest_results/zstats_map_both_runs.nii.gz')
