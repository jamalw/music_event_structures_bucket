import numpy as np
import nibabel as nib
from save_nifti import save_nifti
import scipy.stats as st

parcelNum = 300

datadir = '/jukebox/norman/jamalw/MES/'

parcels = nib.load(datadir + "data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_" + str(parcelNum) + "Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

mask_img = nib.load(datadir + 'data/mask_nonan.nii')

wva_dir = datadir + 'prototype/link/scripts/hmm_K_sweep_paper_results/Schaefer' + str(parcelNum) + '/allROIs/cross_val/'

savedir = datadir + 'prototype/link/scripts/hmm_K_sweep_paper_results/Schaefer' + str(parcelNum) + '/allROIs/cross_val/'

smooth_max_wva_pvals = np.zeros_like(mask_img.get_data(),dtype=float)

for i in range(int(np.max(parcels))):
    print("Parcel Num: ", str(i+1))
    indices = np.where((mask_img.get_data() > 0) & (parcels == i + 1))
    
    data = np.load(wva_dir + 'parcel' + str(i+1) + '_wva_data.npy',allow_pickle=True)
    
    smooth_max_wva_pvals[indices] = st.norm.sf(data.item()['Smooth_Max_WvA'][0])

smooth_fn = savedir + 'allParcels_smooth_max_wva_crossval_pvals'

save_nifti(smooth_max_wva_pvals, mask_img.affine, smooth_fn)

    
    

