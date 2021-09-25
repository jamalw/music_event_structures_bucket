import numpy as np
import nibabel as nib
from save_nifti import save_nifti

parcelNum = 300

datadir = '/jukebox/norman/jamalw/MES/'

# All parcels
parcels = nib.load(datadir + "data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_" + str(parcelNum) + "Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

#Schaefer 300 DMNa only
parcels_idx = np.concatenate([np.arange(111,122),np.arange(271,283)])

mask_img = nib.load(datadir + 'data/mask_nonan.nii')

wva_dir = datadir + 'prototype/link/scripts/hmm_K_sweep_paper_results/Schaefer' + str(parcelNum) + '/allROIs/cross_val/'

savedir = datadir + 'prototype/link/scripts/hmm_K_sweep_paper_results/Schaefer' + str(parcelNum) + '/allROIs/cross_val/'

smooth_max_wva = np.zeros_like(mask_img.get_data(),dtype=float)
raw_max_wva = np.zeros_like(mask_img.get_data(),dtype=float)
pref_event_length_sec = np.zeros_like(mask_img.get_data(),dtype=int)

#for i in range(int(np.max(parcels))):
for i in range(len(parcels_idx)):
    print("Parcel Num: ", str(i+1))
    #indices = np.where((mask_img.get_data() > 0) & (parcels == i + 1))
    indices = np.where((mask_img.get_data() > 0) & (parcels == parcels_idx[i]))

    data = np.load(wva_dir + 'parcel' + str(i+1) + '_wva_data.npy',allow_pickle=True)
    
    smooth_max_wva[indices] = data.item()['Smooth_Max_WvA'][0]
    #raw_max_wva[indices] = data.item()['Raw_Max_WvA'][0]
    pref_event_length_sec[indices] = data.item()['Pref_Event_Length_Sec'][0]


#smooth_fn = savedir + 'allParcels_smooth_max_wva'
#raw_fn = savedir + 'allParcels_raw_max_wva'
#pref_event_length_fn = savedir + 'allParcels_pref_event_length_sec'

smooth_fn = savedir + 'DMNa_smooth_max_wva'
#raw_fn = savedir + 'DMNa_raw_max_wva'
pref_event_length_fn = savedir + 'DMNa_pref_event_length_sec'

save_nifti(smooth_max_wva, mask_img.affine, smooth_fn)
#save_nifti(raw_max_wva, mask_img.affine, raw_fn)
save_nifti(pref_event_length_sec, mask_img.affine, pref_event_length_fn) 

    
    

