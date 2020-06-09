import numpy as np
import nibabel as nib
import os
import time
import subprocess as sp
import scipy.stats as st

datadir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_fit_to_all/ttest_results_test/"

mask = nib.load('/jukebox/norman/jamalw/MES/data/mask_nonan.nii.gz')

data = np.load(datadir + "tstats_map_both_runs_w_perms.npy")

final_TFCE = np.zeros_like(data)

def save_nii(data, save_name):
    maxval = np.max(data)
    minval = np.min(data)
    img = nib.Nifti1Image(data, affine=mask.affine)
    img.header['cal_min'] = minval
    img.header['cal_max'] = maxval
    nib.save(img, save_name)


for i in range(data.shape[-1]):
    print('Running file {}'.format(i))
    permData = data[:,:,:,i]
    permData = np.abs(permData)

    save_name = datadir + 'TFCE_temp_{}.nii.gz'.format(i)
    print('\tSaving first temp file')
    save_nii(permData, save_name)
        
    input_name = save_name
    output_name = datadir + 'TFCE_temp_{}_processed.nii.gz'.format(i)
    print('\tRunning fslmaths command')
    sp.call(["fslmaths", input_name, "-tfce", "2", "0.5", "6", output_name])
    while not os.path.exists(output_name):
        pass
    
    print('\tLoading in result')
    nii_data = nib.load(output_name).get_data()
    final_TFCE[..., i] = nii_data

    print('\tClearing temp files')
    os.remove(input_name)
    os.remove(output_name)


msk_indices = np.where(mask.get_data()>0)

zTFCE = (final_TFCE[:,:,:,0] - np.mean(final_TFCE[:,:,:,1:],axis=3))/np.std(final_TFCE[:,:,:,1:])

final_pmap = np.zeros_like(zTFCE)

masked_pData = st.norm.sf(zTFCE[msk_indices])

final_pmap[msk_indices] = masked_pData
zfn = datadir + "zTFCE"
pfn = datadir + "pTFCE"
save_nii(zTFCE,zfn)
save_nii(final_pmap,pfn)
