import numpy as np
import nibabel as nib
from scipy import stats

nii_template = nib.load('/jukebox/norman/jamalw/MES/subjects/MES_022817_0/analysis/run1.feat/trans_filtered_func_data.nii')
datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

# load,zscore,then store each dataset for each K in a list
k_data = np.zeros((91,109,91,7))
z_data = np.zeros((91,109,91,7))
thresh = np.zeros((91,109,91))
maxval_per_maxK = np.zeros((91,109,91,7)) 
lengths = [10,15,20,25,30,35,40]

for i in range(len(lengths)):
    data = nib.load(datadir + str(lengths[i]) + '_sec_events.nii.gz').get_data()
    z = nib.load(datadir + str(lengths[i]) + '_sec_events_Z.nii.gz').get_data()
    k_data[:,:,:,i] = data
    z_data[:,:,:,i] = z 

max_K = np.argmax(k_data,axis=3)
max_K[np.sum(k_data, axis=3) == 0] = 0       
 
for i in range(91):
    for j in range(109):
        for k in range(91):
            thresh[i,j,k] = z_data[i,j,k,max_K[i,j,k]]

max_K[thresh < 0] = 0

# save final map as nifti
maxval = np.max(max_K)
minval = np.min(max_K)
img = nib.Nifti1Image(max_K,affine = nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img, datadir + 'best_event_length_map.nii.gz')


