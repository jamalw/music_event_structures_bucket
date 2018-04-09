import numpy as np
import nibabel as nib
from scipy import stats

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_srm/'

nii_template = nib.load('/jukebox/norman/jamalw/MES/subjects/MES_022817_0/analysis/run1.feat/trans_filtered_func_data.nii')

mask = nib.load('/jukebox/norman/jamalw/MES/prototype/link/scripts/mask_nonan.nii.gz').get_data()
mask_reshaped = np.reshape(mask,(91*109*91)) != 0
tmap_final1D = np.zeros((len(mask_reshaped)))
pmap_final1D = np.zeros((len(mask_reshaped)))

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

all_songs3D = np.zeros((91,109,91,len(songs)))
all_songs1D = np.zeros((218477,len(songs)))

for i in range(len(songs)):
    data = nib.load(datadir + songs[i] + '/avg_data/globals_avg_raw_n25.nii.gz').get_data()
    all_songs3D[:,:,:,i] = data
    all_songs1D[:,i] = data[mask != 0]

tmap1D = np.zeros((len(all_songs1D[:,0])))
pmap1D = np.zeros((len(all_songs1D[:,0])))

for j in range(len(all_songs1D[:,0])):
    tmap1D[j],pmap1D[j] = stats.ttest_1samp(all_songs1D[j,:],0,axis=0)[0]

tmap_final1D[mask_reshaped==1] = tmap1D
tmap_final3D = np.reshape(tmap_final1D,(91,109,91))

pmap_final1D[mask_reshaped==1] = pmap1D
pmap_final3D = np.reshape(pmap_final1D,(91,109,91))

# save data
maxval = np.max(tmap_final3D)
minval = np.min(tmap_final3D)
img = nib.Nifti1Image(tmap_final3D, affine=nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + '/tstats_map.nii.gz')

maxval = np.max(pmap_final3D)
minval = np.min(pmap_final3D)
img = nib.Nifti1Image(pmap_final3D, affine=nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + '/pstats_map.nii.gz')
