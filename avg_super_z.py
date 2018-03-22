import numpy as np
import nibabel as nib

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

k = '12'

avg_data = np.zeros((91,109,91))

for i in range(len(songs)):
    datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[i] + '/avg_data/'
    fn = datadir + 'globals_avg_n25_k' + k + '_neglog_super_z.nii.gz'
    subj_data = nib.load(fn).get_data()
    avg_data[:,:,:] += subj_data/(len(songs))

 
maxval = np.max(avg_data[~np.isnan(avg_data)])
minval = np.min(avg_data[~np.isnan(avg_data)])
img = nib.Nifti1Image(avg_data, np.eye(4))
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,'avg_super_z_k' + k +'.nii.gz')
