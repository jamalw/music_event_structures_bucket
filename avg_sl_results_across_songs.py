import numpy as np
import nibabel as nib

datadir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/music_features/"
nii_template = nib.load('/jukebox/norman/jamalw/MES/subjects/MES_022817_0/analysis/run1.feat/trans_filtered_func_data.nii')

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

avg_data = np.zeros((91,109,91))

for s in range(len(songs)):
    data = np.load(datadir + songs[s] + '/chroma/zscores/globals_z_srm_k30_fit_run2.npy')
    avg_data[:,:,:] += data/len(songs)
    

# Save all raw averages
maxval = np.max(avg_data)
minval = np.min(avg_data)
img = nib.Nifti1Image(avg_data, affine=nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'avg_chroma_run2_across_songs.nii.gz')

