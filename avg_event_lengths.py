import numpy as np
import nibabel as nib
import sys

event_length = int(sys.argv[1])
nii_template = nib.load('/jukebox/norman/jamalw/MES/subjects/MES_022817_0/analysis/run1.feat/trans_filtered_func_data.nii')

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

#songs = np.array(['St_Pauls_Suite','I_Love_Music','Moonlight_Sonata','Change_of_the_Guard','Waltz_of_Flowers','The_Bird','Island','Allegro_Moderato','Finlandia','Early_Summer','Capriccio_Espagnole','Symphony_Fantastique','Boogie_Stop_Shuffle','My_Favorite_Things','Blue_Monk','All_Blues'])

#durs = np.array([90,180,180,90,135,180,180,225,225,135,90,135,225,225,90,135])

songs = np.array(['I_Love_Music','Moonlight_Sonata','Waltz_of_Flowers','The_Bird','Island','Allegro_Moderato','Finlandia','Early_Summer','Symphony_Fantastique','Boogie_Stop_Shuffle','My_Favorite_Things','All_Blues'])

durs = np.array([180,180,135,180,180,225,225,135,135,225,225,135])



events = np.zeros((91,109,91))


for i in range(len(durs)):
    K = np.round(durs[i]/event_length)
    if K > 19:
        K = 19
    K = str(K).split(".")[0]
    fn = datadir + songs[i] + '/avg_data/globals_avg_real_n25_k' + str(K) + '.npy'
    print(fn)
    data = np.load(fn)
    events[:,:,:] += data/len(durs)

# Plot and save searchlight results
maxval = np.max(events)
minval = np.min(events)
img = nib.Nifti1Image(events, affine=nii_template.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + str(event_length) + '_sec_events.nii.gz') 
