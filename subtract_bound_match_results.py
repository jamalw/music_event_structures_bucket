import numpy as np
import nibabel as nib

datadir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/"

songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']


for i in range(len(songs)):
    data = nib.load(datadir + 'HMM_searchlight_human_bounds_fit_to_all/' + songs[i] + '/zscores/globals_avg_both_z_runs_n25_srmk_30.nii.gz')
    data_regressed = nib.load(datadir + 'HMM_searchlight_human_bounds_regress_music_features/' + songs[i] + '/zscores/globals_avg_both_z_runs_n25_srmk_30.nii.gz')
    
