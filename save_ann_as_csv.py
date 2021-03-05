import numpy as np
import pandas as pd

ann_dir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

data = []

for song_idx in range(len(songs)):
    human_bounds = np.load(ann_dir + songs[song_idx] + '/' + songs[song_idx] + '_beh_seg.npy')
    data.append(pd.Series(human_bounds, name=songs[song_idx]))

data = pd.concat(data, axis=1)

data.to_csv('event_seg_annotations.csv')
