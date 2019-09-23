import numpy as np

datadir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/music_features/"

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

avg_data = np.zeros((91,109,91))

for s in range(len(songs)):
    data = np.load(datadir + songs[s] + '/chroma/zscores/globals_z_srm_k30_fit_run2.npy')
    avg_data[:,:,:] += data/len(songs)
    

np.save(datadir + 'avg_chroma_run2_across_songs',avg_data) 

