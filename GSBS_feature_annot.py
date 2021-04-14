from statesegmentation import GSBS
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

set_bounds = 0

song_bounds = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

def GSBS_helper(feature, K, set_bounds):
    states = GSBS(x=feature.T, kmax=K+1)
    states.fit()

    if set_bounds == 0:
        bounds = np.round(np.nonzero(states.get_bounds())[0])
    elif set_bounds == 1:
        bounds = np.round(np.nonzero(states.get_bounds(K))[0])

    return bounds

# load features
datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_input/'
savedir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/GSBS_annotations/'

chroma = np.load(datadir + 'chromaRun1_no_hrf.npy')
mfcc   = np.load(datadir + 'mfccRun1_no_hrf.npy')[0:12,:]
tempo  = np.load(datadir + 'tempoRun1_12PC_singles_no_hrf.npy')

for i in range(len(songs)):
    # extract song-specific timepoints
    songChroma = chroma[:,song_bounds[i]:song_bounds[i+1]]
    songMFCC = mfcc[:,song_bounds[i]:song_bounds[i+1]]
    songTempo = tempo[:,song_bounds[i]:song_bounds[i+1]]
    songCombo = zscore(np.vstack((songChroma,songMFCC,songTempo)),axis=1)    

    # compute feature boundaries
    print('computing chroma bounds for: ', songs[i])
    chromaBounds = GSBS_helper(songChroma, songChroma.shape[1], set_bounds)

    print('computing mfcc bounds for: ', songs[i])
    mfccBounds   = GSBS_helper(songMFCC, songMFCC.shape[1], set_bounds)

    print('computing tempo bounds for: ', songs[i])
    tempoBounds  = GSBS_helper(songTempo, songTempo.shape[1], set_bounds)   
    
    print('computing combined feature bounds for: '. songs[i])
    comboBounds  = GSBS_helper(songCombo, songCombo.shape[1], set_bounds)   
 
    # save bounds
    np.save(savedir + songs[i] + '/chroma_bounds_kmax_all_timepoints', chromaBounds)
    np.save(savedir + songs[i] + '/mfcc_bounds_kmax_all_timepoints', mfccBounds)
    np.save(savedir + songs[i] + '/tempo_bounds_kmax_all_timepoints', tempoBounds)
    np.save(savedir + songs[i] + '/combo_bounds_kmax_all_timepoints', comboBounds)        
