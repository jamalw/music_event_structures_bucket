import numpy as np

songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

song_durs = np.array([225,89,180,134,90,180,134,90,179,135,224,89,224,225,179,134])

human_dir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'
feature_dir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/GSBS_annotations/'

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/'

# set boundary match window
w = 3

# set number of permutations
nPerm = 1000

# initialize match variables
chroma_match = np.zeros((len(songs),nPerm+1))

# initialize total feature bounds variables
total_chroma_bounds = 0

# loop over each song and compute conditional probability that a human boundary occurs at the same time as a feature boundary across songs
for s in range(len(songs)): 
    # load human bounds and GSBS bounds for song n
    human_bounds = np.load(datadir + 'data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[s] + '/' + songs[s] + '_beh_seg.npy') 
    chroma_bounds = np.load(datadir + 'GSBS_annotations/' + songs[s] + '/chroma_bounds_kmax_all_timepoints.npy')   
    
    # collect total number of feature bounds
    total_chroma_bounds += len(chroma_bounds)

    # get human bound event lengths
    event_lengths = np.diff(np.append(0,human_bounds,song_durs[s])) - 1 

    # compute sum of matches between GSBS bounds and human bounds for each feature separately, counting as a match if 3 seconds away from each other
    chroma_perm_bounds = chroma_bounds.copy()
 
    for p in range(nPerm+1):
        for hb in human_bounds:
            if np.any(np.abs(chroma_perm_bounds - hb) <= w):
                chroma_match[s,p] += 1
        np.random.seed(p)
        chroma_perm_bounds = np.cumsum(np.random.permutation(event_lengths))[:-1]

# sum matches across songs and permutations
chroma_match_sum = np.sum(chroma_match,axis=0)

# compute conditional probability
chroma_conditional = chroma_match_sum/total_chroma_bounds     
