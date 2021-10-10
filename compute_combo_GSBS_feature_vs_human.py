import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc

songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

song_durs = np.array([225,89,180,134,90,180,134,90,179,135,224,89,224,225,179,134])

human_dir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'
feature_dir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/GSBS_annotations/'

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/'

# set boundary match window
w = 3

# set number of permutations
nPerm = 1000

# initialize total feature and human bounds variables
no_match = 0
one_match = 0
two_match = 0
three_match = 0
four_match = 0

# loop over each song and compute conditional probability that a human boundary occurs at the same time as a feature boundary across songs
for s in range(len(songs)): 
    # load human bounds and GSBS bounds for song n
    human_bounds = np.load(datadir + 'data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[s] + '/' + songs[s] + '_beh_seg.npy') 
    chroma_bounds = np.load(datadir + 'GSBS_annotations/' + songs[s] + '/chroma_bounds_kmax_all_timepoints.npy')   
    mfcc_bounds = np.load(datadir + 'GSBS_annotations/' + songs[s] + '/mfcc_bounds_kmax_all_timepoints.npy')   
    tempo_bounds = np.load(datadir + 'GSBS_annotations/' + songs[s] + '/tempo_bounds_kmax_all_timepoints.npy')   
    spect_bounds = np.load(datadir + 'GSBS_annotations/' + songs[s] + '/spect_bounds_kmax_all_timepoints.npy')    
 
    # get human bound event lengths
    event_lengths = np.diff(np.concatenate(([0],human_bounds,[song_durs[s]]))) 

    # compute number of feature matches corresponding to human annotation, counting as a match if 3 seconds away from each other

    # add a 1 to human_bounds_matches variable if human bound matches a feature bound 
    for hb in human_bounds:
    # initalize/reset variable that will count number of feature to human matches to determine if a given human boundary has any match at all
        human_bounds_matches = 0
        if np.any(np.abs(chroma_bounds - hb) <= w):
            human_bounds_matches += 1
        if np.any(np.abs(mfcc_bounds - hb) <= w):
            human_bounds_matches += 1
        if np.any(np.abs(tempo_bounds - hb) <= w):
            human_bounds_matches += 1
        if np.any(np.abs(spect_bounds - hb) <= w):
            human_bounds_matches += 1
        # tally up number of human to feature matches 
        if human_bounds_matches == 0:
            no_match += 1
        if human_bounds_matches == 1:
            one_match += 1
        if human_bounds_matches == 2:
            two_match += 1
        if human_bounds_matches == 3:
            three_match += 1
        if human_bounds_matches == 4:
            four_match += 1

#    for p in range(nPerm+1):
#        for hb in human_bounds:
#            if np.any(np.abs(chroma_bounds - hb) <= w):
#                chroma_match[s,p] += 1
#            if np.any(np.abs(mfcc_bounds - hb) <= w):
#                mfcc_match[s,p] += 1
#            if np.any(np.abs(tempo_bounds - hb) <= w):
#                tempo_match[s,p] += 1
#            if np.any(np.abs(spect_bounds - hb) <= w):
#                spect_match[s,p] += 1
#
#        np.random.seed(p)
#        human_bounds = np.cumsum(np.random.permutation(event_lengths))[:-1]

all_match = np.concatenate(([np.zeros(no_match),np.ones(one_match), np.ones(two_match)*2, np.ones(three_match)*3, np.ones(four_match)*4]))


fig, axs = plt.subplots(1,1,figsize=(8,6))

data = all_match
data = np.array(data)

d = np.diff(np.unique(data)).min()
left_of_first_bin = data.min() - float(d)/2
right_of_last_bin = data.max() + float(d)/2
plt.hist(data, np.arange(left_of_first_bin, right_of_last_bin + d, d),alpha=0.5, histtype='bar', ec='black')
plt.xticks(fontsize = 17) 
plt.yticks(fontsize = 17)
axs.set_xlabel('Number of Overlapping Features',fontsize=18)
axs.set_ylabel('Count',fontsize=18)
#fig.suptitle('Probability of match given a feature boundary',fontweight='bold',fontsize=20)
#
#axs[0,0].set_title('chroma',fontweight='bold',fontsize=17)
#axs[0,0].set_xlabel('p(human|chroma)',fontsize=14)
#axs[0,0].axvline(x=chroma_conditional[0], color='r', linewidth=3)
#
