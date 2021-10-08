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

# initialize match variables
chroma_match = np.zeros((len(songs),nPerm+1))
mfcc_match = np.zeros((len(songs),nPerm+1))
tempo_match = np.zeros((len(songs),nPerm+1))
spect_match = np.zeros((len(songs),nPerm+1))

# initialize total feature and human bounds variables
total_chroma_bounds = 0
total_mfcc_bounds = 0
total_tempo_bounds = 0
total_spect_bounds = 0
total_human_bounds = 0
total_human_bounds_no_match = 0

# loop over each song and compute conditional probability that a human boundary occurs at the same time as a feature boundary across songs
for s in range(len(songs)): 
    # load human bounds and GSBS bounds for song n
    human_bounds = np.load(datadir + 'data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[s] + '/' + songs[s] + '_beh_seg.npy') 
    chroma_bounds = np.load(datadir + 'GSBS_annotations/' + songs[s] + '/chroma_bounds_kmax_all_timepoints.npy')   
    mfcc_bounds = np.load(datadir + 'GSBS_annotations/' + songs[s] + '/mfcc_bounds_kmax_all_timepoints.npy')   
    tempo_bounds = np.load(datadir + 'GSBS_annotations/' + songs[s] + '/tempo_bounds_kmax_all_timepoints.npy')   
    spect_bounds = np.load(datadir + 'GSBS_annotations/' + songs[s] + '/spect_bounds_kmax_all_timepoints.npy')    
 
    # collect total number of feature bounds
    total_chroma_bounds += len(chroma_bounds)
    total_mfcc_bounds += len(mfcc_bounds)
    total_tempo_bounds += len(tempo_bounds)
    total_spect_bounds += len(spect_bounds)

    # collect total number of human bounds
    total_human_bounds += len(human_bounds)

    # get human bound event lengths
    event_lengths = np.diff(np.concatenate(([0],human_bounds,[song_durs[s]]))) 

    # compute sum of matches between GSBS bounds and human bounds for each feature separately, counting as a match if 3 seconds away from each other

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
        if human_bounds_matches == 0:
            total_human_bounds_no_match += 1

    for p in range(nPerm+1):
        for hb in human_bounds:
            if np.any(np.abs(chroma_bounds - hb) <= w):
                chroma_match[s,p] += 1
            if np.any(np.abs(mfcc_bounds - hb) <= w):
                mfcc_match[s,p] += 1
            if np.any(np.abs(tempo_bounds - hb) <= w):
                tempo_match[s,p] += 1
            if np.any(np.abs(spect_bounds - hb) <= w):
                spect_match[s,p] += 1

        np.random.seed(p)
        human_bounds = np.cumsum(np.random.permutation(event_lengths))[:-1]

# sum matches across songs and permutations
chroma_match_sum = np.sum(chroma_match,axis=0)
mfcc_match_sum = np.sum(mfcc_match,axis=0)
tempo_match_sum = np.sum(tempo_match,axis=0)
spect_match_sum = np.sum(spect_match,axis=0)

# compute conditional probability of human feature match given a feature boundary
chroma_conditional = chroma_match_sum/total_chroma_bounds    
mfcc_conditional = mfcc_match_sum/total_mfcc_bounds
tempo_conditional = tempo_match_sum/total_tempo_bounds
spect_conditional = spect_match_sum/total_spect_bounds


# compute conditional probability of human feature match given a human boundary
chroma_human_conditional = chroma_match_sum/total_human_bounds    
mfcc_human_conditional  = mfcc_match_sum/total_human_bounds
tempo_human_conditional = tempo_match_sum/total_human_bounds
spect_human_conditional = spect_match_sum/total_human_bounds

# plot conditionals vs nulls
fig, axs = plt.subplots(2,2,figsize=(17,17))
im1 = axs[0,0].hist(chroma_conditional[1:])

fig.suptitle('Probability of match given a feature boundary',fontweight='bold',fontsize=20)

axs[0,0].set_title('chroma',fontweight='bold',fontsize=17)
axs[0,0].set_xlabel('p(human|chroma)',fontsize=14)
axs[0,0].axvline(x=chroma_conditional[0], color='r', linewidth=3)
chroma_conditional_pval = sc.norm.sf((chroma_conditional[0] - np.mean(chroma_conditional[1:]))/np.std(chroma_conditional[1:]))

print('p(human|chroma) = ' + str(chroma_conditional[0]) + ' pval = ' + str(chroma_conditional_pval) + ' null mean = ' + str(np.mean(chroma_conditional[1:])))

im2 = axs[0,1].hist(mfcc_conditional[1:])
axs[0,1].set_title('mfcc',fontweight='bold',fontsize=17)
axs[0,1].set_xlabel('p(human|mfcc)',fontsize=14)
axs[0,1].axvline(x=mfcc_conditional[0], color='r', linewidth=3)
mfcc_conditional_pval = sc.norm.sf((mfcc_conditional[0] - np.mean(mfcc_conditional[1:]))/np.std(mfcc_conditional[1:]))

print('p(human|mfcc) = ' + str(mfcc_conditional[0]) + ' pval = ' + str(mfcc_conditional_pval) + ' null mean = ' + str(np.mean(mfcc_conditional[1:])))

im3 = axs[1,0].hist(tempo_conditional[1:])
axs[1,0].set_title('tempo',fontweight='bold',fontsize=17)
axs[1,0].set_xlabel('p(human|tempo)',fontsize=14)
axs[1,0].axvline(x=tempo_conditional[0], color='r', linewidth=3)
tempo_conditional_pval = sc.norm.sf((tempo_conditional[0] - np.mean(tempo_conditional[1:]))/np.std(tempo_conditional[1:]))

print('p(human|tempo) = ' + str(tempo_conditional[0]) + ' pval = ' + str(tempo_conditional_pval) + ' null mean = ' + str(np.mean(tempo_conditional[1:])))

im4 = axs[1,1].hist(spect_conditional[1:])
axs[1,1].set_title('spect',fontweight='bold',fontsize=17)
axs[1,1].set_xlabel('p(human|spectrogram)',fontsize=14)
axs[1,1].axvline(x=spect_conditional[0], color='r', linewidth=3)
spect_conditional_pval = sc.norm.sf((spect_conditional[0] - np.mean(spect_conditional[1:]))/np.std(spect_conditional[1:]))

print('p(human|spect) = ' + str(spect_conditional[0]) + ' pval = ' + str(spect_conditional_pval) + ' null mean = ' + str(np.mean(spect_conditional[1:])))

#plt.savefig('plots/paper_versions/conditonal_feature_match_vs_total_feature_bounds_with_spect_fixed_null.png')

# plot conditionals vs nulls
fig, axs = plt.subplots(2,2,figsize=(17,17))

fig.suptitle('Probability of match given a human boundary',fontweight='bold',fontsize=20)

im1 = axs[0,0].hist(chroma_human_conditional[1:])
axs[0,0].set_title('chroma',fontweight='bold',fontsize=17)
axs[0,0].set_xlabel('p(chroma match|human)',fontsize=14)
axs[0,0].axvline(x=chroma_human_conditional[0], color='r', linewidth=3)
chroma_human_conditional_pval = sc.norm.sf((chroma_human_conditional[0] - np.mean(chroma_human_conditional[1:]))/np.std(chroma_human_conditional[1:]))

#print('p(chroma match|human) = ' + str(chroma_human_conditional[0]) + ' pval = ' + str(chroma_human_conditional_pval))

im2 = axs[0,1].hist(mfcc_human_conditional[1:])
axs[0,1].set_title('mfcc',fontweight='bold',fontsize=17)
axs[0,1].set_xlabel('p(mfcc match|human)',fontsize=14)
axs[0,1].axvline(x=mfcc_human_conditional[0], color='r', linewidth=3)
mfcc_human_conditional_pval = sc.norm.sf((mfcc_human_conditional[0] - np.mean(mfcc_human_conditional[1:]))/np.std(mfcc_human_conditional[1:]))

#print('p(mfcc match|human) = ' + str(mfcc_human_conditional[0]) + ' pval = ' + str(mfcc_human_conditional_pval))

im3 = axs[1,0].hist(tempo_human_conditional[1:])
axs[1,0].set_title('tempo',fontweight='bold',fontsize=17)
axs[1,0].set_xlabel('p(tempo match|human)',fontsize=14)
axs[1,0].axvline(x=tempo_human_conditional[0], color='r', linewidth=3)
tempo_human_conditional_pval = sc.norm.sf((tempo_human_conditional[0] - np.mean(tempo_human_conditional[1:]))/np.std(tempo_human_conditional[1:]))

#print('p(tempo match|human) = ' + str(tempo_human_conditional[0]) + ' pval = ' + str(tempo_human_conditional_pval))

im4 = axs[1,1].hist(spect_human_conditional[1:])
axs[1,1].set_title('spectrogram',fontweight='bold')
axs[1,1].set_xlabel('p(spectrogram match|human)',fontsize=14)
axs[1,1].axvline(x=spect_human_conditional[0], color='r', linewidth=3)
spect_human_conditional_pval = sc.norm.sf((spect_human_conditional[0] - np.mean(spect_human_conditional[1:]))/np.std(spect_human_conditional[1:]))

#print('p(spect match|human) = ' + str(spect_human_conditional[0]) + ' pval = ' + str(spect_human_conditional_pval))

#plt.savefig('plots/paper_versions/conditonal_feature_match_vs_total_human_bounds_with_spect_fixed_null.png')




