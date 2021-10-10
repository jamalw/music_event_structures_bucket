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
no_match = np.zeros(nPerm+1)
one_match = np.zeros(nPerm+1)
two_match = np.zeros(nPerm+1)
three_match = np.zeros(nPerm+1)
four_match = np.zeros(nPerm+1)


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

    for p in range(nPerm+1):
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
                no_match[p] += 1
            if human_bounds_matches == 1:
                one_match[p] += 1
            if human_bounds_matches == 2:
                two_match[p] += 1
            if human_bounds_matches == 3:
                three_match[p] += 1
            if human_bounds_matches == 4:
                four_match[p] += 1

        np.random.seed(p)
        human_bounds = np.cumsum(np.random.permutation(event_lengths))[:-1]

# convert raw match counts to tally
all_match = np.concatenate(([np.zeros(int(no_match[0])),np.ones(int(one_match[0])), np.ones(int(two_match[0]))*2, np.ones(int(three_match[0]))*3, np.ones(int(four_match[0]))*4]))

no_match_null = np.zeros(int(np.mean(no_match[1:])))
one_match_null = np.ones(int(np.mean(one_match[1:])))
two_match_null = np.ones(int(np.mean(two_match[1:]))) * 2
three_match_null = np.ones(int(np.mean(three_match[1:]))) * 3
four_match_null = np.ones(int(np.mean(four_match[1:]))) * 4

all_match_null = np.concatenate((no_match_null,one_match_null,two_match_null,three_match_null,four_match_null)) 

# plot data
fig, axs = plt.subplots(1,1,figsize=(8,6))

data = all_match
data = np.array(data)
data_null = all_match_null
data_null = np.array(data_null)

d = np.diff(np.unique(data)).min()
left_of_first_bin = data.min() - float(d)/2
right_of_last_bin = data.max() + float(d)/2
plt.hist(data, np.arange(left_of_first_bin, right_of_last_bin + d, d),alpha=0.5, histtype='bar', ec='black')
plt.hist(data_null, np.arange(left_of_first_bin, right_of_last_bin + d, d),alpha=0.5, histtype='bar', ec='black')
plt.xticks(fontsize = 17) 
plt.yticks(fontsize = 17)
axs.set_xlabel('# of Acoustic Features That Change',fontsize=18)
axs.set_ylabel('Matches to Human Annotations',fontsize=18)
axs.legend(['True','Null'],fontsize=14)

# compute pvalues
no_match_pval = sc.norm.sf((no_match[0] - np.mean(no_match[1:]))/np.std(no_match[1:]))
one_match_pval = sc.norm.sf((one_match[0] - np.mean(one_match[1:]))/np.std(one_match[1:]))
two_match_pval = sc.norm.sf((two_match[0] - np.mean(two_match[1:]))/np.std(two_match[1:]))
three_match_pval = sc.norm.sf((three_match[0] - np.mean(three_match[1:]))/np.std(three_match[1:]))
four_match_pval = sc.norm.sf((four_match[0] - np.mean(four_match[1:]))/np.std(four_match[1:]))

# print pvals
print("no match pval: ", no_match_pval)
print("one match pval: ", one_match_pval)
print("two match pval: ", two_match_pval)
print("three match pval: ", three_match_pval)
print("four match pval: ", four_match_pval)

# add pvalues to bar plots
match_pvals = np.array([no_match_pval,one_match_pval,two_match_pval,three_match_pval,four_match_pval])

data_y_max = np.maximum(np.unique(data,return_counts=True)[1].astype('int'),np.unique(data_null,return_counts=True)[1].astype('int'))

labels = ['p = %.3f' % no_match_pval, 'p = %.3f' % one_match_pval, 'p = %.3f' % two_match_pval + '**', 'p = %.3f' % three_match_pval, 'p = %.3f' % four_match_pval + '**']
xloc = np.arange(len(np.unique(data)))

for i in range(len(xloc)):
    plt.text(x = xloc[i]-0.4 , y = data_y_max[i]+0.2, s = labels[i], size = 12)

plt.tight_layout()

plt.savefig('plots/paper_versions/combo_feature_match_to_humans')
