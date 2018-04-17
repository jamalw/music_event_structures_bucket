import deepdish as dd
import numpy as np
import brainiak.eventseg.event
from scipy.stats import norm, zscore, pearsonr, stats
from scipy.signal import gaussian, convolve
from sklearn import decomposition
import numpy as np
from brainiak.funcalign.srm import SRM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()


datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
ann_dirs = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

# Load in data
train = np.nan_to_num(stats.zscore(np.load(datadir + 'zstats_human_bounds_bilateral_mpfc_run1_n25.npy'),axis=1,ddof=1))
test = np.nan_to_num(stats.zscore(np.load(datadir + 'zstats_human_bounds_bilateral_mpfc_run2_n25.npy'),axis=1,ddof=1))
train_list = []
test_list = []
for i in range(0,train.shape[2]):
    train_list.append(train[:,:,i])
    test_list.append(test[:,:,i])

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

durs = np.array([90,180,180,90,135,180,180,225,225,135,90,135,225,225,90,135])
song_bounds = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])
features = np.array([5,10,15,20,25,30,35,40,45,50])
n_iter = 50
hrf = 5
nPerm = 1000
w = 3
z_scores = np.zeros((len(features),len(songs)))
t_scores = np.zeros((len(features)))

for k in range(len(features)):
    srm = SRM(n_iter=n_iter, features=features[k])

    # Fit model to training data (run 1)
    srm.fit(train_list)

    # Test model on testing data to produce shared response
    shared_data = srm.transform(test_list)

    avg_response = sum(shared_data)/len(shared_data)

    for s in range(len(songs)):
        # Get start and end of chosen song
        start = song_bounds[s] + hrf
        end = song_bounds[s+1] + hrf 

        nTR = shared_data[0][:,start:end].shape[1]
        
        human_bounds = np.load(ann_dirs + songs[s] + '/' + songs[s] + '_beh_seg.npy')

        human_bounds = np.append(0,np.append(human_bounds,durs[s]))

        K = len(human_bounds) - 1

        ev = brainiak.eventseg.event.EventSegment(K)
        ev.fit(avg_response[:,start:end].T)

        bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]
        
        match = np.zeros(nPerm+1)
        perm_bounds = bounds.copy()

        for p in range(nPerm+1):
            for hb in human_bounds:
                if np.any(np.abs(perm_bounds - hb) <= w):
                    match[p] += 1
            match[p] /= len(human_bounds)
            np.random.seed(p)
            perm_bounds = np.random.choice(nTR,K-1,replace=False)
        
        z_scores[k,s] = (match[0] - np.mean(match[1:]))/np.std(match[1:])
 
    t_scores[k] = stats.ttest_1samp(z_scores[k,:],0,axis=0)[0]

savedir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_srm/plots/bilateral_mPFC/'   
np.save(savedir + 'zscores',z_scores)
np.save(savedir + 'tstats', t_scores)

fig, ax1 = plt.subplots()

xp = np.linspace(3,features[-1:],100)

p1 = np.poly1d(np.polyfit(features,mean_z,2))
ax1.plot(features,mean_z,'.',xp, p1(xp), '-',color='k',linewidth=3,markersize=15)
ax1.set_ylabel('average z', color='k', fontsize=18)
ax1.tick_params(labelsize=15)

ax2 = ax1.twinx()

p2 = np.poly1d(np.polyfit(features,t_scores,2))
ax2.plot(features,t_scores,'.',xp, p2(xp), '-',color='m',linewidth=3,markersize=15)
ax2.set_ylabel('t-statistic', color='m', fontsize=18)
ax2.tick_params(labelsize=15)

plt.xticks(features,features)

plt.title('Bilateral mPFC Best SRM K',fontsize=18)

plt.savefig(savedir + 'SRM_K_vs_Z')
