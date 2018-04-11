#import deepdish as dd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import numpy as np
import brainiak.eventseg.event
from scipy.stats import norm, zscore, pearsonr, stats
from scipy.signal import gaussian, convolve
from sklearn import decomposition
import numpy as np
from brainiak.funcalign.srm import SRM
import sys

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
ann_dirs = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

song_number = int(sys.argv[1]) - 1

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

durs = np.array([90,180,180,90,135,180,180,225,225,135,90,135,225,225,90,135]) 

hrf = 5

human_bounds = np.load(ann_dirs + songs[song_number] + '/' + songs[song_number] + '_beh_seg.npy')

human_bounds = np.append(0,np.append(human_bounds,durs[song_number])) 

song_bounds = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

# Get start and end of chosen song
start = song_bounds[song_number] + hrf
end = song_bounds[song_number + 1] + hrf

# Load in data
train = np.nan_to_num(stats.zscore(np.load(datadir + 'zstats_human_bounds_bilateral_mpfc_run1_n25.npy'),axis=1,ddof=1))
test = np.nan_to_num(stats.zscore(np.load(datadir + 'zstats_human_bounds_bilateral_mpfc_run2_n25.npy'),axis=1,ddof=1))

# Convert data into lists where each element is voxels by samples
train_list = []
test_list = []
for i in range(0,train.shape[2]):
    train_list.append(train[:,:,i])
    test_list.append(test[:,:,i])

n_iter = 50
features = 10
# Initialize model
print('Building Model')
srm = SRM(n_iter=n_iter, features=features)

# Fit model to training data (run 1)
print('Training Model')
srm.fit(train_list)

# Test model on testing data to produce shared response
print('Testing Model')
shared_data = srm.transform(test_list)

avg_response = sum(shared_data)/len(shared_data)

nR = shared_data[0].shape[0]
nTR = shared_data[0][:,start:end].shape[1]
nSubj = len(shared_data)

ev = brainiak.eventseg.event.EventSegment(len(human_bounds) - 1)
ev.fit(avg_response[:,start:end].T)

bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]

plt.figure(figsize=(10,10))
plt.imshow(np.corrcoef(avg_response[:,start:end].T))
plt.colorbar()
ax = plt.gca()
bounds_aug = np.concatenate(([0],bounds,[nTR]))
for i in range(len(bounds_aug)-1):
    rect1 = patches.Rectangle((bounds_aug[i],bounds_aug[i]),bounds_aug[i+1]-bounds_aug[i],bounds_aug[i+1]-bounds_aug[i],linewidth=3,edgecolor='w',facecolor='none',label='Model Fit')
    ax.add_patch(rect1)

for i in range(len(human_bounds)-1):
    rect2 = patches.Rectangle((human_bounds[i],human_bounds[i]),human_bounds[i+1]-human_bounds[i],human_bounds[i+1]-human_bounds[i],linewidth=3,edgecolor='k',facecolor='none',label='Human Annotations')
    ax.add_patch(rect2)

song_titles = ['St Pauls Suite', 'I Love Music', 'Moonlight Sonata', 'Change of the Guard','Waltz of Flowers','The Bird', 'Island', 'Allegro Moderato', 'Finlandia', 'Early Summer', 'Capriccio Espagnole', 'Symphony Fantastique', 'Boogie Stop Shuffle', 'My Favorite Things', 'Blue Monk','All Blues']

plt.title('HMM Fit to mPFC for ' + song_titles[song_number],fontsize=18,fontweight='bold')
plt.xlabel('TRs',fontsize=18,fontweight='bold')
plt.ylabel('TRs',fontsize=18,fontweight='bold')
plt.legend(handles=[rect1,rect2])

plt.savefig('plots/' + songs[song_number] + '_srm_k_' + str(features) + '_iter_' + str(n_iter))

#for i in range(len(human_bounds2)-1):
#    plt.figure(i+1)
#    plt.imshow(np.corrcoef(avg_response[:,human_bounds2[i]:human_bounds2[i+1]].T))
#    plt.title(songs[i] + ' A1 SRM',fontsize=18)
#    plt.xlabel('TRs',fontsize=18,fontweight='bold')
#    plt.ylabel('TRs',fontsize=18,fontweight='bold')    
#    plt.savefig('plots/'+songs[i] + '_A1_SRM') 
