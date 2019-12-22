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

#Iterate over all the key value pairs in dictionary and call the given
#callback function() on each pair. Items for which callback() returns True,
#add them to the new dictionary. In the end return the new dictionary.

def filterTheDict(dictObj, callback):
    newDict = dict()
    # Iterate over all the items in dictionary
    for (key, value) in dictObj.items():
        # Check if item satisfies the given condition then add to new dict
        if callback((key, value)):
            newDict[key] = value
    return newDict

dur_vec = np.array([90,135,180,225])
event_length_mat = dur_vec/np.arange(1,226)[:,None]
unique, counts = np.unique(event_length_mat, return_counts=True)
dict1 = dict(zip(unique,counts))
newDict = filterTheDict(dict1, lambda elem: elem[1] == 4)

fairK = []

for i in range(len(newDict)):
    if list(newDict.keys())[i] % 1 == 0:
        fairK.append(list(newDict.items())[i][0])

song_number = int(sys.argv[1]) - 1
roi = str(sys.argv[2])

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
train = np.nan_to_num(stats.zscore(np.load(datadir + 'zstats_human_bounds_left_precuneus_run1_n25.npy'),axis=1,ddof=1))
test = np.nan_to_num(stats.zscore(np.load(datadir + 'zstats_human_bounds_left_precuneus_run2_n25.npy'),axis=1,ddof=1))

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
srm_train_run1 = SRM(n_iter=n_iter, features=features)
srm_train_run2 = SRM(n_iter=n_iter, features=features)

# Fit model to training data (run 1)
print('Training Model')
srm_train_run1.fit(train_list)
srm_train_run2.fit(test_list)

# Test model on testing data to produce shared response
print('Testing Model')
shared_data_train_run1 = srm_train_run1.transform(test_list)
shared_data_train_run2 = srm_train_run2.transform(train_list)

avg_response_train_run1 = sum(shared_data_train_run1)/len(shared_data_train_run1)
avg_response_train_run2 = sum(shared_data_train_run2)/len(shared_data_train_run2)

avg_response = (avg_response_train_run1 + avg_response_train_run2)/2

nTR = shared_data_train_run1[0][:,start:end].shape[1]
nSubj = 25

# Fit HMM
ev = brainiak.eventseg.event.EventSegment(len(human_bounds) - 1)
ev.fit(avg_response[:,start:end].T)
events = np.argmax(ev.segments_[0],axis=1)

max_event_length = stats.mode(events)[1][0]
# compute timepoint by timepoint correlation matrix 
cc = np.corrcoef(loo.T) # Should be a time by time correlation matrix

# Create a mask to only look at values up to max_event_length
local_mask = np.zeros(cc.shape, dtype=bool)
for k in range(1,max_event_length):
	local_mask[np.diag(np.ones(cc.shape[0]-k, dtype=bool), k)] = True

# Compute within vs across boundary correlations
same_event = events[:,np.newaxis] == events
within = cc[same_event*local_mask].mean()
across = cc[(~same_event)*local_mask].mean()
within_across = within - across
