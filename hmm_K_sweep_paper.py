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

K = np.array((3,5,9,15,20,25,30,35,40,45))

def single_gamma_hrf(TR, t=5, d=5.2, onset=0, kernel=32):
    """Single gamma hemodynamic response function.
    Parameters
    ----------
    TR : float
        Repetition time at which to generate the HRF (in seconds).
    t : float (default=5.4)
        Delay of response relative to onset (in seconds).
    d : float (default=5.2)
        Dispersion of response.
    onset : float (default=0)
        Onset of hemodynamic response (in seconds).
    kernel : float (default=32)
        Length of kernel (in seconds).
    Returns
    -------
    hrf : array
        Hemodynamic repsonse function
    References
    ----------
    [1] Adapted from the pymvpa tools.
        https://github.com/PyMVPA/PyMVPA/blob/master/mvpa2/misc/fx.py
    """

    ## Define metadata.
    fMRI_T = 16.0
    TR = float(TR)

    ## Define times.
    dt = TR/fMRI_T
    u  = np.arange(kernel/dt + 1) - onset/dt
    u *= dt

    ## Generate (super-sampled) HRF.
    hrf = (u / t) ** ((t ** 2) / (d ** 2) * 8.0 * np.log(2.0)) \
          * np.e ** ((u - t) / -((d ** 2) / t / 8.0 / np.log(2.0)))

    ## Downsample.
    good_pts=np.array(range(np.int(kernel/TR)))*fMRI_T
    hrf=hrf[good_pts.astype(int)]

    ## Normalize and return.
    hrf = hrf/np.sum(hrf)
    return hrf

#########################################################################################
# Here we train and test the model on both runs separately. This will result in two SRM-ified datasets: one for all of run 1 and one for all of run 2. Songs from these datasets will be indexed separately in the following HMM step and then averaged before fitting the HMM.

# run 1 times
song_bounds_run1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

songs_run1 = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

durs_run1 = np.array([225,90,180,135,90,180,135,90,180,135,225,90,225,225,180,135])

# run 2 times
song_bounds_run2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

songs_run2 = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

durs_run2 = np.array([90,180,180,90,135,180,180,225,225,135,90,135,225,225,90,135])

hrf = 5

# Load in data
run1 = np.nan_to_num(stats.zscore(np.load(datadir + 'zstats_right_a1_version2_run1_n25.npy'),axis=1,ddof=1))
run2 = np.nan_to_num(stats.zscore(np.load(datadir + 'zstats_right_a1_version2_run2_n25.npy'),axis=1,ddof=1))

nSubj = run1.shape[2]

# Convert data into lists where each element is voxels by samples and convolve data with hrf in the process
run1_list = []
run2_list = []
for i in range(0,nSubj):
    run1_list.append(np.apply_along_axis(np.convolve, 1, run1[:,:,i], hrf, 'full')[:run1.shape[1]][:,0:2511])
    run2_list.append(np.apply_along_axis(np.convolve, 1, run2[:,:,i], hrf, 'full')[:run2.shape[1]][:,0:2511])

n_iter = 50
features = 10
# Initialize model
print('Building Model')
srm_train_run1 = SRM(n_iter=n_iter, features=features)
srm_train_run2 = SRM(n_iter=n_iter, features=features)

# Fit model to training data
print('Training Model')
srm_train_run1.fit(run1_list)
srm_train_run2.fit(run2_list)

# Test model on testing data to produce shared response
print('Testing Model')
shared_data_train_run1 = srm_train_run1.transform(run2_list)
shared_data_train_run2 = srm_train_run2.transform(run1_list)

avg_response_train_run1 = sum(shared_data_train_run1)/len(shared_data_train_run1)
avg_response_train_run2 = sum(shared_data_train_run2)/len(shared_data_train_run2)

##################################################################################
wVa_results = np.zeros((16,len(fairK)))

for i in range(16):
    print('song number ',str(i))
    # grab start and end time for each song from bound vectors. for SRM data trained on run 1 and tested on run 2, use song name from run 1 to find index for song onset in run 2 bound vector 
    start_run1 = song_bounds_run2[songs_run2.index(songs_run1[i])]
    end_run1   = song_bounds_run2[songs_run2.index(songs_run1[i])+1]
    start_run2 = song_bounds_run1[i]
    end_run2   = song_bounds_run1[i+1]
    # chop song from bold data
    data1 = avg_response_train_run1[:,start_run1:end_run1]
    data2 = avg_response_train_run2[:,start_run2:end_run2]
    # average song-specific bold data from each run 
    data = (data1 + data2)/2
    for j in range(len(fairK)):
        # Fit HMM
        ev = brainiak.eventseg.event.EventSegment(int(fairK[j]))
        ev.fit(data.T)
        events = np.argmax(ev.segments_[0],axis=1)

        max_event_length = stats.mode(events)[1][0]
        # compute timepoint by timepoint correlation matrix 
        cc = np.corrcoef(data.T) # Should be a time by time correlation matrix

        # Create a mask to only look at values up to max_event_length
        local_mask = np.zeros(cc.shape, dtype=bool)
        for k in range(1,max_event_length):
	        local_mask[np.diag(np.ones(cc.shape[0]-k, dtype=bool), k)] = True

        # Compute within vs across boundary correlations
        same_event = events[:,np.newaxis] == events
        within = cc[same_event*local_mask].mean()
        across = cc[(~same_event)*local_mask].mean()
        within_across = within - across
        wVa_results[i,j] = within_across


np.save('/jukebox/norman/jamalw/MES/prototype//link/scripts/k_sweep_results_paper/a1_wva',wVa_results)
