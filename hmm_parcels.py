import numpy as np
import sys
import nibabel as nib
from scipy.spatial import distance
import brainiak.eventseg.event
from scipy.stats import norm,zscore,pearsonr,stats
from brainiak.funcalign.srm import SRM
import os
from sklearn import linear_model

idx = int(sys.argv[1])

def clean_data(data,motion):
    nTR = data.shape[1]
    motion = motion.T
    regr = linear_model.LinearRegression()
    regr.fit(motion[:,0:nTR].T, data[:,:].T)
    clean_data = data[:,:] - np.dot(regr.coef_, motion[:,0:nTR]) - regr.intercept_[:, np.newaxis]
    return clean_data  

def srm(run1,run2):
    # initialize model
    print('Building Models')
    n_iter= 50
    srm_k = 30
    srm_train_run1 = SRM(n_iter=n_iter, features=srm_k)
    srm_train_run2 = SRM(n_iter=n_iter, features=srm_k)
    
    # fit model to training data
    print('Training Models')
    srm_train_run1.fit(run1)
    srm_train_run2.fit(run2)

    print('Testing Models')
    shared_data_run1 = stats.zscore(np.dstack(srm_train_run2.transform(run1)),axis=1,ddof=1)
    shared_data_run2 = stats.zscore(np.dstack(srm_train_run1.transform(run2)),axis=1,ddof=1)

    # average test data across subjects
    run1 = np.mean(shared_data_run1,axis=2)
    run2 = np.mean(shared_data_run2, axis=2)

    return run1, run2
    
def HMM(X,human_bounds):

    """fit hidden markov model
  
       Fit HMM to average data and cross-validate with leftout subject using within song and between song average correlations              

       Parameters
       ----------
       A: voxel by time ndarray (2D)
       B: voxel by time ndarray (2D)
       C: voxel by time ndarray (2D)
       D: voxel by time ndarray (2D)
       K: # of events for HMM (scalar)
 
       Returns
       -------
       z: z-score after performing permuted cross-validation analysis      

    """

    # Fit to all but one subject
    nPerm=1000
    K = len(human_bounds) + 1
    ev = brainiak.eventseg.event.EventSegment(K)
    ev.fit(X.T)
    bounds = np.where(np.diff(np.argmax(ev.segments_[0],axis=1)))[0]
    match = np.zeros(nPerm+1)
    events = np.argmax(ev.segments_[0],axis=1)
    _, event_lengths = np.unique(events, return_counts=True)
    perm_bounds = bounds.copy()
    nTR = X.shape[1] 

    for p in range(nPerm+1):
        match[p] = sum([np.min(np.abs(perm_bounds - hb)) for hb in human_bounds])
        np.random.seed(p)
        perm_lengths = np.random.permutation(event_lengths)
        events = np.zeros(nTR, dtype=np.int)
        events[np.cumsum(perm_lengths[:-1])] = 1
        # pick number of timepoints to rotate by  
        nrot = np.random.randint(len(events))
        # convert events to list to allow combining lists in next step
        events_lst = list(events)
        # rotate boundaries
        events_rot = np.array(events_lst[-nrot:] + events_lst[:-nrot])
        # select indexes for new boundaries
        perm_bounds = np.where(events_rot == 1)[0]

    return match


# run 1 times
song_bounds1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

songs1 = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']


# get song start time and end time for run 1
start_idx_run1 = song_bounds1[idx]
end_idx_run1   = song_bounds1[idx+1] 

# run 2 times
song_bounds2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

songs2 = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

# get song start time and end time for run 2
song_name = songs1[idx]
start_idx_run2 = song_bounds2[songs2.index(song_name)]
end_idx_run2   = song_bounds2[songs2.index(song_name) + 1]  

subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

subjs = ['MES_022817_0', 'MES_030217_0','MES_032117_1']
#subj = 'MES_022817_0'

datadir = '/jukebox/norman/jamalw/MES/'
motion_dir = datadir + '/prototype/link/scripts/data/searchlight_input/'

mask_img = nib.load(datadir + 'data/mask_nonan.nii').get_data()
mask_reshape = np.reshape(mask_img,(91*109*91))

hrf = 5
human_bounds = np.load(datadir + 'prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + song_name + '/' + song_name + '_beh_seg.npy') + hrf

parcels = nib.load(datadir + "data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_100Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

run1_masked = []
run2_masked = []
indices = np.where((mask_img > 0) & (parcels == 61))

for s in range(len(subjs)):
    # Load subjects nifti and motion data then clean (run1)
    print("Loading Run1 BOLD subj num: " + str(s+1))
    run1 = nib.load(datadir + 'subjects/' + subjs[s] + '/analysis/run1.feat/trans_filtered_func_data.nii').get_data()[:,:,:,0:2511]
    print("Loading Run1 Motion Regressors")
    motion_run1 = np.genfromtxt(motion_dir + subjs[s] + '/EPI_mcf1.par')
    print("Cleaning Run1 BOLD Data")
    clean_run1 = stats.zscore(clean_data(run1[indices][:], motion_run1), axis=1, ddof=1)
    run1_masked.append(run1[indices][:])
   
    # Load subjects nifti and motion data then clean (run2)
    print("Loading Run2 BOLD subj num: " + str(s+1)) 
    run2 = nib.load(datadir + 'subjects/' + subjs[s] + '/analysis/run2.feat/trans_filtered_func_data.nii').get_data()[:,:,:,0:2511]
    print("Loading Run2 Motion Regressors")
    motion_run2 = np.genfromtxt(motion_dir + subjs[s] + '/EPI_mcf2.par')
    print("Cleaning Run2 BOLD Data")
    clean_run2 = stats.zscore(clean_data(run2[indices][:], motion_run2), axis=1, ddof=1)
    run2_masked.append(run2[indices][:])

run1_SRM, run2_SRM = srm(run1_masked,run2_masked)

data_run1 = run1_SRM[:,start_idx_run1:end_idx_run1]
data_run2 = run2_SRM[:,start_idx_run2:end_idx_run2]

data = (data_run1 + data_run2) / 2

print("Fitting HMM")
SL_match = HMM(data,human_bounds)
x=10
