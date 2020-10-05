import numpy as np
import sys
import nibabel as nib
from scipy.spatial import distance
import brainiak.eventseg.event
from scipy.stats import norm,zscore,pearsonr,stats
import os
from sklearn import linear_model
from srm import SRM_V1, SRM_V2, SRM_V3
import scipy.stats as st
import matplotlib.pyplot as plt

idx = int(sys.argv[1])
runNum = int(sys.argv[2])
parcelNum = sys.argv[3]

def save_nifti(data,affine,savedir):
    minval = np.min(data)
    maxval = np.max(data)
    img = nib.Nifti1Image(data,affine)
    img.header["cal_min"] = minval
    img.header["cal_max"] = maxval
    img.header.set_data_dtype(np.float64)
    nib.save(img, savedir)
 
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
    w = 3
    K = len(human_bounds) + 1
    ev = brainiak.eventseg.event.EventSegment(K,split_merge=True,split_merge_proposals=3)
    ev.fit(X.T)
    bounds = np.where(np.diff(np.argmax(ev.segments_[0],axis=1)))[0]
    match = np.zeros(nPerm+1)
    perm_bounds = bounds.copy()
    nTR = X.shape[1] 

    for p in range(nPerm+1):
        #match[p] = sum([np.min(np.abs(perm_bounds - hb)) for hb in human_bounds])
        match[p] = np.sqrt(sum([np.min((perm_bounds - hb)**2) for hb in human_bounds]))
        #for hb in human_bounds:
        #    if np.any(np.abs(perm_bounds - hb) <= w):
        #        match[p] += 1
        #match[p] /= len(human_bounds)
 
        np.random.seed(p)
        perm_bounds = np.random.choice(nTR,K-1,replace=False)

    return match

hrf = 5

if runNum == 0:
    # run 1 times
    song_bounds = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

    songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']


    # get song start time and end time for run 1
    start_idx = song_bounds[idx] 
    end_idx   = song_bounds[idx+1]

elif runNum == 1:
    # run 2 times
    song_bounds = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

    songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

    # get song start time and end time for run 2
    start_idx = song_bounds[idx]
    end_idx   = song_bounds[idx+1]

datadir = '/jukebox/norman/jamalw/MES/'

mask_img = nib.load(datadir + 'data/mask_nonan.nii')

n_iter = 50
srm_k = 30

# load human boundaries
human_bounds = np.load(datadir + 'prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[idx] + '/' + songs[idx] + '_beh_seg.npy') + hrf

parcels = nib.load(datadir + "data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_" + str(parcelNum) + "Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

# create brain-like object to store data into
pvals = np.zeros_like(mask_img.get_data(),dtype=float)
match = np.zeros_like(mask_img.get_data(),dtype=float)
zscores = np.zeros_like(mask_img.get_data(),dtype=float) 

parcel_dir = datadir + "prototype/link/scripts/data/searchlight_input/parcels/Schaefer" + str(parcelNum) + "/"

savedir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/parcels/Schaefer" + str(parcelNum) + "/" + songs[idx]
feature_dir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_input/'


for i in range(int(np.max(parcels))):
    print("Parcel Num: ", str(i+1))
    # get indices where mask and parcels overlap
    indices = np.where((mask_img.get_data() > 0) & (parcels == i + 1))
 
    # initialize list for storing masked data across subjects
    run1 = np.load(parcel_dir + "parcel" + str(i+1) + "_run1.npy")
    run2 = np.load(parcel_dir + "parcel" + str(i+1) + "_run2.npy")

    # here we will regress out music features from bold data
    chroma1 = np.load(feature_dir + 'chromaRun1_hrf.npy')
    chroma2 = np.load(feature_dir + 'chromaRun2_hrf.npy')
    mfcc1   = np.load(feature_dir + 'mfccRun1_hrf.npy')[0:12,:]
    mfcc2   = np.load(feature_dir + 'mfccRun2_hrf.npy')[0:12,:]
    tempo1  = np.load(feature_dir + 'tempoRun1_hrf.npy')[1:,:]
    tempo2  = np.load(feature_dir + 'tempoRun2_hrf.npy')[1:,:]
    fullRegs = np.vstack((chroma1,mfcc1,tempo1))

    run1_regress = []
    run2_regress = []
 
    for s in range(run1.shape[0]):
        # remove music features from run1
        regr1 = linear_model.LinearRegression()
        regr1.fit(fullRegs.T,run1[s,:,:].T)
        run1_regress.append(run1[s,:,:] - np.dot(regr1.coef_, fullRegs) - regr1.intercept_[:, np.newaxis])
        # remove music features from run 2
        regr2 = linear_model.LinearRegression()
        regr2.fit(fullRegs.T,run2[s,:,:].T)
        run2_regress.append(run2[s,:,:] - np.dot(regr2.coef_, fullRegs) - regr2.intercept_[:, np.newaxis])

    # run SRM on masked data
    if runNum == 0:
        shared_data = SRM_V1(run2_regress,run1_regress,srm_k,n_iter)
    elif runNum == 1:
        shared_data = SRM_V1(run1_regress,run2_regress,srm_k,n_iter)

    data = np.mean(stats.zscore(np.dstack(shared_data),axis=1,ddof=1),axis=2)[:,start_idx:end_idx]
    
    # average data without doing SRM
    #if runNum == 0:
    #    data = np.mean(run1,axis=0)[:,start_idx:end_idx]
    #elif runNum == 1:
    #    data = np.mean(run2,axis=0)[:,start_idx:end_idx]      

    # fit HMM to song data and return match data where first entry is true match score and all others are permutation scores
    print("Fitting HMM")
    SL_match = HMM(data,human_bounds)
    
    # compute regular z-score
    #match_z = (SL_match[0] - np.mean(SL_match[1:])) / (np.std(SL_match[1:]))
 
    # compute z-score for euclid by flipping sign after z-scoring
    match_z = ((SL_match[0] - np.mean(SL_match[1:])) / (np.std(SL_match[1:]))) * -1
    
    # convert z-score to p-value
    #match_p =  st.norm.sf(match_z)
 
    # compute p-value
    match_p = (np.sum(SL_match[1:] <= SL_match[0]) + 1) / (len(SL_match))

    # reverse order of numbers such that lower numbers are actually greater
    #SL_match_reverse = 1 - (SL_match - np.min(SL_match))/(np.max(SL_match) - np.min(SL_match))

    # fit match score and pvalue into brain
    pvals[indices] = match_p  
    zscores[indices] = match_z
    match[indices] = SL_match[0]
    

if runNum == 0:
    pfn = savedir + "/pvals_srm_v1_test_run1_pure_random_split_merge_original_match_score_regress_all_features"
    zfn = savedir + "/zscores_srm_v1_test_run1_pure_random_split_merge_original_match_score_regress_all_features"
    mfn = savedir + "/match_scores_srm_v1_test_run1_pure_random_split_merge_original_match_score_regress_all_features"
elif runNum == 1:
    pfn = savedir + "/pvals_srm_v1_test_run2_pure_random_split_merge_original_match_score_regress_all_features"
    zfn = savedir + "/zscores_srm_v1_test_run2_pure_random_split_merge_original_match_score_regress_all_features"
    mfn = savedir + "/match_scores_srm_v1_test_run2_pure_random_split_merge_original_match_score_regress_all_features"


save_nifti(pvals, mask_img.affine, pfn) 
save_nifti(zscores, mask_img.affine, zfn)
save_nifti(match, mask_img.affine, mfn)

