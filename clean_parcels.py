import numpy as np
import sys
import nibabel as nib
from scipy.spatial import distance
from scipy.stats import norm,zscore,pearsonr,stats
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


subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

datadir = '/jukebox/norman/jamalw/MES/'
motion_dir = datadir + '/prototype/link/scripts/data/searchlight_input/'

mask_img = nib.load(datadir + 'data/mask_nonan.nii')

# load parcellations
parcels = nib.load(datadir + "data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_200Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()


for i in range(int(np.max(parcels))):
    print("Parcel Num: ", str(i+1))
    # get indices where mask and parcels overlap
    indices = np.where((mask_img.get_data() > 0) & (parcels == i + 1))

    # initialize list for storing masked data across subjects
    run1_masked = []
    run2_masked = []

    for s in range(len(subjs)):
        # Load subjects nifti and motion data then clean (run1)
        print("Loading Run1 BOLD subj num: " + str(s+1))
        run1 = nib.load(datadir + 'subjects/' + subjs[s] + '/analysis/run1.feat/trans_filtered_func_data.nii').get_data()[:,:,:,0:2511]
        print("Loading Run1 Motion Regressors")
        motion_run1 = np.genfromtxt(motion_dir + subjs[s] + '/EPI_mcf1.par')
        print("Cleaning Run1 BOLD Data")
        clean_run1 = stats.zscore(clean_data(run1[indices][:], motion_run1), axis=1, ddof=1)
        run1_masked.append(clean_run1[indices][:])   

        # Load subjects nifti and motion data then clean (run2)
        print("Loading Run2 BOLD subj num: " + str(s+1)) 
        run2 = nib.load(datadir + 'subjects/' + subjs[s] + '/analysis/run2.feat/trans_filtered_func_data.nii').get_data()[:,:,:,0:2511]
        print("Loading Run2 Motion Regressors")
        motion_run2 = np.genfromtxt(motion_dir + subjs[s] + '/EPI_mcf2.par')
        print("Cleaning Run2 BOLD Data")
        clean_run2 = stats.zscore(clean_data(run2[indices][:], motion_run2), axis=1, ddof=1)
        run2_masked.append(clean_run2[indices][:])
            
savedir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_input/parcels/Schaefer200/"

    np.save(savedir + "parcel" + str(i+1) + "_run1", run1_masked)
    np.save(savedir + "parcel" + str(i+1) + "_run2", run2_masked)



