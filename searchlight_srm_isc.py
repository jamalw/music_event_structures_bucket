import numpy as np
from scipy.stats import norm,zscore,pearsonr,stats
from nilearn.image import load_img
import sys
from brainiak.funcalign.srm import SRM
import nibabel as nib
import os
from scipy.spatial import distance
from brainiak.isfc import isc

subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

datadir = '/jukebox/norman/jamalw/MES/'
mask_img = load_img(datadir + 'data/mask_nonan.nii.gz')
mask = mask_img.get_data()
mask_reshape = np.reshape(mask,(91*109*91))

def searchlight(coords,mask,subjs):
    
    """run searchlight 

       Create searchlight object and perform voxel function at each searchlight location
    
       Parameters
       ----------
       coords : voxel by xyz ndarray (2D, Vx3)
       mask   : x x y x z (e.g. 91,109,91)
       subjs  : list of subject IDs        
 
       Returns
       -------
       3D data: brain (or ROI) filled with searchlight function scores (3D)

    """

    stride = 5
    radius = 5
    min_vox = 10
    nPerm = 1000
    SL_allvox = []
    SL_results = []
    datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_input/'
    for x in range(0,np.max(coords, axis=0)[0]+stride,stride):
        for y in range(0,np.max(coords, axis=0)[1]+stride,stride):
           for z in range(0,np.max(coords, axis=0)[2]+stride,stride):
               if not os.path.isfile(datadir + subjs[0] + '/' + str(x) + '_' + str(y) + '_' + str(z) + '.npy'):
                   continue
               D = distance.cdist(coords,np.array([x,y,z]).reshape((1,3)))[:,0]
               SL_vox = D <= radius
               data = []
               for i in range(len(subjs)):
                   subj_data = np.load(datadir + subjs[i] + '/' + str(x) + '_' + str(y) + '_' + str(z) + '.npy')
                   data.append(np.nan_to_num(stats.zscore(subj_data[:,:,0],axis=1,ddof=1)))
               for i in range(len(subjs)):
                   subj_data = np.load(datadir + subjs[i] + '/' + str(x) + '_' + str(y) + '_' + str(z) + '.npy')
                   data.append(np.nan_to_num(stats.zscore(subj_data[:,:,1],axis=1,ddof=1))) 
               print("Running Searchlight")
               SL_isc_mean_results, SL_isc_results = isc_srm(data)
               SL_results.append(SL_isc_mean_results)
               SL_allvox.append(np.array(np.nonzero(SL_vox)[0])) 
    voxmean = np.zeros((coords.shape[0]))
    vox_SLcount = np.zeros(coords.shape[0])
    for sl in range(len(SL_results)):
       voxmean[SL_allvox[sl]] += SL_results[sl]
       vox_SLcount[SL_allvox[sl]] += 1
    voxmean = voxmean / vox_SLcount[:,np.newaxis]
    
    return voxmean

def isc_srm(X):
    
    """perform isc on srm searchlights

       Parameters
       ----------
       X: list of searchlights where searchlights are voxels by time
       
       Returns
       -------
       r: correlations for each searchlights timecourse correlated with all other timecourses      
    """
   
    # train on run 1 and test on run 2 
    run1 = [X[i] for i in np.arange(0, int(len(X)/2))]
    run2 = [X[i] for i in np.arange(int(len(X)/2), len(X))]
    print('Building Model')
    srm = SRM(n_iter=10, features=5)   
    print('Training Model')
    srm.fit(run1)
    print('Testing Model')
    shared_data = srm.transform(run2)
    shared_data = stats.zscore(np.dstack(shared_data),axis=1,ddof=1)
    
    # run isc
    isc_output = isc(shared_data)   
    mean_isc = np.mean(isc_output)
 
    return mean_isc,isc_output


# initialize data stores
global_outputs_all = np.zeros((91,109,91))
results3d = np.zeros((91,109,91))
results3d_real = np.zeros((91,109,91))
results3d_perms = np.zeros((91,109,91,1001))
# create coords matrix
x,y,z = np.mgrid[[slice(dm) for dm in tuple((91,109,91))]]
x = np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2]))
y = np.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2]))
z = np.reshape(z,(z.shape[0]*z.shape[1]*z.shape[2]))
coords = np.vstack((x,y,z)).T 
coords_mask = coords[mask_reshape>0]
print('Running Distribute...')
voxmean = searchlight(coords_mask,mask,subjs) 
results3d[mask>0] = voxmean

print('Saving ' + subj + ' to Searchlight Folder')
#np.save('/scratch/jamalw/HMM_searchlight_K_sweep_srm_bound_match/' + songs[song_idx] +'/raw/globals_loo_' + subj + '_K_raw', results3d_real)


