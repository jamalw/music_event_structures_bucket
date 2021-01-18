import numpy as np
import brainiak.eventseg.event
from scipy.stats import norm,zscore,pearsonr,stats
from nilearn.image import load_img
import sys
from brainiak.funcalign.srm import SRM
import nibabel as nib
import os
from scipy.spatial import distance
from sklearn import linear_model
from srm import SRM_V1, SRM_V2, SRM_V3


subjs = ['MES_022817_0']

srm_k = 30

datadir = '/jukebox/norman/jamalw/MES/'
mask_img = load_img(datadir + 'data/mask_nonan.nii')
mask = mask_img.get_data()
mask_reshape = np.reshape(mask,(91*109*91))

x,y,z = np.mgrid[[slice(dm) for dm in tuple((91,109,91))]]
x = np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2]))
y = np.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2]))
z = np.reshape(z,(z.shape[0]*z.shape[1]*z.shape[2]))
coords = np.vstack((x,y,z)).T
coords_mask = coords[mask_reshape>0]

stride = 5
radius = 5
min_vox = srm_k
numVox_per_SL = []
counter = 0

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_input/'
for x in range(0,np.max(coords, axis=0)[0]+stride,stride):
    for y in range(0,np.max(coords, axis=0)[1]+stride,stride):
       for z in range(0,np.max(coords, axis=0)[2]+stride,stride):
           if not os.path.isfile(datadir + subjs[0] + '/' + str(x) + '_' + str(y) + '_' + str(z) + '.npy'):
               continue
           data = []
           for i in range(len(subjs)):
               subj_data = np.load(datadir + subjs[i] + '/' + str(x) + '_' + str(y) + '_' + str(z) + '.npy')
               subj_regs = np.genfromtxt(datadir + subjs[i] + '/EPI_mcf1.par')
               motion = subj_regs.T
               regr = linear_model.LinearRegression()
               regr.fit(motion[:,0:2511].T,subj_data[:,:,0].T)
               subj_data1 = subj_data[:,:,0] - np.dot(regr.coef_, motion[:,0:2511]) - regr.intercept_[:, np.newaxis] 
               data.append(np.nan_to_num(stats.zscore(subj_data1,axis=1,ddof=1)))
           
           # only run function on searchlights with voxels greater than or equal to min_vox
           if data[0].shape[0] >= min_vox: 
               numVox_per_SL.append(data[0].shape[0])
               #counter = counter + 1
               #print(str(counter)) 

mean_num_voxels_per_sl = np.mean(numVox_per_SL)
std_num_voxels_per_sl = np.std(numVox_per_SL)
