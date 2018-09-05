import numpy as np
import sys
import nibabel as nib
from scipy.spatial import distance

subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

#subjs = ['MES_022817_0']

idx = int(sys.argv[1])
subj = subjs[int(idx)]

datadir = '/jukebox/norman/jamalw/MES/'

mask_img = nib.load(datadir + 'data/mask_nonan.nii.gz').get_data()
mask_reshape = np.reshape(mask_img,(91*109*91))

for s in range(len(subjs)):
    run1 = nib.load(datadir + 'subjects/' + subj + '/analysis/run1.feat/trans_filtered_func_data.nii').get_data()[:,:,:,0:2511]
    run2 = nib.load(datadir + 'subjects/' + subj + '/analysis/run2.feat/trans_filtered_func_data.nii').get_data()[:,:,:,0:2511]
    print('Data Loaded')
    stride = 5
    radius = 10
    min_vox = 10
    SL_allvox = []
    SL_results = []
    x,y,z = np.mgrid[[slice(dm) for dm in run1.shape[0:3]]] 
    x = np.reshape(x,(x.shape[0]*x.shape[1]*x.shape[2]))
    y = np.reshape(y,(y.shape[0]*y.shape[1]*y.shape[2]))
    z = np.reshape(z,(z.shape[0]*z.shape[1]*z.shape[2]))
    coords = np.vstack((x,y,z)).T
    coords = coords[mask_reshape>0]
    for x in range(0,np.max(coords, axis=0)[0]+stride,stride):
        for y in range(0,np.max(coords, axis=0)[1]+stride,stride):
           for z in range(0,np.max(coords, axis=0)[2]+stride,stride):
              D = distance.cdist(coords,np.array([x,y,z]).reshape((1,3)))[:,0]
              SL_vox = D <= radius
              if np.sum(SL_vox) < min_vox:
                 continue
              SL_mask = np.zeros(run1.shape[:-1],dtype=bool)
              SL_mask[mask_img > 0] = SL_vox
              sl_run1 = run1[SL_mask]
              sl_run2 = run2[SL_mask]
              sl = np.dstack((sl_run1,sl_run2))
              np.save(datadir + 'prototype/link/scripts/data/searchlight_input/' + subj + '/' + str(x)+'_'+str(y)+'_'+str(z),sl)  
