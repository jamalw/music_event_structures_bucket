import numpy as np
import nibabel as nib
import numpy.ma as ma
from brainiak.funcalign.srm import SRM
from scipy import stats

#subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

subjs = ['MES_022817_0','MES_030217_0','MES_032117_1']

datadir = '/jukebox/norman/jamalw/MES/'
mask_fn = 'janice_pvals_mask.nii.gz'
mask = datadir + 'data/' + mask_fn
mask = nib.load(mask).get_data()
masked_data = []
train = []
test = []
exclude_songs = [1,4,8,14]
songs2Dur = np.load(datadir + 'data/' + 'songs2Dur.npy')
omit_mask_list = []
songs2Dur_omit = np.delete(songs2Dur,exclude_songs) 
include_list = []

# create list of indices where timepoints for songs to be included are equal to one and excluded songs are equal to zero
for i in range(len(songs2Dur)):
    if i in exclude_songs:
        vals = np.zeros((songs2Dur[i]))
    else:
        vals = np.ones((songs2Dur[i]))
    omit_mask_list.append([vals])

omit_mask = np.hstack(omit_mask_list)
omit_mask = omit_mask[0].astype(int)

# create list of indices where timepoints for included songs are equal to a new integer
for i in range(len(songs2Dur_omit)):
    ints = np.ones(songs2Dur[i]) * i
    include_list.append(ints)

include_list = np.hstack(include_list)

# grab first 4 songs from run 1 to train which is equal to first 628 seconds
for i in subjs:
    run1 = datadir + 'subjects/' + i + '/analysis/run1.feat/trans_filtered_func_data.nii'
    run2 = datadir + 'subjects/' + i + '/analysis/run2.feat/trans_filtered_func_data.nii'
    run1 = nib.load(run1).get_data()[:,:,:,0:2511]
    run2 = nib.load(run2).get_data()[:,:,:,0:2511]
    run2 = run2[mask == 1,:]
    train.append(run1[mask == 1,0:628])
    test.append(run2[:,omit_mask == 1])
     
print('Building Model')
srm = SRM(n_iter=10, features=50)
print('Training Model')
srm.fit(train)
print('Testing Model')
shared_data = srm.transform(test)
shared_data = stats.zscore(np.dstack(shared_data),axis=1,ddof=1)

corrAB_holder = []
correct_songs = np.zeros((25,12))

for i in range(len(subjs)):
    loo = shared_data[:,:,i]
    others = np.mean(shared_data[:,:,np.arange(shared_data.shape[-1]) != i],axis=2)
    corrAB = np.corrcoef(loo.T,others.T)[12:,:12]
    corrAB_holder.append(corrAB)
    for j in range(len(corrAB)):
        mask2_idx = ma.masked_where(np.arange(len(corrAB)) != j,np.arange(len(corrAB))).mask
        idx2 = np.where(mask2_idx)[0]
        row = corrAB[j,:]
        row = row[idx2]
        same_song = corrAB[j,j]
        if all(same_song > row):
            print(i)
            correct_songs[i,j] = 1

print(correct_songs)
percent_correct = np.mean(np.sum(correct_songs,axis=1)/16)        
     
