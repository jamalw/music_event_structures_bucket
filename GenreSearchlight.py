# This script computes the pattern similarity within and between genres (classical and jazz) at every voxel in the brain via a Brainiak searchlight.

# Author: Jamal Williams
# Princeton Neuroscience Institute, Princeton University 2017

import numpy as np
from nilearn.image import load_img
import sys
from brainiak.searchlight.searchlight import Searchlight
from scipy import stats

# Take subject ID as input
subj = sys.argv[1]

datadir = '/jukebox/norman/jamalw/MES/'

# Load functional data and mask data
data1 = load_img(datadir + 'subjects/' + subj + '/data/avg_reorder1.nii')
data2 = load_img(datadir + 'subjects/' + subj + '/data/avg_reorder2.nii')
mask_img = load_img(datadir + 'data/MNI152_T1_2mm_brain_mask.nii')
data1 = data1.get_data()
data2 = data2.get_data()
mask_img = mask_img.get_data()

# Flatten data, then zscore data, then reshape data back into MNI coordinate space
data1 = stats.zscore(np.reshape(data1,(91*109*91,16)))
data1 = np.reshape(data1,(91,109,91,16))
data2 = stats.zscore(np.reshape(data2,(91*109*91,16)))
data2 = np.reshape(data2,(91,109,91,16))

np.seterr(divide='ignore',invalid='ignore')

# Definte function that takes the difference between within vs. between genre comparisons
def corr2_coeff(AB,msk,myrad,bcast_var):
    if not np.all(msk):
        return None
    A,B = (AB[0], AB[1])
    A = A.reshape((-1,A.shape[-1]))
    B = B.reshape((-1,B.shape[-1]))
    corrAB = np.corrcoef(A.T,B.T)[16:,:16]
    classical_within  = np.mean(corrAB[0:8,0:8])
    jazz_within       = np.mean(corrAB[8:16,8:16])
    classJazz_between = np.mean(corrAB[8:16,0:8])
    jazzClass_between = np.mean(corrAB[0:8,8:16])
    within_genre =  np.mean([classical_within,jazz_within])
    between_genre = np.mean([classJazz_between,jazzClass_between])
    diff = within_genre - between_genre
    print("SL_Scores")
    print(diff)
    return diff

# Create and run searchlight
sl = Searchlight(sl_rad=2,max_blk_edge=5)
sl.distribute([data1,data2],mask_img)
sl.broadcast(None)
print('Running Searchlight...')
global_outputs = sl.run_searchlight(corr2_coeff)

# Plot and save searchlight results
maxval = np.max(global_outputs[np.not_equal(global_outputs,None)])
minval = np.min(global_outputs[np.not_equal(global_outputs,None)])
global_outputs = np.array(global_outputs, dtype=np.float)
global_nonans = global_outputs[np.not_equal(global_outputs,None)]
global_nonans = np.reshape(global_nonans,(91,109,91))
np.save(datadir + 'subjects/' + subj + '/data/genre_mat',global_nonans)

print('Searchlight is Complete!')

import matplotlib.pyplot as plt
for (cnt, img) in enumerate(global_outputs):
  plt.imshow(img,vmin=minval,vmax=maxval)
  plt.colorbar()
  plt.savefig(datadir + 'searchlight_images/' + 'img' + str(cnt) + '.png')
  plt.clf()


