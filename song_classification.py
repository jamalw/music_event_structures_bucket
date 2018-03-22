import numpy as np
import nibabel as nib
import numpy.ma as ma


subjs = ['MES_022817_0','MES_030217_0','MES_032117_1','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0','MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

#subjs = ['MES_022817_0','MES_030217_0','MES_032117_1']

datadir = '/jukebox/norman/jamalw/MES/'
mask_fn = 'janice_pvals_mask.nii.gz'
mask = datadir + 'data/' + mask_fn
mask = nib.load(mask).get_data()
masked_data = []

for i in subjs:
    run1 = datadir + 'subjects/' + i + '/data/avg_reorder1.nii'
    run2 = datadir + 'subjects/' + i + '/data/avg_reorder2.nii'
    run1 = nib.load(run1).get_data()
    run2 = nib.load(run2).get_data()
    run1_masked = run1[mask == 1]
    run2_masked = run2[mask == 1]
    data_msk = (run1_masked + run2_masked)/2
    masked_data.append(data_msk)
    
#masked_data = np.reshape(np.asarray(masked_data),(data_msk.shape[0],16,len(subjs)))

corrAB_holder = []
correct_songs = np.zeros((25,16))

for i in range(len(subjs)):
    loo = masked_data[i]
    mask_idx = ma.masked_where(np.arange(len(masked_data)) != i,np.arange(len(masked_data))).mask 
    idx = np.where(mask_idx)[0]
    others = masked_data[idx[0]:idx[-1]]
    others_avg = sum(others)/len(others)
    corrAB = np.corrcoef(loo.T,others_avg.T)[16:,:16]
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
     
