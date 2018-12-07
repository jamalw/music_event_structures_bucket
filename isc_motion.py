import numpy as np
import nibabel as nib
import numpy.ma as ma
from brainiak.isc import isc

# This script calls the brainiak isc and isfc function to perform full brain isc on music data.

subjs = ['MES_022817_0','MES_030217_0','MES_040217_0','MES_041117_0','MES_041217_0','MES_041317_0', 'MES_041417_0','MES_041517_0','MES_042017_0','MES_042317_0','MES_042717_0','MES_050317_0','MES_051317_0','MES_051917_0','MES_052017_0','MES_052017_1','MES_052317_0','MES_052517_0','MES_052617_0','MES_052817_0','MES_052817_1','MES_053117_0','MES_060117_0','MES_060117_1']

#MES_032117_1

datadir = '/jukebox/norman/jamalw/MES/subjects/'

data_run1 = np.empty((2511,6,len(subjs)))
data_run2 = np.empty((2511,6,len(subjs)))

# Structure data for brainiak isc function
for i in range(len(subjs)):
    data_run1[:,:,i] = np.loadtxt(datadir + subjs[i] + '/data/nifti/EPI_mcf1.par')[0:2511,:]     
    data_run2[:,:,i] = np.loadtxt(datadir + subjs[i] + '/data/nifti/EPI_mcf2.par')[0:2511,:]     
    
iscs_run1 = isc(data_run1, pairwise=False, summary_statistic='mean')
iscs_run2 = isc(data_run2, pairwise=False, summary_statistic='mean')


#save_dir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/'
#print('Saving ISC Results')
#np.save(save_dir + 'full_brain_ISC_run1',ISC1)
#np.save(save_dir + 'full_brain_ISC_run2',ISC2)
