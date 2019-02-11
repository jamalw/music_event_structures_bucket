import numpy as np
import nibabel as nib
import numpy.ma as ma
from scipy.stats import zscore
from isc_standalone import isc
import sys 
import glob

# This script calls the brainiak isc and isfc function to perform full brain isc on music data.

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

datadir = '/jukebox/norman/jamalw/MES/data/single_song_niftis/'

version = 'avg'

song_idx = int(sys.argv[1])

TRs = nib.load(datadir + songs[song_idx] + '/' + version + '/subj1.nii.gz').get_data().shape[3]

fn = glob.glob(datadir + songs[song_idx] + '/' + version + '/*')

data = np.empty((91*109*91,TRs,len(fn)))

# Structure data for brainiak isc function
for i in range(len(fn)):
    subj = nib.load(fn[i]).get_data()
    subj_reshape = np.reshape(subj,(91*109*91,TRs))
    data[:,:,i] = subj_reshape
    print('stored subj: ', i)    
 
iscs = isc(data, pairwise=False, summary_statistic='mean')


print('Saving ISC Results')
save_dir = datadir + songs[song_idx] + '/analysis_results/'
np.save(save_dir + version + '_isc',iscs)
