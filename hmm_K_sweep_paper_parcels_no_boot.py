from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import numpy as np
import brainiak.eventseg.event
from scipy.stats import norm, zscore, pearsonr, stats
from scipy.signal import gaussian, convolve
from sklearn import decomposition
import numpy as np
from brainiak.funcalign.srm import SRM
import sys
from scipy.ndimage.filters import gaussian_filter1d
from statsmodels.nonparametric.kernel_regression import KernelReg


# Custom mean estimator with Fisher z transformation for correlations
def fisher_mean(correlation, axis=None):
    return np.tanh(np.mean(np.arctanh(correlation), axis=axis))

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_input/parcels/Schaefer300/'
ann_dirs = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

K_set = np.array((3,5,9,15,20,25,30,35,40,45))

#########################################################################################
# Here we train and test the model on both runs separately. This will result in two SRM-ified datasets: one for all of run 1 and one for all of run 2. Songs from these datasets will be indexed separately in the following HMM step and then averaged before fitting the HMM.

# run 1 times
song_bounds_run1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

songs_run1 = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

durs_run1 = np.array([225,90,180,135,90,180,135,90,180,135,225,90,225,225,180,135])

# run 2 times
song_bounds_run2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

songs_run2 = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

durs_run2 = np.array([90,180,180,90,135,180,180,225,225,135,90,135,225,225,90,135])

# Load in data and reshape for Schaefer parcellations where the dimensionality is nSubjs X nVox X Time whereas the dimensions for the data used in the original version of the analysis was nVox X Time X nSubjs 
run1 = np.load(datadir + 'parcel118_run1.npy')
run2 = np.load(datadir + 'parcel118_run2.npy')

nSubj = run1.shape[0]

# Convert data into lists where each element is voxels by samples
run1_list = []
run2_list = []
for i in range(0,nSubj):
    run1_list.append(run1[i,:,:])
    run2_list.append(run2[i,:,:])

run1_list = run1_list.copy()
run2_list = run2_list.copy()

wVa_results = np.zeros((16,len(K_set)))


n_iter = 50
features = 10
# Initialize model
print('Building Model')
srm_train_run1 = SRM(n_iter=n_iter, features=features)
srm_train_run2 = SRM(n_iter=n_iter, features=features)

# Fit model to training data
print('Training Model')
srm_train_run1.fit(run1_list)
srm_train_run2.fit(run2_list)

# Test model on testing data to produce shared response
print('Testing Model')
shared_data_train_run1 = srm_train_run1.transform(run2_list)
shared_data_train_run2 = srm_train_run2.transform(run1_list)

avg_response_train_run1 = sum(shared_data_train_run1)/len(shared_data_train_run1)
avg_response_train_run2 = sum(shared_data_train_run2)/len(shared_data_train_run2)

##################################################################################

for i in range(16):
    print('song number ',str(i))
    # grab start and end time for each song from bound vectors. for SRM data trained on run 1 and tested on run 2, use song name from run 1 to find index for song onset in run 2 bound vector 
    start_run1 = song_bounds_run2[songs_run2.index(songs_run1[i])]
    end_run1   = song_bounds_run2[songs_run2.index(songs_run1[i])+1]
    start_run2 = song_bounds_run1[i]
    end_run2   = song_bounds_run1[i+1]
    # chop song from bold data
    data1 = avg_response_train_run1[:,start_run1:end_run1]
    data2 = avg_response_train_run2[:,start_run2:end_run2]
    # average song-specific bold data from each run 
    data = (data1 + data2)/2
    for j in range(len(K_set)):
        # Fit HMM
        ev = brainiak.eventseg.event.EventSegment(int(K_set[j]))
        ev.fit(data.T)
        events = np.argmax(ev.segments_[0],axis=1)
                         
        max_event_length = stats.mode(events)[1][0]
        # compute timepoint by timepoint correlation matrix 
        cc = np.corrcoef(data.T) # Should be a time by time correlation matrix
                 
        # Create a mask to only look at values up to max_event_length
        local_mask = np.zeros(cc.shape, dtype=bool)
        for k in range(1,max_event_length):
            local_mask[np.diag(np.ones(cc.shape[0]-k, dtype=bool), k)] = True
              
            # Compute within vs across boundary correlations
            same_event = events[:,np.newaxis] == events
            within = fisher_mean(cc[same_event*local_mask])
            across = fisher_mean(cc[(~same_event)*local_mask])
            within_across = within - across
            wVa_results[i,j] = within_across

#np.save('/jukebox/norman/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/Schaefer300/lprec118_wva' + str(bootNum), wVa_results)

prec_data_list = []
a1_data_list = []
AG_data_list = []
lvmpfc_data_list = []

# load in each job (20 of which contain 50 bootstraps each which is 1000 boostraps total) for each ROI separately to be converted into one large matrix containing all bootstraps

prec_data_jobNum = np.load(datadir + 'lprec_wva_regress_mfcc' + str(i) + '.npy')
prec_data_list.append(prec_data_jobNum)
a1_data_jobNum = np.load(datadir + 'rA1_wva_regress_mfcc' + str(i) + '.npy')
a1_data_list.append(a1_data_jobNum)
AG_data_jobNum = np.load(datadir + 'lAG_wva_regress_mfcc' + str(i) + '.npy')
AG_data_list.append(AG_data_jobNum)
#lvmpfc_data_jobNum = np.load(datadir + 'lvmpfc' + str(i) + '_no_fisher.npy')
#lvmpfc_data_list.append(lvmpfc_data_jobNum)


prec_data = np.dstack(prec_data_list)
a1_data = np.dstack(a1_data_list)
AG_data = np.dstack(AG_data_list)
#lvmpfc_data = np.dstack(lvmpfc_data_list)

sigma = '5'

durs_run1 = np.array([225,90,180,135,90,180,135,90,180,135,225,90,225,225,180,135])

durs_run1_new = durs_run1[:,np.newaxis]

fairK = np.array((3,5,9,15,20,25,30,35,40,45))

event_lengths = durs_run1_new/fairK

unique_event_lengths = np.unique(event_lengths)
x = event_lengths.ravel()

#ROI_data = [a1_data, AG_data, prec_data, lvmpfc_data]
ROI_data = [a1_data,AG_data,prec_data]

test_x = np.linspace(min(x), max(x), num=100)
smooth_wva = np.zeros((len(unique_event_lengths), len(ROI_data)))


opt_bw = 0

for ROI in range(len(ROI_data)):
    y = ROI_data[ROI][:,:].ravel()
    KR = KernelReg(y,x,var_type='c')
    opt_bw += KR.bw/len(ROI_data)

max_wva = np.zeros(len(ROI_data))
for ROI in range(len(ROI_data)):
    y = ROI_data[ROI][:,:].ravel()
    KR = KernelReg(y,x,var_type='c', bw=opt_bw)
    max_wva[ROI] = np.argmax(KR.fit(test_x)[0])  # Find peak on fine grid
    smooth_wva[:, ROI] += KR.fit(unique_event_lengths)[0]

np.save(datadir + 'smooth_wva_regress_mfcc',smooth_wva)
