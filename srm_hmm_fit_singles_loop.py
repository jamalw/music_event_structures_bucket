from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
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

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
ann_dirs = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

# run 1 durations
durs1 = np.array([225,89,180,134,90,180,134,90,179,135,224,89,224,225,179,134])

songs1 = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

# run 1 times
song_bounds1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

# run 2 durations

durs2 = np.array([90,180,180,90,135,180,180,225,225,135,90,135,225,225,90,135]) 

songs2 = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

# run 2 times
song_bounds2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

hrf = 5
n_iter = 10
features = 10

song_names = ['Finlandia', 'Blue Monk', 'I Love Music','Waltz of Flowers','Capriccio Espagnole','Island','All Blues','St Pauls Suite','Moonlight Sonata','Symphony Fantastique','Allegro Moderato','Change of the Guard','Boogie Stop Shuffle','My Favorite Things','The Bird','Early Summer']

roi = 'A1'

for s in range(len(songs1)):
    human_bounds = np.load(ann_dirs + songs1[s] + '/' + songs1[s] + '_beh_seg.npy')
    human_bounds = np.append(0,np.append(human_bounds,durs1[s])) 
    fig, axs = plt.subplots(1, 2, figsize=(15, 15))
 
    for r in range(2):
        # Load in data
        run1 = np.nan_to_num(stats.zscore(np.load(datadir + roi + '_run1_n25.npy'),axis=1,ddof=1))
        run2 = np.nan_to_num(stats.zscore(np.load(datadir + roi + '_run2_n25.npy'),axis=1,ddof=1))
       
        # Here I set training set equal to run 2 and test to run 1 when r = 0 or where r = test on run 1 and vice versus for run 2  
        if r == 0:
            train = run2
            test  = run1
            start = song_bounds1[s] + hrf
            end   = song_bounds1[s+1] + hrf
        elif r == 1:
            train = run1
            test  = run2
            start = song_bounds2[songs2.index(songs1[s])] + hrf
            end   = song_bounds2[songs2.index(songs1[s]) + 1] + hrf
        
        # Convert data into lists where each element is voxels by samples
        train_list = []
        test_list = []
        for i in range(0,train.shape[2]):
            train_list.append(train[:,:,i])
            test_list.append(test[:,:,i])

        # Initialize model
        print('Building Model')
        srm = SRM(n_iter=n_iter, features=features)

        # Fit model to training data (run 1)
        print('Training Model')
        srm.fit(train_list)

        # Test model on testing data to produce shared response
        print('Testing Model')
        shared_data = srm.transform(test_list)
        shared_data = stats.zscore(np.dstack(shared_data),axis=1,ddof=1)
        data = np.mean(shared_data[:,start:end],axis=2)

        nTR = data.shape[1]

        ev = brainiak.eventseg.event.EventSegment(len(human_bounds) - 1)
        ev.fit(data.T)

        bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]

        ax = axs[r]
        ax.imshow(np.corrcoef(data.T))
        #cbar = fig.colorbar(data)
        ax = plt.gca()
        #cbar.ax.tick_params(labelsize=15)
        #ax.tick_params(labelsize=18)
        bounds_aug = np.concatenate(([0],bounds,[nTR]))
        for i in range(len(bounds_aug)-1):
            rect1 = patches.Rectangle((bounds_aug[i],bounds_aug[i]),bounds_aug[i+1]-bounds_aug[i],bounds_aug[i+1]-bounds_aug[i],linewidth=5,edgecolor='w',facecolor='none',label='Model Fit')
            axs[r].add_patch(rect1)

        for i in range(len(human_bounds)-1):
            rect2 = patches.Rectangle((human_bounds[i],human_bounds[i]),human_bounds[i+1]-human_bounds[i],human_bounds[i+1]-human_bounds[i],linewidth=5,edgecolor='k',facecolor='none',label='Human Annotations')
            axs[r].add_patch(rect2)

        fig.suptitle('HMM Fit to ' + roi + ' for ' + song_names[s],fontsize=18,fontweight='bold')
        axs[r].set_xlabel('TRs',fontsize=18,fontweight='bold')
        axs[r].set_ylabel('TRs',fontsize=18,fontweight='bold')
        plt.legend(handles=[rect1,rect2])
        #plt.colorbar()

    plt.savefig('plots/srm_both_runs/' + song_names[s] + '_' + roi + '_srm_k_' + str(features) + '_iter_' + str(n_iter))

