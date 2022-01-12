#import deepdish as dd
import matplotlib.pyplot as plt
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
from srm import SRM_V1
from mpl_toolkits.axes_grid1 import make_axes_locatable

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
ann_dirs = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'
roi = bil_A1

for i in range(16):
    song_number = i

    #do_srm = int(sys.argv[2])

    hrf = 0

    # run 1 durations
    durs1 = np.array([225,89,180,134,90,180,134,90,179,135,224,89,224,225,179,134])

    songs1 = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

    print(songs1[song_number])

    # run 1 times
    song_bounds1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

    songs2 = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

    # run 2 times
    song_bounds2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

    human_bounds = np.load(ann_dirs + songs1[song_number] + '/' + songs1[song_number] + '_beh_seg.npy') + hrf

    human_bounds = np.append(0,np.append(human_bounds,durs1[song_number])) 

    start_run1 = song_bounds1[song_number]
    end_run1   = song_bounds1[song_number+1]

    start_run2 = song_bounds2[songs2.index(songs1[song_number])]
    end_run2 = song_bounds2[songs2.index(songs1[song_number]) + 1]

    # Load in data
    run1 = stats.zscore(np.load(datadir + 'fdr_01_' + roi + '_split_merge_no_srm_run1_n25.npy'),axis=1, ddof=1)
    run2 = stats.zscore(np.load(datadir + 'fdr_01_' + roi + '_split_merge_no_srm_run2_n25.npy'), axis=1, ddof=1)

    if do_srm == 0:
        run1DataAvg = np.mean(run1,axis=2)
        run2DataAvg = np.mean(run2,axis=2)

        song1 = run1DataAvg[:,start_run1:end_run1]
        song2 = run2DataAvg[:,start_run2:end_run2]
    #elif do_srm == 1:
        # Convert data into lists where each element is voxels by samples
        #run1_list = []
        #run2_list = []
        #for i in range(0,run1.shape[2]):
            #run1_list.append(run1[:,:,i])
            #run2_list.append(run2[:,:,i])

        #n_iter = 10
        #features = 10

        ## run SRM on ROIs
        #shared_data_test1 = SRM_V1(run2_list,run1_list,features,n_iter)
        #shared_data_test2 = SRM_V1(run1_list,run2_list,features,n_iter)

        #avg_response_test_run1 = sum(shared_data_test1)/len(shared_data_test1)
        #avg_response_test_run2 = sum(shared_data_test2)/len(shared_data_test2)

        song1 = avg_response_test_run1[:,start_run1:end_run1]
        song2 = avg_response_test_run2[:,start_run2:end_run2]

        # perform correlation on all voxels between run1 and run2
        corrs_run1_run2 = np.zeros(song1.shape[0])

        for i in range(song1.shape[0]):
            corrs_run1_run2[i] = np.corrcoef(song1[i,:],song2[i,:])[0][1]


plt.figure(figsize=(12,12))
ax = plt.gca()
im = ax.hist(corrs_run1_run2)

for t in cbar.ax.get_yticklabels():
     t.set_fontsize(31)

# Plot human boundaries
for i in range(len(human_bounds)-1):
    rect2 = patches.Rectangle((human_bounds[i],human_bounds[i]),human_bounds[i+1]-human_bounds[i],human_bounds[i+1]-human_bounds[i],linewidth=5,edgecolor='k',facecolor='none',label='Human Annotations')
    ax.add_patch(rect2)

song_names = ['Finlandia', 'Blue Monk', 'I Love Music','Waltz of Flowers','Capriccio Espagnole','Island','All Blues','St Pauls Suite','Moonlight Sonata','Symphony Fantastique','Allegro Moderato','Change of the Guard','Boogie Stop Shuffle','My Favorite Things','The Bird','Early Summer']

ax.set_xlabel('TRs',fontsize=33)
ax.set_ylabel('TRs',fontsize=33)
plt.setp(ax.get_xticklabels(), fontsize=32,rotation=45)
plt.setp(ax.get_yticklabels(), fontsize=32)

#if do_srm == 0:
#    ax.set_title('bil mPFC', fontsize=45,y=1.015,fontweight='bold')
#    ax.set_title('bil prec ' + song_names[song_number] + ' run ' + str(runNum + 1),fontsize=36,y=1.015)
#    plt.tight_layout()
#    plt.savefig('plots/paper_versions/debug/no_srm_sl/' + roi + '/separate_runs/no_SRM/' + song_names[song_number] + '_' + roi + '_no_SRM_run' + str(runNum + 1))
#elif do_srm == 1:   
#    ax.set_title(roi + ' ' + song_names[song_number] + ' Run ' + str(runNum + 1),fontsize=32,fontweight='bold')    
#    plt.savefig('plots/paper_versions/debug/no_srm_sl/' + roi + '/separate_runs/with_SRM/' + song_names[song_number] + '_' + roi + '_with_SRM_run' + str(runNum + 1))
#
#


