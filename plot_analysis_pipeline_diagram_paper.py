#%% Imports
from brainiak.eventseg.event import EventSegment
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import deepdish as dd
import numpy as np
from scipy import stats
import nibabel as nib
import sys

# code for toy srm plot
def generate_data(event_labels, noise_sigma=0.1):
    n_events = np.max(event_labels) + 1
    n_voxels = 10
    event_patterns = np.random.rand(n_events, 10)
    data = np.zeros((len(event_labels), n_voxels))
    for t in range(len(event_labels)):
        data[t, :] = event_patterns[event_labels[t], :] +\
                     noise_sigma * np.random.rand(n_voxels)
    return data


def plot_data(data, t, prob=None, event_patterns=None, create_fig=True):
    if create_fig:
        if event_patterns is not None:
            plt.figure(figsize=(6, 6))
        else:
            plt.figure(figsize=(6, 3))
    if event_patterns is not None:
        plt.figure(1)
    data_z = stats.zscore(data.T, axis=0)
    plt.imshow(data_z, origin='lower',aspect='auto')
    plt.xlabel('Time (s)',fontsize=16,fontweight='bold')
    plt.ylabel('Voxels',fontsize=16,fontweight='bold')
    plt.xticks(np.arange(0, 90, 10),fontsize=12)
    plt.xticks(fontsize=18)
    plt.yticks([])
    if prob is not None:
        plt.plot(9.5*prob/np.max(prob), color='k')

    if event_patterns is not None:
        plt.subplot(2,1,2)
        plt.imshow(stats.zscore(event_patterns, axis=0),
        	       origin='lower')
        plt.xlabel('Events')
        plt.ylabel('Voxels')
        n_ev = event_patterns.shape[1]
        plt.xticks(np.arange(0, n_ev),
                   [str(i) for i in range(1, n_ev+1)])
        plt.yticks([])
        plt.clim(data_z.min(), data_z.max())


#%% Simulation #1
event_labels = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])
t = len(event_labels)
np.random.seed(0)
data = generate_data(event_labels)
plot_data(data,t)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=20)
plt.title("Average subjects' data",fontsize=20,fontweight='bold')
plt.tight_layout()
#plt.show()
#plt.savefig('plots/paper_versions/toy_time_series_black_neuro_no_srm.png')
plt.figure(2)
corr = np.corrcoef(data)
plt.imshow(corr)
plt.xlabel('time (s)',fontsize=23,fontweight='bold')
plt.ylabel('time (s)',fontsize=23,fontweight='bold')
plt.colorbar()
plt.tight_layout()
plt.savefig('plots/toy_corrmat_black_neuro_no_srm.png') 

# code for example subjects fmri voxel by time plots
datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'

song_number = int(sys.argv[1])

# run 1 durations
durs1 = np.array([225,89,180,134,90,180,134,90,179,135,224,89,224,225,179,134])

songs1 = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

# run 1 times
song_bounds1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

songs2 = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

# run 2 times
song_bounds2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

# Load in data
run1 = np.nan_to_num(stats.zscore(np.load(datadir + 'fdr_01_lprec_split_merge_run1_n25.npy'),axis=1,ddof=1))
run2 = np.nan_to_num(stats.zscore(np.load(datadir + 'fdr_01_lprec_split_merge_run2_n25.npy'),axis=1,ddof=1))

subj1 = run2[0:40,0:50,0]
subj2 = run2[0:40,0:50,1]
subj3 = run2[0:40,0:50,2]
subj4 = run2[0:40,0:50,3]

avg_subjs = np.mean([subj1, subj2, subj3, subj4],axis=0)

plt.figure(3)
ax1 = plt.gca()
ax1.imshow(subj1)
plt.xticks([],[])
plt.yticks([],[])
ax1.patch.set_edgecolor('black')  
ax1.patch.set_linewidth('5')  
plt.tight_layout()
plt.savefig('plots/paper_versions/example_subj1.png',bbox_inches='tight',pad_inches = 0)

plt.figure(4)
ax2 = plt.gca()
ax2.imshow(subj2)
plt.xticks([],[])
plt.yticks([],[])
plt.tight_layout()
ax2.patch.set_edgecolor('black')
ax2.patch.set_linewidth('5')
plt.savefig('plots/paper_versions/example_subj2.png',bbox_inches='tight',pad_inches = 0)

plt.figure(5)
ax3 = plt.gca()
ax3.imshow(subj3)
plt.xticks([],[])
plt.yticks([],[])
ax3.patch.set_edgecolor('black')
ax3.patch.set_linewidth('5')
plt.tight_layout()
plt.savefig('plots/paper_versions/example_subj3.png',bbox_inches='tight',pad_inches = 0)

plt.figure(6)
ax4 = plt.gca()
ax4.imshow(subj4)
plt.xticks([],[])
plt.yticks([],[])
ax4.patch.set_edgecolor('black')
ax4.patch.set_linewidth('5')
plt.tight_layout()
plt.savefig('plots/paper_versions/example_subj4.png',bbox_inches='tight',pad_inches = 0)

plt.figure(7)
ax5 = plt.gca()
ax5.imshow(avg_subjs)
plt.xticks([],[])
plt.yticks([],[])
ax5.patch.set_edgecolor('black')
ax5.patch.set_linewidth('5')
plt.tight_layout()
plt.savefig('plots/paper_versions/example_avg_subjs.png',bbox_inches='tight',pad_inches = 0)


#cb = plt.colorbar()
#cb.ax.tick_params(labelsize=15)
#plt.title("song-specific fMRI \n timecourses",fontsize=20,fontweight='bold')
#plt.xlabel('Time (s)',fontsize=20,fontweight='bold')
#plt.ylabel('Voxels', fontsize=20,fontweight='bold')
#
