#%% Imports
from brainiak.eventseg.event import EventSegment
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
#import deepdish as dd
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


def plot_data(data, t, create_fig=True, event_patterns = None):
    if create_fig:
        if event_patterns is not None:
            plt.figure(figsize=(6, 6))
        else:
            plt.figure(figsize=(6, 3))
    if event_patterns is not None:
        plt.figure(1)
    data_z = stats.zscore(data.T, axis=0)
    plt.imshow(data_z, origin='lower',aspect='auto')
    #plt.xlabel('Time (s)',fontsize=16,fontweight='bold')
    #plt.ylabel('Voxels',fontsize=16,fontweight='bold')
    plt.xticks(np.arange(0, 90, 10),fontsize=12)
    plt.xticks(fontsize=18)
    plt.yticks([])
    ax1 = plt.gca()
    ax1.patch.set_edgecolor('black')
    ax1.patch.set_linewidth(10)
    #cb = plt.colorbar()
    #plt.clim(-2,2)
    #cb.ax.tick_params(labelsize=18) 
    plt.tight_layout()

def plot_data_no_labels(data, t, create_fig=True, event_patterns = None):
    if create_fig:
        if event_patterns is not None:
            plt.figure(figsize=(6, 6))
        else:
            plt.figure(figsize=(6, 3))
    if event_patterns is not None:
        plt.figure(1)
    data_z = stats.zscore(data.T, axis=0)
    plt.imshow(data_z, origin='lower',aspect='auto')
    plt.xticks([])
    plt.yticks([])
    ax2 = plt.gca()
    ax2.patch.set_edgecolor('black')
    ax2.patch.set_linewidth(10) 
    plt.tight_layout()

def plot_colorbar(data, t, create_fig=True, event_patterns = None):
    if create_fig:
        if event_patterns is not None:
            plt.figure(figsize=(6, 6))
        else:
            plt.figure(figsize=(6, 3))
    if event_patterns is not None:
        plt.figure(1)
    data_z = stats.zscore(data.T, axis=0)
    im = plt.imshow(data_z, origin='lower',aspect='auto')
    plt.xlabel('Time (s)',fontsize=16,fontweight='bold')
    plt.ylabel('Voxels',fontsize=16,fontweight='bold')
    plt.xticks(np.arange(0, 90, 10),fontsize=12)
    plt.xticks(fontsize=18)
    plt.yticks([])
    ax3 = plt.gca()
    ax3.patch.set_edgecolor('black')
    ax3.patch.set_linewidth(10)
    cb = plt.colorbar(ax=ax3)
    ax3.remove()
    im.set_clim(-2,2)
    cb.ax.tick_params(labelsize=18) 
    plt.tight_layout()

n_subj = 5
event_label_length = 90

#%% Simulation #1
event_labels_1 = np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])

event_labels_2 = np.array([0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5])

event_labels_3 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5])

event_labels_4 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5])

event_labels_mean = np.round((event_labels_1 + event_labels_2 + event_labels_3 + event_labels_4)/n_subj)
event_labels_mean = np.asarray([int(x) for x in event_labels_mean])

data_lst = [event_labels_1, event_labels_2, event_labels_3, event_labels_4, event_labels_mean]

for s in range(n_subj): 
    np.random.seed(s)
    data = generate_data(data_lst[s])
    if s < 4:
        plot_data_no_labels(data, event_label_length)
        plt.savefig('plots/paper_versions/toy_time_series_no_srm_subj_' + str(s) + '.png')
    elif s == 4:
        plot_data_no_labels(data, event_label_length)
        #cb = plt.colorbar()
        #cb.set_clim(-2.0, 2.0)
        #cb.ax.tick_params(labelsize=13)
        plt.savefig('plots/paper_versions/toy_time_series_no_srm_subj_mean.png')
        # save out plot for cropping out x-ticks since the figure sizes won't match up otherwise
        plot_data(data, event_label_length)
        plt.savefig('plots/paper_versions/to_be_used_for_xticks.png')
        plot_colorbar(data, event_label_length)
        plt.savefig('plots/paper_versions/toy_time_series_no_srm_subj_mean_colorbar.png',bbox_inches='tight') 

