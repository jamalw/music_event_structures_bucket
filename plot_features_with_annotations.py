import numpy as np
import brainiak.eventseg.event
from scipy.stats import norm,zscore,pearsonr,stats
from nilearn.image import load_img
import sys
from brainiak.funcalign.srm import SRM
import nibabel as nib
import os
from scipy.spatial import distance
from sklearn import linear_model
from srm import SRM_V1, SRM_V2, SRM_V3
import matplotlib.pyplot as plt

def convert(seconds):
    min,sec = divmod(seconds, 60)
    return "%02d:%02d" % (min, sec)

datadir = '/jukebox/norman/jamalw/MES/'
music_feat_dir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_input/'

song_bounds = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

for s in range(len(songs)):
    human_bounds = np.load(datadir + 'prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[s] + '/' + songs[s] + '_beh_seg.npy')

    chroma = np.load(music_feat_dir + 'chromaRun1_no_hrf.npy')[:,song_bounds[s]:song_bounds[s+1]]
    mfcc  = np.load(music_feat_dir + 'mfccRun1_no_hrf.npy')[0:12,song_bounds[s]:song_bounds[s+1]]
    tempo  = np.load(music_feat_dir + 'tempoRun1_12PC_singles_no_hrf.npy')[:,song_bounds[s]:song_bounds[s+1]]

    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(13,9))
    fig.suptitle(songs[s],size=15,fontweight='bold',x=.435) 
    # plotting chroma
    im1 = ax1.imshow(chroma,aspect='auto',origin='lower')
    ax1.set_title('chroma',fontweight='bold')
    ax1.set_xticks([])
    ax1.tick_params(labelsize=12)
    for xc in human_bounds:
        ax1.axvline(x=xc,color='r',linewidth=3)
    fig.colorbar(im1, ax=ax1)   
 
    # plotting mfcc
    im2 = ax2.imshow(mfcc,aspect='auto',origin='lower')
    ax2.set_title('mfcc',fontweight='bold')
    ax2.set_xticks([])
    ax2.tick_params(labelsize=12)
    for xc in human_bounds:
        ax2.axvline(x=xc,color='r',linewidth=3)
    fig.colorbar(im2, ax=ax2)

    # plotting tempo
    im3 = ax3.imshow(tempo,aspect='auto',origin='lower')
    ax3.set_title('tempo 12PC',fontweight='bold')
    ax3.set_xlabel('time (s)',fontsize=12,fontweight='bold')
    ax3.tick_params(labelsize=12)
    for xc in human_bounds:
        ax3.axvline(x=xc,color='r',linewidth=3)
    fig.colorbar(im3, ax=ax3)    

    plt.savefig('plots/music_features/' + songs[s] + '_features_and_bounds.png')
