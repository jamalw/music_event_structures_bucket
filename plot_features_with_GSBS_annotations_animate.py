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
import time
import matplotlib
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = '/jukebox/pkgs/FFMPEG/4.1.4/ffmpeg' # Add the path of ffmpeg here!!

def update_line(num, line):
    i = X_VALS[num]
    line[0].set_data( [i, i], [Y_MIN, Y_MAX])
    line[1].set_data( [i, i], [Y_MIN, Y_MAX]) 
    line[2].set_data( [i, i], [Y_MIN, Y_MAX]) 

    return line

datadir = '/jukebox/norman/jamalw/MES/'
music_feat_dir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_input/'

song_bounds = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

durs = np.array([225,89,180,134,90,180,134,90,179,135,224,89,224,225,179,134])

songs = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

for s in range(len(songs)):
    human_bounds = np.load(datadir + 'prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/' + songs[s] + '/' + songs[s] + '_beh_seg.npy')

    chroma = np.load(music_feat_dir + 'chromaRun1_no_hrf.npy')[:,song_bounds[s]:song_bounds[s+1]]
    mfcc  = np.load(music_feat_dir + 'mfccRun1_no_hrf.npy')[0:12,song_bounds[s]:song_bounds[s+1]]
    tempo  = np.load(music_feat_dir + 'tempoRun1_12PC_singles_no_hrf.npy')[:,song_bounds[s]:song_bounds[s+1]]

    X_MIN = 0
    X_MAX = durs[s]
    Y_MIN = 0
    Y_MAX = chroma.shape[0]
    X_VALS = range(X_MIN, X_MAX)

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
    l1, v1 = ax1.plot(X_MIN, Y_MAX, X_MIN, Y_MIN, linewidth=2, color='k') 
    ax1.set_xlim(X_MIN, X_MAX-1)
    ax1.set_ylim(Y_MIN, Y_MAX-1)
 
    # plotting mfcc
    im2 = ax2.imshow(mfcc,aspect='auto',origin='lower')
    ax2.set_title('mfcc',fontweight='bold')
    ax2.set_xticks([])
    ax2.tick_params(labelsize=12)
    for xc in human_bounds:
        ax2.axvline(x=xc,color='r',linewidth=3)
    fig.colorbar(im2, ax=ax2)
    l2, v2 = ax2.plot(X_MIN, Y_MAX, X_MIN, Y_MIN, linewidth=2, color='k') 
    ax2.set_xlim(X_MIN, X_MAX-1)
    ax2.set_ylim(Y_MIN, Y_MAX-1)
 

    # plotting tempo
    im3 = ax3.imshow(tempo,aspect='auto',origin='lower')
    ax3.set_title('tempo 12PC',fontweight='bold')
    ax3.set_xlabel('time (s)',fontsize=12,fontweight='bold')
    ax3.tick_params(labelsize=12)
    for xc in human_bounds:
        ax3.axvline(x=xc,color='r',linewidth=3)
    fig.colorbar(im3, ax=ax3)    
    l3, v3 = ax3.plot(X_MIN, Y_MAX, X_MIN, Y_MIN, linewidth=2, color='k') 
    ax3.set_xlim(X_MIN, X_MAX-1)
    ax3.set_ylim(Y_MIN, Y_MAX-1)
 
    l = [l1,l2,l3]

    line_anim = animation.FuncAnimation(fig, update_line, len(X_VALS),
                                    fargs=(l, ), interval=100,
                                    blit=True, repeat=False)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
    line_anim.save('test_video.gif', writer='imagemagick',fps=1)
    print('video saved')
 
    x = 10


    #plt.savefig('plots/music_features/' + songs[s] + '_features_and_bounds.png')
