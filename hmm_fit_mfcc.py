#import deepdish as dd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import numpy as np
import brainiak.eventseg.event
from scipy.stats import norm, zscore, pearsonr, stats
from scipy.signal import gaussian, convolve
from sklearn import decomposition
import numpy as np
from scipy.signal import spectrogram
from pydub import AudioSegment
from scipy.fftpack import dct
from scipy.io import wavfile

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
ann_dirs = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'
songs_dir = '/jukebox/norman/jamalw/MES/data/songs/'

songs = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

song_titles = ['St Pauls Suite', 'I Love Music', 'Moonlight Sonata', 'Change of the Guard','Waltz of the Flowers','The Bird', 'Island', 'Allegro Moderato', 'Finlandia', 'Early Summer', 'Capriccio Espagnole', 'Symphony Fantastique', 'Boogie Stop Shuffle', 'My Favorite Things', 'Blue Monk','All Blues']


durs = np.array([90,180,180,90,135,180,180,225,225,135,90,135,225,225,90,135]) 

song_bounds = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

for s in range(len(songs)):
    human_bounds = np.load(ann_dirs + songs[s] + '/' + songs[s] + '_beh_seg.npy')

    human_bounds = np.append(0,np.append(human_bounds,durs[s])) 

    # Get start and end of chosen song
    start = song_bounds[s]
    end = song_bounds[s + 1]
    
    # load audio data
    rate, audio = wavfile.read(songs_dir + songs[s] + '.wav')
    audio_array = np.mean(audio,axis=1)
    print('computing spectrogram')
    f,t,spect = spectrogram(audio_array,44100)
    print('spectrogram computed')
 
    w = np.round(spect.shape[1]/(len(audio_array)/44100))
    output = np.zeros((spect.shape[0],int(np.round((t.shape[0]/w)))))
    forward_idx = np.arange(0,len(t) + 1,w)
    num_ceps = 12
 
    for i in range(len(forward_idx)):
        if spect[:,int(forward_idx[i]):int(forward_idx[i])+int(w)].shape[1] != w:
            continue
        else:
            output[:,i] = np.mean(spect[:,int(forward_idx[i]):int(forward_idx[i])+int(w)],axis=1).T

    # compute similarity matrix for mfcc
    mfcc = dct(output.T, type=2,axis=1,norm='ortho')[:,1:(num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    cep_lifter = 12
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift
    mfcc_corr = np.corrcoef(mfcc,mfcc)[mfcc.shape[0]:,:mfcc.shape[0]]

    # compute normalized mfcc similarity matrix
    mfcc_norm = mfcc
    mfcc_norm -= (np.mean(mfcc_norm,axis=0) + 1e-8)
    mfcc_norm_corr = np.corrcoef(mfcc_norm,mfcc_norm)[mfcc_norm.shape[0]:,:mfcc_norm.shape[0]]

    nTR = output.shape[1]

    # fit HMM to MFCC
    ev = brainiak.eventseg.event.EventSegment(len(human_bounds) - 1)
    ev.fit(mfcc)

    bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]

    plt.figure(figsize=(10,10))
    plt.imshow(mfcc_norm_corr)
    plt.colorbar()
    ax = plt.gca()
    bounds_aug = np.concatenate(([0],bounds,[nTR]))
    for i in range(len(bounds_aug)-1):
        rect1 = patches.Rectangle((bounds_aug[i],bounds_aug[i]),bounds_aug[i+1]-bounds_aug[i],bounds_aug[i+1]-bounds_aug[i],linewidth=3,edgecolor='w',facecolor='none',label='Model Fit')
        ax.add_patch(rect1)

    for i in range(len(human_bounds)-1):
        rect2 = patches.Rectangle((human_bounds[i],human_bounds[i]),human_bounds[i+1]-human_bounds[i],human_bounds[i+1]-human_bounds[i],linewidth=3,edgecolor='k',facecolor='none',label='Human Annotations')
        ax.add_patch(rect2)

    plt.title('HMM Fit to ' + song_titles[s] + ' MFCC',fontsize=18,fontweight='bold')
    plt.xlabel('TRs',fontsize=18,fontweight='bold')
    plt.ylabel('TRs',fontsize=18,fontweight='bold')
    plt.legend(handles=[rect1,rect2])

    plt.savefig('data/song_analysis/plots/' + songs[s] + '_human_bounds_model_fit.png')

