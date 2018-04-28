import numpy as np
from scipy.signal import spectrogram, gaussian, convolve
import matplotlib.pyplot as plt
import glob
from scipy.stats import norm, zscore, pearsonr
from pydub import AudioSegment
from scipy.fftpack import dct
import matplotlib.animation as animation
import brainiak.eventseg.event
from sklearn import decomposition
from brainiak.funcalign.srm import SRM
import nibabel as nib
from scipy.io import wavfile
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
song_bounds = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])
songs = ['St Pauls Suite', 'I Love Music', 'Moonlight Sonata', 'Change of the Guard','Waltz of Flowers','The Bird', 'Island', 'Allegro Moderato', 'Finlandia', 'Early Summer', 'Capriccio Espagnole', 'Symphony Fantastique', 'Boogie Stop Shuffle', 'My Favorite Things', 'Blue Monk','All Blues']
songdir = '/jukebox/norman/jamalw/MES/'
all_songs_fn = [songdir + 'data/songs/Change_of_the_Guard.wav']
idx = 3
song = songs[idx]

spects = []
audio_array_holder = []

# load data
FFMPEG_BIN = "ffmpeg"

def update_line(num, line):
    i = X_VALS[num]
    line[0].set_data( [i, i], [Y_MIN, Y_MAX])
    line[1].set_data( [i, i], [Y_MIN, Y_MAX])
    line[2].set_data( [i, i], [Y_MIN, Y_MAX])

    return line 


for j in range(len(all_songs_fn)):
    rate, audio = wavfile.read(all_songs_fn[j])
    audio_array = np.mean(audio,axis=1)
    print('computing spectrogram')     
    f,t,spect = spectrogram(audio_array,44100)
    spects.append(spect)
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

# compute similarity matrix for spectrogram
spect_corr = np.corrcoef(output.T,output.T)[output.shape[1]:,:output.shape[1]]

# compute similarity matrix for mfcc
mfcc = dct(output.T, type=2,axis=1,norm='ortho')[:,1:(num_ceps + 1)]
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
cep_lifter = 12
#lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
#mfcc *= lift
mfcc_corr = np.corrcoef(mfcc,mfcc)[mfcc.shape[0]:,:mfcc.shape[0]]

# compute normalized mfcc similarity matrix
mfcc_norm = mfcc
mfcc_norm -= (np.mean(mfcc_norm,axis=0) + 1e-8)
mfcc_norm_corr = np.corrcoef(mfcc_norm,mfcc_norm)[mfcc_norm.shape[0]:,:mfcc_norm.shape[0]]

comp_spect = (spect_corr + mfcc_corr + mfcc_norm_corr)/3

# Load in data
train_roi_1 = np.nan_to_num(zscore(np.load(datadir + 'A1_run1_n25.npy'),axis=1,ddof=1))
test_roi_1 = np.nan_to_num(zscore(np.load(datadir + 'A1_run2_n25.npy'),axis=1,ddof=1))
train_roi_2 = np.nan_to_num(zscore(np.load(datadir + 'zstats_human_bounds_superior_parietal_tight_run1_n25.npy'),axis=1,ddof=1))
test_roi_2 = np.nan_to_num(zscore(np.load(datadir + 'zstats_human_bounds_superior_parietal_tight_run2_n25.npy'),axis=1,ddof=1))

# Convert data into lists where each element is voxels by samples
train_list_roi_1 = []
test_list_roi_1 = []
train_list_roi_2 = []
test_list_roi_2 = []

for i in range(0,train_roi_1.shape[2]):
    train_list_roi_1.append(train_roi_1[:,:,i])
    test_list_roi_1.append(test_roi_1[:,:,i])

for i in range(0,train_roi_2.shape[2]):
    train_list_roi_2.append(train_roi_2[:,:,i])
    test_list_roi_2.append(test_roi_2[:,:,i])

    
# Initialize models
print('Building Model')
srm_roi_1 = SRM(n_iter=10, features=50)
srm_roi_2 = SRM(n_iter=10, features=10)

# Fit model to training data (run 1)
print('Training Model')
srm_roi_1.fit(train_list_roi_1)
srm_roi_2.fit(train_list_roi_2)

# Test model on testing data to produce shared response
print('Testing Model')
shared_data_roi_1 = srm_roi_1.transform(test_list_roi_1)
shared_data_roi_2 = srm_roi_2.transform(test_list_roi_2)

avg_response_roi_1 = sum(shared_data_roi_1)/len(shared_data_roi_1)
avg_response_roi_2 = sum(shared_data_roi_2)/len(shared_data_roi_2)

X_MIN = 0
X_MAX = spect_corr.shape[0] 
Y_MIN = spect_corr.shape[0]
Y_MAX = 0
X_VALS = range(X_MIN, X_MAX);

fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
fs = 7

# Plot figures
im1 = ax1.imshow(np.corrcoef(avg_response_roi_1[:,song_bounds[idx]:song_bounds[idx+1]].T))
#fig.colorbar(im1,ax=ax1)
ax1.set_title(songs[idx] + ' A1',fontsize=fs)
#ax1.set_xlabel('trs',fontsize=fs,fontweight='bold')
#ax1.set_ylabel('trs',fontsize=fs,fontweight='bold')    
ax1.set_aspect('equal',adjustable='box')
l1 , v1 = ax1.plot(X_MIN, Y_MAX, X_MIN, Y_MIN, linewidth=2, color= 'red')
ax1.set_xlim(X_MIN, X_MAX-2)
ax1.set_ylim(Y_MIN-2, Y_MAX)
ax1.tick_params(axis='both', which='major', labelsize=6)


im2 = ax2.imshow(np.corrcoef(avg_response_roi_2[:,song_bounds[idx]:song_bounds[idx+1]].T))
#fig.colorbar(im2,ax=ax2)
ax2.set_title(songs[idx] + ' Sup Pariet',fontsize=fs)
#ax2.set_xlabel('trs',fontsize=fs,fontweight='bold')
#ax2.set_ylabel('trs',fontsize=fs,fontweight='bold')    
ax2.set_aspect('equal',adjustable='box')
l2 , v2 = ax2.plot(X_MIN, Y_MAX, X_MIN, Y_MIN, linewidth=2, color= 'red')
ax2.set_xlim(X_MIN, X_MAX-2)
ax2.set_ylim(Y_MIN-2, Y_MAX)
ax2.tick_params(axis='both', which='major', labelsize=6)

im3 = ax3.imshow(mfcc_norm_corr)
#fig.colorbar(im3,ax=ax3)
ax3.set_title(songs[idx] + ' MFCC',fontsize=fs)
#ax3.set_xlabel('trs',fontsize=fs,fontweight='bold')
#ax3.set_ylabel('trs',fontsize=fs,fontweight='bold')    
ax3.set_aspect('equal',adjustable='box')
l3 , v3 = ax3.plot(X_MIN, Y_MAX, X_MAX, Y_MIN, linewidth=2, color= 'red')
ax3.set_xlim(X_MIN, X_MAX-2)
ax3.set_ylim(Y_MIN-2, Y_MAX)
ax3.tick_params(axis='both', which='major', labelsize=6)

l = [l1,l2,l3]

line_anim = animation.FuncAnimation(fig, update_line, len(X_VALS),   
                                    fargs=(l, ), interval=100,
                                    blit=True, repeat=False)


FFwriter = animation.FFMpegWriter()
line_anim.save('basic_animation.mp4', writer = FFwriter, fps=30, extra_args=['-vcodec', 'libx264'])
print('video saved')


