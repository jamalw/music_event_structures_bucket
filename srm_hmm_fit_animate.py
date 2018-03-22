from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import numpy as np
import brainiak.eventseg.event
from scipy.stats import zscore, pearsonr, stats
from scipy.signal import spectrogram, gaussian, convolve
from sklearn import decomposition
import numpy as np
from brainiak.funcalign.srm import SRM
import nibabel as nib
import matplotlib.animation as animation
from scipy.io import wavfile
from pydub import AudioSegment


datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
songdir = '/jukebox/norman/jamalw/MES/data/songs/'
song_fn = [songdir + 'Early_Summer.wav']

FFMPEG_BIN = "ffmpeg"

def update_line(num, line):
    i = X_VALS[num]
    line[0].set_data( [i,i], [Y_MIN, Y_MAX])

    return line

# Load in song data and compute spectrogram
rate,audio = wavfile.read(song_fn[0])
audio_array = np.mean(audio,axis=1)
print('computing spectrogram')
f,t,spect = spectrogram(audio_array,44100)
print('spectrogram computed')
w = np.round(spect.shape[1]/(len(audio_array)/44100))
output = np.zeros((spect.shape[0],int(np.round((t.shape[0]/w)))))
forward_idx = np.arange(0,len(t) + 1,w)

for i in range(len(forward_idx)):
    if spect[:,int(forward_idx[i]):int(forward_idx[i])+int(w)].shape[1] != w:
        continue
    else:
        output[:,i] = np.mean(spect[:,int(forward_idx[i]):int(forward_idx[i])+int(w)],axis=1).T

spect_corr = np.corrcoef(output.T,output.T)[output.shape[1]:,:output.shape[1]]

# Load in neural data
train = np.nan_to_num(stats.zscore(np.load(datadir + 'vmPFC_point1_run1_n25.npy'),axis=1,ddof=1))
test = np.nan_to_num(stats.zscore(np.load(datadir + 'vmPFC_point1_run2_n25.npy'),axis=1,ddof=1))

# Convert data into lists where each element is voxels by samples
train_list = []
test_list = []
for i in range(0,train.shape[2]):
    train_list.append(train[:,:,i])
    test_list.append(test[:,:,i])

# Initialize model
print('Building Model')
srm = SRM(n_iter=10, features=10)

# Fit model to training data (run 1)
print('Training Model')
srm.fit(train_list)

# Test model on testing data to produce shared response
print('Testing Model')
shared_data = srm.transform(test_list)

avg_response = sum(shared_data)/len(shared_data)
human_bounds = np.cumsum(np.load(datadir + 'songs2Dur.npy'))[:-1]
human_bounds2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

nR = shared_data[0].shape[0]
nTR = shared_data[0][:,1486:1620].shape[1]
nSubj = len(shared_data)

ev = brainiak.eventseg.event.EventSegment(10)
ev.fit(avg_response[:,1486:1620].T)

bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]

X_MIN = 0 
X_MAX = spect_corr.shape[0]
Y_MIN = spect_corr.shape[0]
Y_MAX = 0
X_VALS = range(X_MIN, X_MAX);

fig, ((ax1)) = plt.subplots(1)

# Plot figures
# figure 1
im1 = ax1.imshow(np.corrcoef(avg_response[:,1486:1620].T))
ax1 = plt.gca()
bounds_aug = np.concatenate(([0],bounds,[nTR]))
for i in range(len(bounds_aug)-1):
    rect = patches.Rectangle((bounds_aug[i],bounds_aug[i]),bounds_aug[i+1]-bounds_aug[i],bounds_aug[i+1]-bounds_aug[i],linewidth=2,edgecolor='w',facecolor='none')
    ax1.add_patch(rect)

ax1.set_aspect('equal',adjustable='box')
ax1.set_title('HMM Fit to vmPFC')
ax1.set_xlabel('TRs')
ax1.set_ylabel('TRs')
l1,v1 = ax1.plot(X_MIN,Y_MAX,X_MIN,Y_MIN, linewidth=2, color='red')
ax1.set_xlim(X_MIN,X_MAX-2)
ax1.set_ylim(Y_MIN-2, Y_MAX)
ax1.tick_params(axis='both', which='major', labelsize=6)

l = [l1]

line_anim = animation.FuncAnimation(fig, update_line, len(X_VALS),
                                    fargs=(l, ), interval=100,
                                    blit=True, repeat=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=1800)
line_anim.save('Early_Summer_vmPFC_K9_hrf_6.mp4',writer=writer)
print('video saved')
