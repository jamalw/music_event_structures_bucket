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
from ffmpy import FFmpeg

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
songdir = '/jukebox/norman/jamalw/MES/data/songs/'
song_fn = [songdir + 'Early_Summer.wav']
ann_dirs = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'

FFMPEG_BIN = "ffmpeg"

def update_line(num, line):
    i = X_VALS[num]
    line[0].set_data( [i,i], [Y_MIN, Y_MAX])

    return line

# set song names and bounds
song_name = 'Change_of_the_Guard'

# run 1 durations
durs1 = np.array([225,89,180,134,90,180,134,90,179,135,224,89,224,225,179,134])

songs1 = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

# run 1 times
song_bounds1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

songs2 = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

# run 2 times
song_bounds2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])

hrf = 0
song_number1 = songs1.index(song_name)
song_number2 = songs2.index(song_name)

human_bounds = np.load(ann_dirs + song_name + '/' + song_name + '_beh_seg.npy') + hrf

human_bounds = np.append(0,np.append(human_bounds,durs1[song_number1]))

# Load in neural data
train = np.nan_to_num(stats.zscore(np.load(datadir + 'fdr_01_bil_mPFC_split_merge_run1_n25.npy'),axis=1,ddof=1))
test = np.nan_to_num(stats.zscore(np.load(datadir + 'fdr_01_bil_mPFC_split_merge_run2_n25.npy'),axis=1,ddof=1))

# Convert data into lists where each element is voxels by samples
train_list = []
test_list = []
for i in range(0,train.shape[2]):
    train_list.append(train[:,:,i])
    test_list.append(test[:,:,i])

# Initialize model
numFeatures = 10
print('Building Model')
srm = SRM(n_iter=10, features=numFeatures)

# Fit model to training data (run 1)
print('Training Model')
srm.fit(train_list)

# Test model on testing data to produce shared response
print('Testing Model')
shared_data = srm.transform(test_list)

avg_response1 = sum(shared_data)/len(shared_data)

avg_response1_crop = avg_response1[:,song_bounds2[song_number2]:song_bounds2[song_number2+1]]

# Initialize model
print('Building Model')
srm = SRM(n_iter=10, features=numFeatures)

# Fit model to test data (run 2)
print('Training Model')
srm.fit(test_list)

# Test model on training data to produce shared response
print('Testing Model')
shared_data = srm.transform(train_list)

avg_response2 = sum(shared_data)/len(shared_data)

avg_response2_crop = avg_response2[:,song_bounds1[song_number1]:song_bounds1[song_number1+1]]

# average shared data splits together
avg_response_combo = (avg_response1_crop+avg_response2_crop)/2

nTR = avg_response_combo.shape[1]

ev = brainiak.eventseg.event.EventSegment(len(human_bounds)-1)
ev.fit(avg_response_combo.T)

bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]

X_MIN = 0 
X_MAX = human_bounds[-1]
Y_MIN = human_bounds[-1]
Y_MAX = 0
X_VALS = range(X_MIN, X_MAX);

fig, ((ax1)) = plt.subplots(1)

# Plot figures
# figure 1
im1 = ax1.imshow(np.corrcoef(avg_response_combo.T))
ax1 = plt.gca()
bounds_aug = np.concatenate(([0],bounds,[nTR]))

# save out human bounds, hmm bounds, and shared data
animation_dir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/plots/IMS_animation/'
np.save(animation_dir + song_name + '_shared_data', avg_response_combo)
np.save(animation_dir + song_name + '_human_bounds',human_bounds)
np.save(animation_dir + song_name + '_hmm_bounds', bounds_aug)

for i in range(len(bounds_aug)-1):
    rect = patches.Rectangle((bounds_aug[i],bounds_aug[i]),bounds_aug[i+1]-bounds_aug[i],bounds_aug[i+1]-bounds_aug[i],linewidth=3,edgecolor='w',facecolor='none')
    ax1.add_patch(rect)

for i in range(len(human_bounds)-1):
    rect2 = patches.Rectangle((human_bounds[i],human_bounds[i]),human_bounds[i+1]-human_bounds[i],human_bounds[i+1]-human_bounds[i],linewidth=3,edgecolor='k',facecolor='none',label='Human Annotations')
    ax1.add_patch(rect2)

ax1.set_aspect('equal',adjustable='box')
ax1.set_title('Human and HMM Fit to mPFC',fontsize=18,fontweight='bold')
ax1.set_xlabel('TRs',fontsize=16,fontweight='bold')
ax1.set_ylabel('TRs',fontsize=16,fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=10)
l1,v1 = ax1.plot(X_MIN,Y_MAX,X_MIN,Y_MIN, linewidth=2, color='red')
ax1.set_xlim(X_MIN,X_MAX-2)
ax1.set_ylim(Y_MIN-2, Y_MAX)
ax1.tick_params(axis='both', which='minor', labelsize=6)

l = [l1]

line_anim = animation.FuncAnimation(fig, update_line, len(X_VALS),
                                    fargs=(l, ), interval=100,
                                    blit=True, repeat=False)

gifName = 'temp_plot_data/' + song_name + '.gif'
mp4Name = 'temp_plot_data/' + song_name + '_temp.mp4'
mp4_musicName = 'temp_plot_data/' + song_name + '_with_music.mp4'

Writer = animation.writers['ffmpeg']
writer = Writer(fps=1, metadata=dict(artist='Me'),bitrate=1800, extra_args=['-vcodec', 'libx264'])
line_anim.save(song_name + '_mPFC_K10_IMS.gif',writer='imagekick',fps=1)

ff_gif_to_mp4 = FFmpeg(inputs={gifName:None},outputs={mp4Name:'-vb 40M -pix_fmt yuv420p -y'})

ff_gif_to_mp4.run()

subprocess.call('rm ' + gifName, shell=True)

subprocess.call('ffmpeg -i ' + mp4Name + ' -i ' + songdir + song_name + '.wav -vcodec copy -map 0:0 -map 1:0 ' + mp4_musicName, shell=True)

subprocess.call('rm ' + mp4Name, shell=True)

print('video saved')
