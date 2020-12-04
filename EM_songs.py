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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import minmax_scale, LabelEncoder, MinMaxScaler
import seaborn as sns
from scipy.stats import multivariate_normal
from time import sleep

# Define a function to draw ellipses that we'll use later
def plot_ellipse(mu, cov, std, ax, **kwargs):
    U,s,Vt = np.linalg.svd(cov)

    theta = np.arange(0,2*np.pi,0.01)
    X = np.vstack((np.cos(theta), np.sin(theta)))
    ellipse = std * U @ np.diag(np.sqrt(s)) @ X + mu[:,None]

    ax.plot(ellipse[0], ellipse[1], **kwargs)

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/chris_dartmouth/data/'
ann_dirs = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_K_sweep_srm/'
    
# run 1 durations
durs1 = np.array([225,89,180,134,90,180,134,90,179,135,224,89,224,225,179,134])

songs1 = ['Finlandia', 'Blue_Monk', 'I_Love_Music','Waltz_of_Flowers','Capriccio_Espagnole','Island','All_Blues','St_Pauls_Suite','Moonlight_Sonata','Symphony_Fantastique','Allegro_Moderato','Change_of_the_Guard','Boogie_Stop_Shuffle','My_Favorite_Things','The_Bird','Early_Summer']

# run 1 times
song_bounds1 = np.array([0,225,314,494,628,718,898,1032,1122,1301,1436,1660,1749,1973, 2198,2377,2511])

songs2 = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

# run 2 times
song_bounds2 = np.array([0,90,270,449,538,672,851,1031,1255,1480,1614,1704,1839,2063,2288,2377,2511])


# Load in data
run1 = np.nan_to_num(stats.zscore(np.load(datadir + 'fdr_01_rA1_split_merge_run1_n25.npy'),axis=1,ddof=1))
run2 = np.nan_to_num(stats.zscore(np.load(datadir + 'fdr_01_rA1_split_merge_run2_n25.npy'),axis=1,ddof=1))


##minmax scaling##

#run1 = np.load(datadir + 'fdr_01_rA1_split_merge_run1_n25.npy')
#run2 = np.load(datadir + 'fdr_01_rA1_split_merge_run2_n25.npy')

#run1_scaled = np.zeros_like(run1)
#run2_scaled = np.zeros_like(run2)

#for s in range(run1.shape[2]):
#    run1_scaled[:,:,s] = minmax_scale(run1[:,:,s],axis=1)
#    run2_scaled[:,:,s] = minmax_scale(run2[:,:,s],axis=1)
###################

num_vox = run1.shape[0]
num_subjs = run1.shape[2]

# Convert data into lists where each element is voxels by samples
run1_list = []
run2_list = []
for i in range(0,run1.shape[2]):
    run1_list.append(run1[:,:,i])
    run2_list.append(run2[:,:,i])

# create true labels 
#genre_labels = np.tile(np.array(['Classical','Jazz','Jazz','Classical','Classical','Classical','Jazz', 'Classical','Classical','Classical','Classical', 'Jazz','Jazz','Jazz','Jazz','Jazz']),25)
selected_songs = [0,13]
 
classical_rep = np.tile(['classical'],durs1[selected_songs[0]])
jazz_rep = np.tile(['jazz'],durs1[selected_songs[1]])
song_labels = np.hstack((classical_rep,jazz_rep)) 

label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(song_labels)

### Generate data

# Set means and covariances
mu1 = np.array([-1,0])
mu2 = np.array([ 2,2])
cov1 = np.array([[1,0],[0,10]])
cov2 = np.array([[1,0.98],[0.98,1]])*15
axes = 15*np.array([-1,1,-1,1])

# Sample from two Gaussians
nsamps1 = 200 # number of samples from first Gaussian
nsamps2 = 300 # number of samples from second Gaussian
smps1 = np.random.multivariate_normal(mu1, cov1, nsamps1)
smps2 = np.random.multivariate_normal(mu2, cov2, nsamps2)

# Now plot the samples
fig, [ax0,ax1] = plt.subplots(1,2, figsize=(12,6))
ax0.plot(smps1[:,0],smps1[:,1], 'bo')
ax0.plot(smps2[:,0],smps2[:,1], 'ro')
ax0.axis(axes); ax0.set_aspect('equal')
ax0.set_title('raw data (with labels)', fontsize=18);

### Initialize EM
EMiteration = 1

# Compress samples into a single set of (unlabelled) samples.
smps = np.vstack((smps1,smps2))
nsamps = nsamps1+nsamps2

# Initialize means randomly from observed datapoints
m1 = smps[np.random.choice(nsamps),:]
m2 = smps[np.random.choice(nsamps),:]

# Initialize other params
v1 = 10*np.eye(2); v2 = 10*np.eye(2) # Set initial variances
w1 = .5; w2 = .5                     # Set initial mixing weights

# Plot initialization
ax1.clear()
ax1.plot(smps[:,0],smps[:,1], 'ko')
plot_ellipse(m1, v1, 3, ax1, color='blue', lw=3); plot_ellipse(m1, v1, 3, ax1, color='white', lw=1)
plot_ellipse(m2, v2, 3, ax1, color='red', lw=3); plot_ellipse(m2, v2, 3, ax1, color='white', lw=1)
ax1.plot(m1[0],m1[1],'bd',markersize=15, markerfacecolor='None', markeredgewidth=3)
ax1.plot(m2[0],m2[1],'rd',markersize=15, markerfacecolor='None', markeredgewidth=3)
ax1.axis(axes); ax1.set_aspect('equal')
ax1.set_title('EM initialization', fontsize=18);

### Set animation parameters for algorithm visualization
pause_duration = 0.5      # how long (in sec) to pause after each step
manual_progress = False    # set to True to press Enter to progress through each step
total_iterations = 10     # total number of EM steps for the algorithm to run

if manual_progress: pause_duration=0.01
while EMiteration <= total_iterations:
    ##### Run single step of EM
    ### E-step : compute soft cluster assignments
    
    # Evaluate numerator (probability that each point came from each cluster)
    num1 = w1*multivariate_normal(m1,v1).pdf(smps)
    num2 = w2*multivariate_normal(m2,v2).pdf(smps)
    
    # normalize probabilities to sum to 1
    p1 = num1/(num1+num2)
    p2 = num2/(num1+num2)
    
    # make plot
    ax1.clear()
    ax1.plot(smps[p1>p2,0], smps[p1>p2,1], 'bo')
    ax1.plot(smps[p1<=p2,0],smps[p1<=p2,1], 'ro')
    plot_ellipse(m1, v1, 3, ax1, color='blue', lw=3); plot_ellipse(m1, v1, 3, ax1, color='white', lw=1)
    plot_ellipse(m2, v2, 3, ax1, color='red', lw=3); plot_ellipse(m2, v2, 3, ax1, color='white', lw=1)
    ax1.plot(m1[0],m1[1],'bd',markersize=15, markerfacecolor='None', markeredgewidth=3)
    ax1.plot(m2[0],m2[1],'rd',markersize=15, markerfacecolor='None', markeredgewidth=3)
    ax1.axis(axes); ax1.set_aspect('equal')
    ax1.set_title('EM initialization, E-step ' + str(EMiteration), fontsize=18)
    
    if manual_progress: input("Press Enter to progress to the next M-step")
    plt.pause(pause_duration)
    
    ### M-step: update parameters (means, covariances, and weights)
    m1 = p1@smps/np.sum(p1) # updated mean 1
    m2 = p2@smps/np.sum(p2) # updated mean 2

    v1 = p1*(smps-m1).T @ (smps-m1) / np.sum(p1) # updated cov 1
    v2 = p2*(smps-m2).T @ (smps-m2) / np.sum(p2) # updated cov 2

    w1 = np.sum(p1)/nsamps
    w2 = np.sum(p2)/nsamps

    # make plot
    ax1.clear()
    ax1.plot(smps[p1>p2,0], smps[p1>p2,1], 'bo')
    ax1.plot(smps[p1<=p2,0],smps[p1<=p2,1], 'ro')
    plot_ellipse(m1, v1, 3, ax1, color='blue', lw=3); plot_ellipse(m1, v1, 3, ax1, color='white', lw=1)
    plot_ellipse(m2, v2, 3, ax1, color='red', lw=3); plot_ellipse(m2, v2, 3, ax1, color='white', lw=1)
    ax1.plot(m1[0],m1[1],'bd',markersize=15, markerfacecolor='None', markeredgewidth=3)
    ax1.plot(m2[0],m2[1],'rd',markersize=15, markerfacecolor='None', markeredgewidth=3)
    ax1.axis(axes); ax1.set_aspect('equal')
    ax1.set_title('EM initialization, M-step ' + str(EMiteration), fontsize=18)

    if manual_progress: input("Press Enter to progress to the next E-step")
    plt.pause(pause_duration)
    
    EMiteration += 1
