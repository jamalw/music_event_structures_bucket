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


num_vox = run1.shape[0]
num_subjs = run1.shape[2]

# Convert data into lists where each element is voxels by samples
run1_list = []
run2_list = []
for i in range(0,run1.shape[2]):
    run1_list.append(run1[:,:,i])
    run2_list.append(run2[:,:,i])


# create sample labels
#classical_rep = np.tile(['classical'],durs1[selected_songs[0]])
#jazz_rep = np.tile(['jazz'],durs1[selected_songs[1]])
#song_labels = np.hstack((classical_rep,jazz_rep)) 
#label_encoder = LabelEncoder()
#true_labels = label_encoder.fit_transform(song_labels)

n_iter = 10
features = int(sys.argv[1])

song1 = int(sys.argv[2])
song2 = int(sys.argv[3])
selected_songs = [song1,song2]

if int(sys.argv[2]) == int(sys.argv[3]):
    sys.exit()

# run SRM on ROIs looping over number of features
shared_data_test1 = SRM_V1(run2_list,run1_list,features,n_iter)
shared_data_test2 = SRM_V1(run1_list,run2_list,features,n_iter)

avg_song_across_subjs = []

for s in selected_songs:
    data = []
    # first get start and end time for each song in run 1 
    start_run1 = song_bounds1[s]
    end_run1   = song_bounds1[s+1]
    # get start and end time for same song in run 2
    start_run2 = song_bounds2[songs2.index(songs1[s])]
    end_run2 = song_bounds2[songs2.index(songs1[s]) + 1]

    # loop over each subject and crop out song data, average across runs and then time, and store in primary data matrix
    for p in range(len(shared_data_test1)):
        song_data1 = shared_data_test1[p][:,start_run1:end_run1]
        song_data2 = shared_data_test2[p][:,start_run2:end_run2]
        song_data_both_runs = (song_data1+song_data2)/2
        #data.append(np.mean(song_data_both_runs,axis=1)) 
        data.append(song_data_both_runs)
    avg_song_across_subjs.append(np.mean(np.asarray(data),axis=0))
data_array = np.hstack((avg_song_across_subjs[0],avg_song_across_subjs[1]))
song1_dur = np.zeros(avg_song_across_subjs[0].shape[1])
song2_dur = np.ones(avg_song_across_subjs[1].shape[1])
song_labels = np.hstack((song1_dur,song2_dur))

perms = 10

ll_train = np.zeros((perms))
ll_test = np.zeros((perms))
perc_corr_train = np.zeros((perms))
perc_corr_test = np.zeros((perms))
accTrain = np.zeros((perms))
accTest = np.zeros((perms))

for p in range(perms):
    np.random.seed(p)
    print("perm = ", str(p))

    randInds=np.random.permutation(data_array.shape[1])
    randLabels = song_labels[randInds]
    randData = data_array[:,randInds]

    # take top 90% for training
    testNum = int(data_array.shape[1] * .1)

    trainData = randData[:,testNum:]
    trainLabs = randLabels[testNum:]
    testData = randData[:,:testNum]
    testLabs = randLabels[:testNum]

    #mu1 = np.array([np.mean(avg_song_across_subjs[0][0,:]),np.mean(avg_song_across_subjs[0][1,:])])
    #mu2 = np.array([np.mean(avg_song_across_subjs[1][0,:]),np.mean(avg_song_across_subjs[1][1,:])])
    #cov1 = np.diag(np.array([np.cov(avg_song_across_subjs[0][0,:]),np.cov(avg_song_across_subjs[0][1,:])]))
    #cov2 = np.diag(np.array([np.cov(avg_song_across_subjs[1][0,:]),np.cov(avg_song_across_subjs[1][1,:])]))
    ### Generate data

    # Set means and covariances
    #mu1 = np.array([-1,0])
    #mu2 = np.array([ 2,2])
    #cov1 = np.array([[1,0],[0,10]])
    #cov2 = np.array([[1,0.98],[0.98,1]])*15
    #axes = np.max([cov1,cov2])/2*np.array([-1,1,-1,1])

    # Sample from two Gaussians
    #nsamps1 = 200 # number of samples from first Gaussian
    #nsamps2 = 300 # number of samples from second Gaussian
    #smps1 = np.random.multivariate_normal(mu1, cov1, nsamps1)
    #smps2 = np.random.multivariate_normal(mu2, cov2, nsamps2)

    #smps1 = avg_song_across_subjs[0].T
    #smps2 = avg_song_across_subjs[1].T
    #nsamps1 = smps1.shape[0]
    #nsamps2 = smps2.shape[0]

    # Now plot the samples
    #fig, [ax0,ax1] = plt.subplots(1,2, figsize=(12,6))
    #ax0.plot(smps1[:,0],smps1[:,1], 'bo',label=songs1[selected_songs[0]])
    #ax0.plot(smps2[:,0],smps2[:,1], 'ro',label=songs1[selected_songs[1]])
    #ax0.axis(axes); ax0.set_aspect('equal')
    #ax0.set_title('raw data (with labels)', fontsize=18);
    #ax0.legend()

    ### Initialize EM
    EMiteration = 1

    # Compress samples into a single set of (unlabelled) samples.
    #smps = np.vstack((smps1,smps2))
    smps = trainData.T
    nsamps = smps.shape[0]

    # Initialize means randomly from observed datapoints
    m1 = smps[np.random.choice(nsamps),:]
    m2 = smps[np.random.choice(nsamps),:]

    # Initialize other params
    v1 = 10*np.eye(features); v2 = 10*np.eye(features) # Set initial variances
    w1 = .5; w2 = .5                     # Set initial mixing weights

    # Plot initialization
    #ax1.clear()
    #ax1.plot(smps[:,0],smps[:,1], 'ko')
    #plot_ellipse(m1, v1, 3, ax1, color='blue', lw=3); plot_ellipse(m1, v1, 3, ax1, color='white', lw=1)
    #plot_ellipse(m2, v2, 3, ax1, color='red', lw=3); plot_ellipse(m2, v2, 3, ax1, color='white', lw=1)
    #ax1.plot(m1[0],m1[1],'bd',markersize=15, markerfacecolor='blue', markeredgewidth=3,markeredgecolor='k')
    #ax1.plot(m2[0],m2[1],'rd',markersize=15, markerfacecolor='red', markeredgewidth=3,markeredgecolor='k')
    #ax1.axis(axes); ax1.set_aspect('equal')
    #ax1.set_title('EM initialization', fontsize=18);

    ### Set animation parameters for algorithm visualization
    pause_duration = 0.5      # how long (in sec) to pause after each step
    manual_progress = False    # set to True to press Enter to progress through each step
    total_iterations = 50    # total number of EM steps for the algorithm to run

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
        #ax1.clear()
        #ax1.plot(smps[p1>p2,0], smps[p1>p2,1], 'bo')
        #ax1.plot(smps[p1<=p2,0],smps[p1<=p2,1], 'ro')
        #plot_ellipse(m1, v1, 3, ax1, color='blue', lw=3); plot_ellipse(m1, v1, 3, ax1, color='white', lw=1)
        #plot_ellipse(m2, v2, 3, ax1, color='red', lw=3); plot_ellipse(m2, v2, 3, ax1, color='white', lw=1)
        #ax1.plot(m1[0],m1[1],'bd',markersize=15, markerfacecolor='blue', markeredgewidth=1,markeredgecolor='k')
        #ax1.plot(m2[0],m2[1],'rd',markersize=15, markerfacecolor='red', markeredgewidth=1,markeredgecolor='k')
        #ax1.axis(axes); ax1.set_aspect('equal')
        #ax1.set_title('EM initialization, E-step ' + str(EMiteration), fontsize=18)
        
        if manual_progress: input("Press Enter to progress to the next M-step")
        plt.pause(pause_duration)
        
        ### M-step: update parameters (means, covariances, and weights)
        m1 = p1@smps/np.sum(p1) # updated mean 1
        m2 = p2@smps/np.sum(p2) # updated mean 2

        v1 = p1*(smps-m1).T @ (smps-m1) / np.sum(p1) # updated cov 1
        v2 = p2*(smps-m2).T @ (smps-m2) / np.sum(p2) # updated cov 2

        v1 = v1 + np.eye(features) * 0.01
        v2 = v2 + np.eye(features) * 0.01

        w1 = np.sum(p1)/nsamps
        w2 = np.sum(p2)/nsamps

        # make plot
        #ax1.clear()
        #ax1.plot(smps[p1>p2,0], smps[p1>p2,1], 'bo')
        #ax1.plot(smps[p1<=p2,0],smps[p1<=p2,1], 'ro')
        #plot_ellipse(m1, v1, 3, ax1, color='blue', lw=3); plot_ellipse(m1, v1, 3, ax1, color='white', lw=1)
        #plot_ellipse(m2, v2, 3, ax1, color='red', lw=3); plot_ellipse(m2, v2, 3, ax1, color='white', lw=1)
        #ax1.plot(m1[0],m1[1],'bd',markersize=15, markerfacecolor='blue', markeredgewidth=1,markeredgecolor='k')
        #ax1.plot(m2[0],m2[1],'rd',markersize=15, markerfacecolor='red', markeredgewidth=1,markeredgecolor='k')
        #ax1.axis(axes); ax1.set_aspect('equal')
        #ax1.set_title('EM initialization, M-step ' + str(EMiteration), fontsize=18)

        if manual_progress: input("Press Enter to progress to the next E-step")
        plt.pause(pause_duration)
        
        EMiteration += 1

    eps = 1e-6
    num1 = w1*multivariate_normal(m1,v1).pdf(smps) + eps
    num2 = w2*multivariate_normal(m2,v2).pdf(smps) + eps
    p1 = num1/(num1+num2)
    p2 = num2/(num1+num2)

    smpsTest = testData.T
    num1Test = w1*multivariate_normal(m1,v1).pdf(smpsTest) + eps
    num2Test = w2*multivariate_normal(m2,v2).pdf(smpsTest) + eps
    p1Test = num1Test/(num1Test+num2Test)
    p2Test = num2Test/(num1Test+num2Test)

    # check which cluster belongs to which class since assignment is arbitrary
    nll1_train = -1 * np.mean(trainLabs * np.log(p1) + (1-trainLabs) * np.log(p2))
    nll2_train = -1 * np.mean(trainLabs * np.log(p2) + (1-trainLabs) * np.log(p1))
    print(nll1_train, nll2_train)
    if nll2_train > nll1_train:
        temp=p2
        p2 = p1
        p1 = temp
        temp=p2Test
        p2Test = p1Test
        p1Test = temp

    nll_train = min(nll1_train,nll2_train)
    nll_test = -1 * np.mean(testLabs * np.log(p2Test) + (1-testLabs) * np.log(p1Test))

    accTrain[p] = np.mean(np.round(p2) == trainLabs)
    accTrain[p] = max(accTrain[p], 0.5)
    accTest[p] = np.mean(np.round(p2Test) == testLabs)
    accTest[p] = max(accTest[p], 0.5)

    print("train: ", str(accTrain))
    print("test: ", str(accTest))

allACC = np.zeros((4,perms))
allACC[0,:] = nll_train
allACC[1,:] = nll_test
allACC[2,:] = accTrain
allACC[3,:] = accTest

np.save('EM_song_data/allACC_' + str(features) + str(song1) + str(song2), allACC )

