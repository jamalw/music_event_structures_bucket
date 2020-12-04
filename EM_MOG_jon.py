import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from time import sleep

# Define a function to draw ellipses that we'll use later
def plot_ellipse(mu, cov, std, ax, **kwargs):
    U,s,Vt = np.linalg.svd(cov)

    theta = np.arange(0,2*np.pi,0.01)
    X = np.vstack((np.cos(theta), np.sin(theta)))
    ellipse = std * U @ np.diag(np.sqrt(s)) @ X + mu[:,None]

    ax.plot(ellipse[0], ellipse[1], **kwargs)

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



