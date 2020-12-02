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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import seaborn as sns


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

# create true labels 
genre_labels = np.tile(np.array(['Classical','Jazz','Jazz','Classical','Classical','Classical','Jazz', 'Classical','Classical','Classical','Classical', 'Jazz','Jazz','Jazz','Jazz','Jazz']),25)

label_encoder = LabelEncoder()
true_labels = label_encoder.fit_transform(genre_labels)

# build k-means clustering pipeline
n_clusters = 2

clusterer = Pipeline(
   [
       (
           "kmeans",
           KMeans(
               n_clusters=n_clusters,
               init="k-means++",
               n_init=50,
               max_iter=500,
               random_state=42,
           ),
       ),
   ]
)

pipe = Pipeline(
    [
        ("clusterer",clusterer)
    ]
)

n_iter = 10
#features = np.arange(5,200,5)
features = 5
silhouette_scores = []
ari_scores = []

for n_features in [features]:
    print("running with K = ",n_features)
    # reinitialize primary data matrix for every K
    data = []
    # run SRM on ROIs looping over number of features
    shared_data_test1 = SRM_V1(run2_list,run1_list,n_features,n_iter)
    shared_data_test2 = SRM_V1(run1_list,run2_list,n_features,n_iter)

    for s in range(len(songs1)):
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
            data.append(np.mean(song_data_both_runs,axis=1)) 

    data_array = np.asarray(data).T


    # RUN K-MEANS
    clusterer.fit(data_array.T)

    predicted_labels = pipe["clusterer"]["kmeans"].labels_

    silhouette_coef = silhouette_score(data_array.T,predicted_labels)

    ari = adjusted_rand_score(true_labels,predicted_labels)

    # Add metrics to their lists
    silhouette_scores.append(silhouette_coef)
    ari_scores.append(ari)

#print("silhouette score: ",silhouette_score(data_array.T, predicted_labels))

#print("adjusted rand score: ", adjusted_rand_score(true_labels, predicted_labels))

pcadf = pd.DataFrame(
    data_array.T,
    columns=["component_1", "component_2"],
)

pcadf["predicted_cluster"] = pipe["clusterer"]["kmeans"].labels_
pcadf["true_label"] = label_encoder.inverse_transform(true_labels)

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))

scat = sns.scatterplot(
    "component_1",
    "component_2",
    s=50,
    data=pcadf,
    hue="predicted_cluster",
    style="true_label",
    palette="Set2",
)

scat.set_title(
    "Clustering Genre Data"
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.tight_layout()

plt.show()
