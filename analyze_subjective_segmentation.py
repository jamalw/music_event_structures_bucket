import pandas as pd
import numpy as np
import csv
import peakutils
import matplotlib.pyplot as plt

songs = ['Pauls','I Love Music','Moonlight Sonata','Change of the Guard','Waltz of the Flowers','The Bird','Island','Allegro Moderato','Finlandia','Early Summer','Capriccio Espagnole','Symphony - Fantastique','Boogie Stop Shuffle','My Favorite Things','Blue Monk','All Blues']

subjs = ['SS_021618_0','SS_021618_1','SS_021718_0','SS_021718_2','SS_021818_0','SS_021818_1','SS_021818_2']

durs = np.array([90,180,180,89,134,179,180,224,225,134,90,135,225,225,89,134])

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/'
# initialize our final output: which is a list where each element is a matrix corresponding to a song. the shape of each matrix is nSubjs x nTRs and contains ones at TRs where there was a button press and zeros everywhere else
allPressTimes = []
bounds = []

for i in range(len(songs)):
    # initialize song specific button press time holder (nSubjs x nTrs)
    songPressTimes = np.array([])
    song_bounds = []
    for j in range(len(subjs)):
        # load in subject data. here i load the full csv file AND timestamps column only into two separate variables.  
        data = pd.read_csv(datadir + 'beh/' + subjs[j] + '/' + subjs[j] + '_subjective_segmentation.csv',na_values=" NaN")
        timestamps = pd.read_csv(datadir + 'beh/' + subjs[j] + '/' + subjs[j] + '_subjective_segmentation.csv',index_col='timestamp')

        # grab song end and start times (with event type and info)
        song_end_and_start = data[(data['info'].str.contains(songs[i],na=False))]

        # store start and end time separately
        start = song_end_and_start['timestamp'].iloc[0]
        end = song_end_and_start['timestamp'].iloc[1]

        # compute number of TRs and initialize subject-specific button press vector (vector of zeros that will be replaced with ones according to timestep index) 
        #nTRs = np.round(end - start)
        nTRs = durs[i]
        subj_press_vector = np.zeros((int(nTRs)))
        # grab full range of rows and colums between start and end time 
        button_presses_full = timestamps[start:end]

        # grab only button press timesteps including song start and end time
        button_presses = np.round(button_presses_full[(button_presses_full['info'].str.contains('j',na=False))].index.values.tolist())

        # compute distances between button presses and sum to get cumulative press time distribution
        distance_btwn_presses = np.cumsum(np.diff(button_presses)[:-1]).astype(int)
        if distance_btwn_presses[-1] == nTRs:
            distance_btwn_presses[-1] = nTRs - 1
        song_bounds.append([distance_btwn_presses]) 
        # replace zeros with ones where presses occurred and store vector in song specific button press time holder
        subj_press_vector[distance_btwn_presses] = 1
        songPressTimes = np.concatenate([songPressTimes,subj_press_vector],axis=0)
    bounds.append(song_bounds)
    allPressTimes.append(np.reshape(songPressTimes,(len(subjs),int(nTRs))))

all_songs_indexes = []

for i in range(len(durs)):
    combined = np.zeros((durs[i],1))
    for t in np.arange(0,len(combined)):
        combined[t] = sum([min(abs(x[0]-(t+1)))<=3 for x in bounds[i]])
    combined = combined.reshape((durs[i]))
    indexes = peakutils.indexes(combined,thres=0.5, min_dist=5)
    all_songs_indexes.append([indexes])

# compute subject specific bound average across songs
subj_bounds = np.zeros((len(subjs),len(durs)))

for s in range(len(subjs)):
    for i in range(len(durs)):
        subj_bounds[s,i] = len(bounds[i][s][0])

subj_bounds_mean = np.mean(subj_bounds,axis=1)
mean_of_mean_bounds = np.mean(subj_bounds_mean)
std_of_mean_bounds = np.std(subj_bounds_mean)

# plot peaks for each song
for i in range(len(all_songs_indexes)):
    peak_vector = np.zeros((durs[i]))
    peak_vector[all_songs_indexes[i][0]] = 1
    plt.figure(i)
    plt.plot(peak_vector)
    plt.title(songs[i],fontsize=18)
    plt.xlabel('time(s)',fontsize=18)
    plt.ylabel('boundary(0 or 1)',fontsize=18)
    plt.tight_layout()

# compute average (and std) number of event boundaries across all songs
num_bounds_total = 0
num_bounds_per_song = np.zeros((len(all_songs_indexes)))
num_songs = len(songs)

for i in range(num_songs):
    num_bounds_total += len(all_songs_indexes[i][0])
    num_bounds_per_song[i] = len(all_songs_indexes[i][0])

avg_num_bounds = num_bounds_total/len(all_songs_indexes)

std_num_bounds = (np.sqrt(np.sum((num_bounds_per_song - avg_num_bounds)**2)))/num_songs 

#out = csv.writer(open("beh_peaks.csv","w"), delimiter=',',quoting=csv.QUOTE_ALL)
#out.writerow(all_songs_indexes)

songs_fn = ['St_Pauls_Suite', 'I_Love_Music', 'Moonlight_Sonata', 'Change_of_the_Guard','Waltz_of_Flowers','The_Bird', 'Island', 'Allegro_Moderato', 'Finlandia', 'Early_Summer', 'Capriccio_Espagnole', 'Symphony_Fantastique', 'Boogie_Stop_Shuffle', 'My_Favorite_Things', 'Blue_Monk','All_Blues']

#for i in range(len(songs_fn)):
#    np.save(datadir + 'searchlight_output/HMM_searchlight_K_sweep_srm/' + songs_fn[i] + '/' + songs_fn[i] + '_beh_seg',all_songs_indexes[i][0])    


