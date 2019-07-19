import csv
import os, fnmatch
import numpy as np
import pandas

subjdir = '/jukebox/norman/jamalw/MES/subjects/'
datadir = '/data/session_info/'
subj_ids = np.sort(fnmatch.filter(os.listdir(subjdir), 'MES_*'))
ratings_clean = np.empty((2,48))

songs_run1 = []
songs_run2 = []

for run in range(2):
    # load in logfile
    fsuffix = '_prescan_num' + str(run + 1) + '.log' 
    
    with open(subjdir + subj_ids[0] + datadir + subj_ids[0] + fsuffix, newline='') as f:
        csvread = csv.reader(f)
        batch_data = list(csvread)

    data = [''.join(x) for x in batch_data]

    # first get ratings only from logfile
    ratings = [x for x in data if not ('sync' in x or 'space' in x or 'play' in x or 'Very' in x or 'Give' in x or 'End' in x or 'engaging' in x or 'enjoyable' in x or 'familiar' in x or 'Log' in x)]

    ratings = list(filter(None, ratings))

    for i in range(ratings_clean.shape[1]):
        ratings_clean[run,i] = ratings[i][-1]
    
    # next get song names
    songs = [x for x in data if('.wav' in x)]
   
    for i in range(16):
        if run == 0:
            songs_run1.append(songs[i].rsplit('/',1)[1])
        elif run == 1:
            songs_run2.append(songs[i].rsplit('/',1)[1])

# split ratings into 16 (song) chunks of 3 (questions). this makes it easier to pair song names with ratings
ratings1 = np.array_split(np.array(ratings_clean[0,:]),16)
ratings2 = np.array_split(np.array(ratings_clean[1,:]),16)

# format songs plus ratings(spr) as dictionary
spr1 = dict(zip(songs_run1,ratings1))
spr2 = dict(zip(songs_run2,ratings2))


x = 10
