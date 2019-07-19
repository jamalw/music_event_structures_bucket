import csv
import os, fnmatch
import numpy as np
import pandas

subjdir = '/jukebox/norman/jamalw/MES/subjects/'
datadir = '/data/session_info/'
fsuffix = '_prescan_num1.log'
subj_ids = np.sort(fnmatch.filter(os.listdir(subjdir), 'MES_*'))
ratings_clean = np.empty(48)

for r in range(2):
    # load in logfile
    fsuffix = '_prescan_num' + str(r + 1) + '.log' 
    
    with open(subjdir + subj_ids[0] + datadir + subj_ids[0] + fsuffix, newline='') as f:
        csvread = csv.reader(f)
        batch_data = list(csvread)

    data = [''.join(x) for x in batch_data]

    # first get ratings only from logfile
    ratings = [x for x in data if not ('sync' in x or 'space' in x or 'play' in x or 'Very' in x or 'Give' in x or 'End' in x or 'engaging' in x or 'enjoyable' in x or 'familiar' in x or 'Log' in x)]

    ratings = list(filter(None, ratings))

    for i in range(len(ratings_clean)):
        ratings_clean[i] = ratings[i][-1]

    # next get song names
    songs_clean = []

    songs = [x for x in data if('.wav' in x)]

    for i in range(16):
        songs_clean.append(songs[i].rsplit('/',1)[1])

x = 10
