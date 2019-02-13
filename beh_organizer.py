import csv
import os, fnmatch
import numpy as np

subjdir = '/jukebox/norman/jamalw/MES/subjects/'
datadir = '/data/session_info/'
fsuffix = '_prescan_num1.log'
subj_ids = np.sort(fnmatch.filter(os.listdir(subjdir), 'MES_*'))

with open(subjdir + subj_ids[0] + datadir + subj_ids[0] + fsuffix, newline='') as f:
    csvread = csv.reader(f)
    batch_data = list(csvread)

data = [''.join(x) for x in batch_data]

data_clean = [x for x in data if not ('sync' in x or 'space' in x or 'play' in x or 'Very' in x or 'Give' in x or 'End' in x)]

data_clean = list(filter(None, data_clean))

x = 10
