import numpy as np
import glob as glob

roi = 'rA1'
datadir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/parcel_output/" + roi + '/'
nPerm = 1000

fn = glob.glob(datadir + '*matches.npy')

matches = np.zeros((len(fn),nPerm+1))

for i in range(len(fn)):
    matches[i,:] = np.load(fn[i])

avg_matches = np.mean(matches, axis=0)
avg_pval = (np.sum(avg_matches[1:] <= avg_matches[0]) + 1) / (len(avg_matches[1:]) + 1)


