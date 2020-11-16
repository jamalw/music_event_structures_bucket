import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

datadir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_input/"

run1_tempo = np.load(datadir + "tempoRun1_hrf.npy")
run2_tempo = np.load(datadir + "tempoRun2_hrf.npy")

pca_run1 = PCA(n_components=12)
pca_run2 = PCA(n_components=12)

pca_run1.fit(run1_tempo)
pca_run2.fit(run2_tempo)

np.save(datadir + "tempoRun1_hrf_12PC",pca_run1.components_)
np.save(datadir + "tempoRun2_hrf_12PC",pca_run2.components_)

