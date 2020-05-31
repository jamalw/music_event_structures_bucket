import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_bound_match_shuffle_event_lengths_rotation/ttest_results/'

tstats = nib.load(datadir + 'tstats_map_both_runs.nii.gz').get_data()
pstats = nib.load(datadir + 'pstats_map_both_runs.nii.gz').get_data()

idx1 = np.where(np.logical_and(tstats>=2, tstats<=3))
idx2 = np.where(np.logical_and(tstats>=3, tstats<=4))
idx3 = np.where(np.logical_and(tstats>=4, tstats<=5))


p1 = pstats[idx1]
p2 = pstats[idx2]
p3 = pstats[idx3]

plt.figure(1)
plt.hist(p1, edgecolor='black', linewidth=1.2)
plt.title('P-Values where T >= 2 and T <= 3')
plt.xlabel('p-values')
plt.ylabel('# of voxels')

plt.figure(2)
plt.hist(p2, edgecolor='black', linewidth=1.2)
plt.title('P-Values where T >= 3 and T <= 4')
plt.xlabel('p-values')
plt.ylabel('# of voxels')

plt.figure(3)
plt.hist(p3, edgecolor='black', linewidth=1.2)
plt.title('P-Values where T >= 4 and T <= 5')
plt.xlabel('p-values')
plt.ylabel('# of voxles')

plt.savefig(datadir + 'tp_2_3.pdf')
plt.savefig(datadir + 'tp_3_4.pdf')
plt.savefig(datadir + 'tp_4_5.pdf')
