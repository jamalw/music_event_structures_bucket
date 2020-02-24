import numpy as np
import nibabel as nib

datadir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/"

mask = nib.load('/jukebox/norman/jamalw/MES/prototype/link/scripts/mask_nonan.nii.gz').get_data()

# load all data
paper_bound_match = nib.load(datadir + 'HMM_searchlight_human_bounds_fit_to_all/ttest_results/tstats_map_both_runs_no_motion.nii.gz').get_data()

shuffle_event_lengths_discrete = nib.load(datadir + 'HMM_searchlight_bound_match_shuffle_event_lengths/ttest_results/tstats_map_both_runs.nii.gz').get_data()

shuffle_event_lengths_earth = nib.load(datadir + 'HMM_searchlight_bound_match_shuffle_event_lengths_earth/ttest_results/tstats_map_both_runs.nii.gz').get_data()

# flatten data and grab only voxels > zero
A = paper_bound_match[mask != 0]
B = shuffle_event_lengths_discrete[mask != 0]
C = shuffle_event_lengths_earth[mask != 0]

# correlate paper bound match results (random bounds) with shufffle event lengths discrete match tmap

rand_vs_shuffle = np.corrcoef(A,B)

rand_vs_earth = np.corrcoef(A,C)


