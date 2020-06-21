import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

max_perms_fn = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_bound_match_shuffle_event_lengths_rotation/ttest_results/sorted_tfce_max_perms.npy"

true_data_fn = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_bound_match_shuffle_event_lengths_rotation/ttest_results/rawTFCE.nii"

data = np.load(max_perms_fn)

# plot distribution
plt.hist(data)

# compute threshold
thresh = data[int(len(data) * 0.95)]

# grab max TFCE in real data
true_max = np.max(nib.load(true_data_fn).get_data())

# plot 95 percentile line
plt.axvline(x=thresh,color='r',label="95th percentile")

# plot true max
plt.axvline(x=true_max,color='k',label="True Max TFCE Value")

# plot legend
plt.legend()

# label plot
plt.title("Distribution of Max TFCE Values Across Permutations")
plt.xlabel("TFCE Values")

plt.savefig('plots/perm_tfce_shuffle_rotation')
