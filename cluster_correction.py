from scipy.ndimage import label, generate_binary_structure
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import surface, datasets, plotting


def plot_data(data,affine,plot_title,thresh):
    minval = np.min(data)
    maxval = np.max(data)
    img = nib.Nifti1Image(data,affine)
    img.header["cal_min"] = minval
    img.header["cal_max"] = maxval
    fsaverage = datasets.fetch_surf_fsaverage()
    texture_right = surface.vol_to_surf(img, fsaverage.pial_right) 
    texture_left = surface.vol_to_surf(img, fsaverage.pial_left)
    plotting.plot_surf_stat_map(fsaverage.infl_right, texture_right, hemi='right', title=plot_title, colorbar=True, threshold=thresh, bg_map=fsaverage.sulc_right)    
    plotting.plot_surf_stat_map(fsaverage.infl_left, texture_left, hemi='left', title=plot_title, threshold=thresh, bg_map=fsaverage.sulc_left)
    plt.show() 

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/HMM_searchlight_human_bounds_fit_to_all/'

mask = nib.load('/jukebox/norman/jamalw/MES/data/mask_nonan.nii')

affine = mask.affine

fn = 'ttest_results_test/pstats_map_both_runs_w_perms.npy'

print("Loading Data")
pvals = np.load(datadir + fn)
max_cluster = np.zeros(1000)
connectivity = 3
s = generate_binary_structure(3,connectivity)

for i in range(0,1000):
    print("Copying Data")
    image = pvals.copy()[:,:,:,i+1]
    # use z = 1.96 for p < 0.05 and z = 1.28 for p < 0.1
    image[pvals[:,:,:,i+1] > 0.05] = 0
    image[pvals[:,:,:,i+1] <= 0.05] = 1
    image[mask.get_data() == 0] = 0
    larray, nf = label(image,s)
    cluster_sizes = np.unique(larray[larray>0], return_counts=True)[1]

    # find SIZE_THRESH such that
    try:
        max_cluster[i] = np.max(cluster_sizes)
    except ValueError:
        pass
    # is false for 95% of null maps

sorted_max_cluster = np.sort(max_cluster)
# use confidence interval of .95 for p < 0.05 and .90 for p < 0.1
thresh = sorted_max_cluster[int(len(sorted_max_cluster)*0.95)]

# run clustering on real pvals[:,:,:,0] to get image, larray, cluster_sizes
# use z=1.28 for p < 0.1 and use z=1.96 for p < 0.05
image = pvals.copy()[:,:,:,0]
image[pvals[:,:,:,0] > 0.05] = 0
image[pvals[:,:,:,0] <= 0.05] = 1
image[mask.get_data() == 0] = 0
larray, nf = label(image,s)
cluster_sizes = np.unique(larray[larray>0], return_counts=True)[1]

invalid_clusters = np.nonzero(cluster_sizes < thresh)[0] + 1
for c in invalid_clusters:
    image[larray == c] = 0

# write image to nifti
minval = np.min(image)
maxval = np.max(image)
img = nib.Nifti1Image(image, affine=affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,datadir + 'cluster_corrected_005.nii.gz')
