import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import surface, datasets, plotting


def plot_data(data,affine,thresh):
    minval = np.min(data)
    maxval = np.max(data)
    img = nib.Nifti1Image(data,affine)
    img.header["cal_min"] = minval
    img.header["cal_max"] = maxval
    
    fsaverage = datasets.fetch_surf_fsaverage()

    texture_right = surface.vol_to_surf(img, fsaverage.pial_right)
    texture_left =  surface.vol_to_surf(img, fsaverage.pial_left)   
 
    # plot right lateral
    plotting.plot_surf_stat_map(fsaverage.infl_right, texture_right, hemi='right', view='lateral', title="right lateral", threshold=thresh, bg_map=fsaverage.sulc_right)
    plt.tight_layout()
   
    # plot left lateral
    plotting.plot_surf_stat_map(fsaverage.infl_left, texture_left, hemi='left', view='lateral', title="left lateral", threshold=thresh, bg_map=fsaverage.sulc_left)
    plt.tight_layout()   

    # plot right medial
    plotting.plot_surf_stat_map(fsaverage.infl_right, texture_right, hemi='right', view='medial', title="right medial", threshold=thresh, bg_map=fsaverage.sulc_right)
    plt.tight_layout()   

    # plot left medial
    plotting.plot_surf_stat_map(fsaverage.infl_left, texture_left, hemi='left', view='medial', title="left medial", threshold=thresh, bg_map=fsaverage.sulc_left)
    plt.tight_layout()   
 
    plt.show()
