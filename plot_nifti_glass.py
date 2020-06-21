import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
import os

def plot_data(data,affine,thresh,save,*args):
    minval = np.min(data)
    maxval = np.max(data)
    img = nib.Nifti1Image(data,affine)
    img.header["cal_min"] = minval
    img.header["cal_max"] = maxval
    nib.save(img,'temp_plot_data/temp_plot_data.nii.gz')  
    plotting.plot_glass_brain('temp_plot_data/temp_plot_data.nii.gz', black_bg=True,colorbar=True, threshold = thresh)
    os.remove('temp_plot_data/temp_plot_data.nii.gz')
    if save == 1:
        plt.savefig('plots/' + str(args)) 
 
    plt.show()
