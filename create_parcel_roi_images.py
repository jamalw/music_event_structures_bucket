import numpy as np
import nibabel as nib
from scipy import stats
import glob

parcelNum = 100

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/parcels/Schaefer' + str(parcelNum) + '/'

parcels = nib.load("/jukebox/norman/jamalw/MES/data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_" + str(parcelNum) + "Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

mask = nib.load('/jukebox/norman/jamalw/MES/data/mask_nonan.nii')

# Schaefer 100 DMNa and Auditory
#parcel_idx = np.concatenate([np.arange(10,13),np.arange(38,50),np.arange(61,64),np.arange(89,98)])

#parcel_idx = np.arange(1,101)

# Schaefer 100 DMNa only
#parcel_idx = np.concatenate([np.arange(38,41),np.arange(89,93)])

# Schaefer 100 DMNb only
#parcel_idx = np.concatenate([np.arange(41,48),np.arange(93,96)])

# Schaefer 100 DMNc only
#parcel_idx = np.concatenate([np.arange(48,50),np.arange(96,98)])

# Schaefer 100 mPFC 
parcel_idx = [40, 92]

#Schaefer 200 DMNa and Auditory
#parcel_idx = np.concatenate([np.arange(21,27),np.arange(75,99),np.arange(124,130),np.arange(184,197)])

#Schaefer 200 DMNa only
#parcel_idx = np.concatenate([np.arange(75,83),np.arange(184,190)])

#Schaefer 200 DMNb only
#parcel_idx = np.concatenate([np.arange(83,96),np.arange(190,194)])

#Schaefer 200 DMNc only
#parcel_idx = np.concatenate([np.arange(96,99),np.arange(194,197)])

#Schaefer 200 mPFC only
#parcel_idx = [80,187]

#parcel_idx = np.arange(1,201)

#Schaefer 300 DMNa and Auditory
#parcel_idx = np.concatenate([np.arange(36,45),np.arange(111,146),np.arange(186,193),np.arange(271,294)])

#Schaefer 300 DMNa only
#parcel_idx = np.concatenate([np.arange(111,122),np.arange(271,283)])

#Schaefer 300 DMNb only
#parcel_idx = np.concatenate([np.arange(122,140),np.arange(283,290)])

#Schaefer 300 DMNc only
#parcel_idx = np.concatenate([np.arange(140,146),np.arange(290,294)])

#parcel_idx = np.arange(1,301)

x = []
y = []
z = []

x_single = []
y_single = []
z_single = []

for i in range(len(parcel_idx)):
    mask_x = np.where((mask.get_data() > 0) & (parcels == parcel_idx[i]))[0]
    mask_y = np.where((mask.get_data() > 0) & (parcels == parcel_idx[i]))[1]
    mask_z = np.where((mask.get_data() > 0) & (parcels == parcel_idx[i]))[2]
    x.append(mask_x)
    y.append(mask_y)
    z.append(mask_z)
    # take first set of coordinates from mask for each parcel 
    x_single.append(mask_x[0])
    y_single.append(mask_y[0])
    z_single.append(mask_z[0])

x_stack = np.hstack(x)
y_stack = np.hstack(y)
z_stack = np.hstack(z)

indices = np.array((x_stack,y_stack,z_stack)) 

x_single_stack = np.hstack(x_single)
y_single_stack = np.hstack(y_single)
z_single_stack = np.hstack(z_single)

single_indices = np.array((x_single_stack,y_single_stack,z_single_stack)) 

mask_reshaped = np.reshape(mask.get_data(),(91*109*91))
roi_final3D = np.zeros_like(mask.get_data(),dtype=float)

# Fit data back into whole brain
for i in range(len(parcel_idx)):
    roi_final3D[parcels==parcel_idx[i]] = i+1


maxval = np.max(roi_final3D)
minval = np.min(roi_final3D)
img = nib.Nifti1Image(roi_final3D, affine=mask.affine)
img.header['cal_min'] = minval
img.header['cal_max'] = maxval
nib.save(img,'/jukebox/norman/jamalw/MES/data/Schaefer100_DMNa_mPFC1.nii.gz')
