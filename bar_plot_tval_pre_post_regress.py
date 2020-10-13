import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

parcelNum = 300

results_dir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/data/searchlight_output/parcels/Schaefer" + str(parcelNum) + "/ttest_results/"

fn = "tstats_map_both_runs_srm_v1_all_pure_random_split_merge_original_match_score_DMN_A1.nii.gz"
fn_regress = "tstats_map_both_runs_srm_v1_all_pure_random_split_merge_original_match_score_regress_all_features_DMN_A1.nii.gz"

parcels = nib.load("/jukebox/norman/jamalw/MES/data/CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_" + str(parcelNum) + "Parcels_17Networks_order_FSLMNI152_2mm.nii.gz").get_data()

mask_img = nib.load('/jukebox/norman/jamalw/MES/data/mask_nonan.nii')

data = nib.load(results_dir + fn).get_data()
data_reg = nib.load(results_dir + fn_regress).get_data()

parcels_select = [115,140,186,189,119]

# for output matrix rows correspond to ROI and columns correspond to tval before and after regression
out = np.zeros((len(parcels_select),2))

for i in range(len(parcels_select)):
    print("Parcel Num: ", parcels_select[i])
    indices = np.where((mask_img.get_data() > 0) & (parcels == parcels_select[i]))
    out[i,0] = np.mean(data[indices])
    out[i,1] = np.mean(data_reg[indices])

labels = ['lprec', 'lAG','rA1_1','rA1_2','lvmpfc']

fig,ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

rect1 = ax.bar(x - width/2,out[:,0],width,label="before",color='darkgrey')
rect2 = ax.bar(x + width/2,out[:,1],width,label="after",color='tomato')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('t-value',fontsize=15)
ax.set_title('Parcel T-Values Before and After Regression Schaefer 300',fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels,fontsize=15)
ax.legend(loc='upper right')
plt.tight_layout()

plt.savefig('/jukebox/norman/jamalw/MES/prototype/link/scripts/plots/tval_pre_post_regress_schaefer300.png')



