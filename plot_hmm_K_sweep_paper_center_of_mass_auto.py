import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/principled/'
smooth_data = np.load(datadir + 'smooth_wva_split_merge_01_lprec_rA1_bil_PHC_bil_mPFC_auto.npy')

smooth_data_temp = np.zeros_like(smooth_data)
smooth_data_temp[:,0,:] = smooth_data[:,1,:]
smooth_data_temp[:,1,:] = smooth_data[:,0,:]
smooth_data_temp[:,2,:] = smooth_data[:,2,:]
smooth_data_temp[:,3,:] = smooth_data[:,3,:]

roi_names = ['rA1', 'lprec', 'bil PHC', 'mPFC']

roi_data_mean = np.zeros((smooth_data.shape[1],smooth_data.shape[0]))

for i in range(roi_data_mean.shape[0]):
    roi_data_mean[i,:] = smooth_data_temp[:,i,:].mean(axis=1) 

#a1 = smooth_data[:,0,:]
#AG = smooth_data[:,1,:]
#prec = smooth_data[:,2,:]
#
#a1_mean = a1.mean(axis=1)
#AG_mean = AG.mean(axis=1)
#prec_mean = prec.mean(axis=1)
#
durs_run1 = np.array([225,90,180,135,90,180,135,90,180,135,225,90,225,225,180,135])

durs_run1_new = durs_run1[:,np.newaxis]

fairK = np.array((3,5,9,15,20,25,30,35,40,45))

event_lengths = durs_run1_new/fairK

unique_event_lengths = np.unique(event_lengths)
x = unique_event_lengths.ravel()

roi_min = np.zeros((roi_data_mean.shape[0],len(unique_event_lengths)))
roi_max = np.zeros((roi_data_mean.shape[0],len(unique_event_lengths)))

#a1_min = np.zeros((len(unique_event_lengths)))
#a1_max = np.zeros((len(unique_event_lengths)))
#
#AG_min  = np.zeros((len(unique_event_lengths)))
#AG_max  = np.zeros((len(unique_event_lengths)))
#
#prec_min  = np.zeros((len(unique_event_lengths)))
#prec_max  = np.zeros((len(unique_event_lengths))) 
#
nBoot = 1000

# compute 95% confidence intervals for each bootstrap sample
for r in range(roi_data_mean.shape[0]):
    for i in range(len(unique_event_lengths)):
        roi_sorted = np.sort(smooth_data[i,r,:])
        roi_min[r,i] = roi_sorted[int(np.round(nBoot*0.025))]
        roi_max[r,i] = roi_sorted[int(np.round(nBoot*0.975))]          

#        a1_sorted = np.sort(a1[i,:])
#        a1_min[i] = a1_sorted[int(np.round(nBoot*0.025))]
#        a1_max[i] = a1_sorted[int(np.round(nBoot*0.975))]     
#
#        AG_sorted = np.sort(AG[i,:])
#        AG_min[i] = AG_sorted[int(np.round(nBoot*0.025))]
#        AG_max[i] = AG_sorted[int(np.round(nBoot*0.975))]     
#
#        prec_sorted = np.sort(prec[i,:])
#        prec_min[i] = prec_sorted[int(np.round(nBoot*0.025))]
#        prec_max[i] = prec_sorted[int(np.round(nBoot*0.975))]     
#

plt.figure(figsize=(10,5))

for p in range(roi_data_mean.shape[0]):
    plt.plot(unique_event_lengths, roi_data_mean[p], label=roi_names[p],linewidth=3)
    plt.fill_between(unique_event_lengths, roi_min[p,:], roi_max[p,:], alpha=0.3)

#plt.plot(unique_event_lengths, a1_mean, color='indigo', label='right auditory cortex',linewidth=3)
#plt.fill_between(unique_event_lengths, a1_min, a1_max,color='indigo',alpha=0.3)
#
#plt.plot(unique_event_lengths, AG_mean, color='magenta', label='left angular gyrus',linewidth=3)
#plt.fill_between(unique_event_lengths, AG_min, AG_max,color='magenta',alpha=0.3)
#
#plt.plot(unique_event_lengths, prec_mean, color='green', label='left precuneus',linewidth=3)
#plt.fill_between(unique_event_lengths, prec_min, prec_max,color='green',alpha=0.3)
#
plt.legend(fontsize=15)

event_lengths_str = ['2','','','3','','','','4','','5','','','','6','','','','','9','10','','12','15','18','20','25','27','30','36','45','60','75']

plt.xticks(unique_event_lengths,event_lengths_str,rotation=45,fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Event Length (s)', fontsize=18,fontweight='bold')
plt.ylabel('Model Fit', fontsize=18,fontweight='bold')
plt.title('Preferred Event Length', fontsize=18,fontweight='bold')
plt.tight_layout()

# compute pvals
prefix = 'A1_peaks_less_than_'
suffix = '_peaks'

# initialize variables for counting peaks less than A1 peak and storing corresponding p-values
roi_vs_a1_peaks = {}
roi_pvals = {}
for i in range(len(roi_names) - 1):
    roi_vs_a1_peaks['A1_peaks_less_than_' + roi_names[i+1] + '_peaks'] = np.zeros((1000))
    roi_pvals['pvals_A1_' + roi_names[i+1]] = np.zeros(())

roi_vs_a1_peaks_names = list(roi_vs_a1_peaks.keys())
roi_pvals_names = list(roi_pvals.keys())

# initialize variable for storing center of mass for each ROI
roi_com = {}
for c in range(len(roi_names)):    
    roi_com[roi_names[c] + '_com'] = np.zeros((1000))

roi_com_names = list(roi_com.keys())

#A1_peaks_less_than_AG_peaks = np.zeros((1000))
#A1_peaks_less_than_prec_peaks = np.zeros((1000))
#
x = np.arange(1,len(unique_event_lengths)+1)

# initialize center of mass storage variables
#A1_com = np.zeros((1000))
#AG_com = np.zeros((1000))
#prec_com = np.zeros((1000))
#

# count A1 peaks less than ROI r peaks
for i in range(1000):
    peak_holder = np.zeros((len(roi_names)))
    for r1 in range(len(roi_names)):
        # compute peak for roi r as weighted average of wva and event length
        peak_holder[r1] = np.sum(x*smooth_data_temp[:,r1,i])/np.sum(smooth_data_temp[:,r1,i])
        # store center of mass in roi r at bootstrap #n
        roi_com[roi_com_names[r1]][i] = peak_holder[r1] 
    # store boolean for A1 peak less than ROI r peak
    for r2 in range(len(roi_names) - 1):
        roi_vs_a1_peaks[roi_vs_a1_peaks_names[r2]][i] = peak_holder[0] < peak_holder[r2+1]
    #A1_peak = np.sum(x*a1[:,i])/np.sum(a1[:,i])
    #AG_peak = np.sum(x*AG[:,i])/np.sum(AG[:,i])
    #prec_peak = np.sum(x*prec[:,i])/np.sum(prec[:,i])
    #A1_peaks_less_than_AG_peaks[i] = A1_peak < AG_peak
    #A1_peaks_less_than_prec_peaks[i] = A1_peak < prec_peak

#for r in range(len(roi_names)):
#    A1_com[i] = A1_peak
#    AG_com[i] = AG_peak
#    prec_com[i] = prec_peak
#

for p in range(len(roi_pvals_names)):
    roi_pvals[roi_pvals_names[p]] = 1-np.sum(roi_vs_a1_peaks[roi_vs_a1_peaks_names[p]])/len(roi_vs_a1_peaks[roi_vs_a1_peaks_names[p]])


#pvals_A1_AG = 1-np.sum(A1_peaks_less_than_AG_peaks)/len(A1_peaks_less_than_AG_peaks)
#pvals_A1_prec = 1-np.sum(A1_peaks_less_than_prec_peaks)/len(A1_peaks_less_than_prec_peaks)

# compute rois preferred event length in seconds (max wva across bootstraps) and center of mass in seconds
pref_event_length_sec = {}
roi_com_mean = {}
for p in range(len(roi_data_mean)):
    pref_event_length_sec[roi_names[p]] = unique_event_lengths[np.argmax(roi_data_mean[p])]
    roi_com_mean[roi_com_names[p]] = np.mean(roi_com[roi_com_names[p]])

#a1_pref = unique_event_lengths[np.argmax(a1_mean)] 
#AG_pref = unique_event_lengths[np.argmax(AG_mean)]
#prec_pref = unique_event_lengths[np.argmax(prec_mean)] 
#
# compute rois preferred event length as average center of mass across bootstraps
#A1_com_mean = np.mean(A1_com)
#AG_com_mean = np.mean(AG_com)
#prec_com_mean = np.mean(prec_com)

# plot vertical lines corresponding to the preferred event length for each ROI 
plt.axvline(A1_com_mean,color='indigo',linewidth=3)
plt.axvline(AG_com_mean,color='magenta',linewidth=3)
plt.axvline(prec_com_mean,color='green',linewidth=3)

plt.savefig('hmm_K_sweep_paper_results/principled/preferred_event_length_split_merge_01_lprec_full.png')


