import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/principled/'
smooth_data = np.load(datadir + 'smooth_wva_split_merge_01_lprec_full.npy')

a1 = smooth_data[:,0,:]
AG = smooth_data[:,1,:]
prec = smooth_data[:,2,:]

a1_mean = a1.mean(axis=1)
AG_mean = AG.mean(axis=1)
prec_mean = prec.mean(axis=1)

durs_run1 = np.array([225,90,180,135,90,180,135,90,180,135,225,90,225,225,180,135])

durs_run1_new = durs_run1[:,np.newaxis]

fairK = np.array((3,5,9,15,20,25,30,35,40,45))

event_lengths = durs_run1_new/fairK

unique_event_lengths = np.unique(event_lengths)
x = unique_event_lengths.ravel()

a1_min = np.zeros((len(unique_event_lengths)))
a1_max = np.zeros((len(unique_event_lengths)))

AG_min  = np.zeros((len(unique_event_lengths)))
AG_max  = np.zeros((len(unique_event_lengths)))

prec_min  = np.zeros((len(unique_event_lengths)))
prec_max  = np.zeros((len(unique_event_lengths))) 

nBoot = a1.shape[1]

# compute 95% confidence intervals for each bootstrap sample
for i in range(len(unique_event_lengths)):
    a1_sorted = np.sort(a1[i,:])
    a1_min[i] = a1_sorted[int(np.round(nBoot*0.025))]
    a1_max[i] = a1_sorted[int(np.round(nBoot*0.975))]     

    AG_sorted = np.sort(AG[i,:])
    AG_min[i] = AG_sorted[int(np.round(nBoot*0.025))]
    AG_max[i] = AG_sorted[int(np.round(nBoot*0.975))]     

    prec_sorted = np.sort(prec[i,:])
    prec_min[i] = prec_sorted[int(np.round(nBoot*0.025))]
    prec_max[i] = prec_sorted[int(np.round(nBoot*0.975))]     


plt.figure(figsize=(10,5))

plt.plot(unique_event_lengths, a1_mean, color='indigo', label='right auditory cortex',linewidth=3)
plt.fill_between(unique_event_lengths, a1_min, a1_max,color='indigo',alpha=0.3)

plt.plot(unique_event_lengths, AG_mean, color='magenta', label='left angular gyrus',linewidth=3)
plt.fill_between(unique_event_lengths, AG_min, AG_max,color='magenta',alpha=0.3)

plt.plot(unique_event_lengths, prec_mean, color='green', label='left precuneus',linewidth=3)
plt.fill_between(unique_event_lengths, prec_min, prec_max,color='green',alpha=0.3)

#plt.legend(fontsize=15)

event_lengths_str = ['2','','','3','','','','4','','5','','','','6','','','','','9','10','','12','15','18','20','25','27','30','36','45','60','75']

plt.xticks(unique_event_lengths,event_lengths_str,rotation=45,fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel('Event Length (s)', fontsize=18,fontweight='bold')
plt.ylabel('Model Fit', fontsize=18,fontweight='bold')
plt.title('Preferred Event Length', fontsize=18,fontweight='bold')
plt.tight_layout()

# compute pvals
A1_peaks_less_than_AG_peaks = np.zeros((1000))
A1_peaks_less_than_prec_peaks = np.zeros((1000))
x = np.arange(1,len(unique_event_lengths)+1)

# initialize center of mass storage variables
A1_com = np.zeros((1000))
AG_com = np.zeros((1000))
prec_com = np.zeros((1000))

for i in range(1000):
    A1_peak = np.sum(x*a1[:,i])/np.sum(a1[:,i])
    AG_peak = np.sum(x*AG[:,i])/np.sum(AG[:,i])
    prec_peak = np.sum(x*prec[:,i])/np.sum(prec[:,i])
    A1_com[i] = A1_peak
    AG_com[i] = AG_peak
    prec_com[i] = prec_peak
    A1_peaks_less_than_AG_peaks[i] = A1_peak < AG_peak
    A1_peaks_less_than_prec_peaks[i] = A1_peak < prec_peak

pvals_A1_AG = 1-np.sum(A1_peaks_less_than_AG_peaks)/len(A1_peaks_less_than_AG_peaks)
pvals_A1_prec = 1-np.sum(A1_peaks_less_than_prec_peaks)/len(A1_peaks_less_than_prec_peaks)

# compute rois preferred event length in seconds
a1_pref = unique_event_lengths[np.argmax(a1_mean)] 
AG_pref = unique_event_lengths[np.argmax(AG_mean)]
prec_pref = unique_event_lengths[np.argmax(prec_mean)] 

# compute rois preferred event length as average center of mass across bootstraps
A1_com_mean = np.mean(A1_com)
AG_com_mean = np.mean(AG_com)
prec_com_mean = np.mean(prec_com)

# plot vertical lines corresponding to the preferred event length for each ROI 
plt.axvline(A1_com_mean,color='indigo',linewidth=3)
plt.axvline(AG_com_mean,color='magenta',linewidth=3)
plt.axvline(prec_com_mean,color='green',linewidth=3)

plt.savefig('hmm_K_sweep_paper_results/principled/preferred_event_length_split_merge_01_lprec_full.png')


