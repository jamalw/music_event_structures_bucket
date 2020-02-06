import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/'

a1 = np.load(datadir + 'smooth_y_a1.npy')
AG = np.load(datadir + 'smooth_y_AG.npy')
prec = np.load(datadir + 'smooth_y_prec.npy')

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

plt.plot(unique_event_lengths, a1_mean, color='k', label='rA1')
plt.fill_between(unique_event_lengths, a1_min, a1_max,color='k',alpha=0.3)

plt.plot(unique_event_lengths, AG_mean, color='magenta', label='lAG')
plt.fill_between(unique_event_lengths, AG_min, AG_max,color='magenta',alpha=0.3)

plt.plot(unique_event_lengths, prec_mean, color='green', label='lprec')
plt.fill_between(unique_event_lengths, prec_min, prec_max,color='green',alpha=0.3)

plt.legend()

event_lengths_str = ['2','','','3','','','','4','','5','','','','6','','','','','9','10','','12','15','18','20','25','27','30','36','45','60','75']

plt.xticks(unique_event_lengths,event_lengths_str,rotation=45)
plt.xlabel('Event Length (s)', fontsize=18)
plt.ylabel('WvA Score', fontsize=18)
plt.title('ROIs Preferred Event Length', fontsize=18)
plt.tight_layout()

plt.savefig('hmm_K_sweep_paper_results/preferred_event_length')

A1_peaks_less_than_AG_peaks = np.zeros((len(unique_event_lengths)))
A1_peaks_less_than_prec_peaks = np.zeros((len(unique_event_lengths)))

for i in range(len(unique_event_lengths)):
    A1_peak = np.max(a1[i,:])
    AG_peak = np.max(AG[i,:])
    prec_peak = np.max(prec[i,:]) 
    A1_peaks_less_than_AG_peaks[i] = A1_peak < AG_peak
    A1_peaks_less_than_prec_peaks[i] = A1_peak < prec_peak

pvals_A1_AG = np.sum(A1_peaks_less_than_AG_peaks)/len(A1_peaks_less_than_AG_peaks)
pvals_A1_prec = np.sum(A1_peaks_less_than_prec_peaks)/len(A1_peaks_less_than_prec_peaks)

