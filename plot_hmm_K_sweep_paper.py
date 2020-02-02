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

# compute 95% confidence intervals for each bootstrap sample
for i in range(len(unique_event_lengths)):
    a1_boot_mean = np.mean(a1[i,:])
    a1_95ci = 1.96 * sem(a1[i,:])
    a1_min[i] = a1_boot_mean - a1_95ci
    a1_max[i] = a1_boot_mean + a1_95ci

    AG_boot_mean = np.mean(AG[i,:])
    AG_95ci = 1.96 * sem(AG[i,:])
    AG_min[i] = AG_boot_mean - AG_95ci
    AG_max[i] = AG_boot_mean + AG_95ci

    prec_boot_mean = np.mean(prec[i,:])
    prec_95ci = 1.96 * sem(prec[i,:])
    prec_min[i] = prec_boot_mean - prec_95ci
    prec_max[i] = prec_boot_mean + prec_95ci



plt.figure(figsize=(10,5))

plt.plot(range(len(unique_event_lengths)), a1_mean, color='red', label='rA1')
plt.fill_between(range(len(unique_event_lengths)), a1_min, a1_max,color='red',alpha=0.3)

plt.plot(range(len(unique_event_lengths)), AG_mean, color='magenta', label='lAG')
plt.fill_between(range(len(unique_event_lengths)), AG_min, AG_max,color='magenta',alpha=0.3)

plt.plot(range(len(unique_event_lengths)), prec_mean, color='green', label='lprec')
plt.fill_between(range(len(unique_event_lengths)), prec_min, prec_max,color='green',alpha=0.3)

plt.legend()

plt.xticks(range(len(x)),unique_event_lengths.round(decimals=1),rotation=45)
plt.xlabel('Event Length (s)', fontsize=18)
plt.ylabel('WvA Score', fontsize=18)
plt.title('ROIs Preferred Event Length', fontsize=18)
plt.tight_layout()

plt.savefig('hmm_K_sweep_paper_results/preferred_event_length')
