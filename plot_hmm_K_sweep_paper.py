import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/k_sweep_results_paper/'

prec_data = np.load(datadir + 'prec_wva.npy')
a1_data = np.load(datadir + 'a1_wva.npy')
AG_data = np.load(datadir + 'AG_wva.npy')

durs_run1 = np.array([225,90,180,135,90,180,135,90,180,135,225,90,225,225,180,135])

durs_run1_new = durs_run1[:,np.newaxis]

fairK = np.array((3,5,9,15,45))

event_lengths = durs_run1_new/fairK

unique_event_lengths = np.unique(event_lengths)

plot_prec_data = np.zeros((len(unique_event_lengths)))
plot_a1_data = np.zeros((len(unique_event_lengths)))
plot_AG_data = np.zeros((len(unique_event_lengths)))

for i in range(len(unique_event_lengths)):
    plot_prec_data[i] = np.mean(prec_data[event_lengths == unique_event_lengths[i]])    
    plot_a1_data[i] = np.mean(a1_data[event_lengths == unique_event_lengths[i]])
    plot_AG_data[i] = np.mean(AG_data[event_lengths == unique_event_lengths[i]])

sigma = 3
x = np.arange(len(unique_event_lengths))

plot_a1_data_smooth  = gaussian_filter1d(plot_a1_data,sigma=sigma)
plt.plot(x,plot_a1_data_smooth, color='red', label='A1')

plot_AG_data_smooth = gaussian_filter1d(plot_AG_data,sigma=sigma)
plt.plot(x,plot_AG_data_smooth, color='magenta', label='AG')

plot_prec_data_smooth = gaussian_filter1d(plot_prec_data,sigma=sigma)
plt.plot(x,plot_prec_data_smooth, color='green', label='prec')

plt.legend()

plt.xticks(x,unique_event_lengths,rotation=45)
plt.xlabel('Event Length', fontsize=18)
plt.ylabel('WvA Score', fontsize=18)
plt.title('ROIs Preferred Event Length', fontsize=18)
plt.tight_layout()

plt.savefig('k_sweep_results_paper/preferred_event_length')
