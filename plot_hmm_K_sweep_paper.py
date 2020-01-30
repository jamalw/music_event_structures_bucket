import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from statsmodels.nonparametric.kernel_regression import KernelReg

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/'

jobNum = 20
nBoots = 1000
prec_data_list = []
a1_data_list = []
AG_data_list = []

# load in each job (20 of which contain 50 bootstraps each which is 1000 boostraps total) for each ROI separately to be converted into one large matrix containing all bootstraps
for i in range(jobNum):
    prec_data_jobNum = np.load(datadir + 'prec_wva' + str(i) + '.npy')
    prec_data_list.append(prec_data_jobNum)
    a1_data_jobNum = np.load(datadir + 'a1_wva' + str(i) + '.npy')
    a1_data_list.append(a1_data_jobNum)
    AG_data_jobNum = np.load(datadir + 'AG_wva' + str(i) + '.npy')
    AG_data_list.append(AG_data_jobNum)

prec_data = np.dstack(prec_data_list) 
a1_data = np.dstack(a1_data_list)
AG_data = np.dstack(AG_data_list)

sigma = '3'

durs_run1 = np.array([225,90,180,135,90,180,135,90,180,135,225,90,225,225,180,135])

durs_run1_new = durs_run1[:,np.newaxis]

fairK = np.array((3,5,9,15,20,25,30,35,40,45))

event_lengths = durs_run1_new/fairK

unique_event_lengths = np.unique(event_lengths)
x = event_lengths.ravel()

for b in range(nBoots):
    y_a1 = a1_data[:,:,b].ravel()
    y_AG = AG_data[:,:,b].ravel()
    y_prec = prec_data[:,:,b].ravel()

KR = KernelReg(y_a1,x,var_type='c', bw=sigma)
smooth_y_a1 = KR.fit(unique_event_lengths)
plt.plot(unique_event_lengths, smooth_y_a1[0], color='red', label='A1')

KR = KernelReg(y_AG,x,var_type='c', bw=sigma)
smooth_y_AG = KR.fit(unique_event_lengths)
plt.plot(unique_event_lengths, smooth_y_AG[0], color='magenta', label='AG')

KR = KernelReg(y_prec,x,var_type='c', bw=sigma)
smooth_y_prec = KR.fit(unique_event_lengths)
plt.plot(unique_event_lengths, smooth_y_prec[0], label='prec')


plt.legend()

#plt.xticks(x,unique_event_lengths,rotation=45)
plt.xlabel('Event Length (s)', fontsize=18)
plt.ylabel('WvA Score', fontsize=18)
plt.title('ROIs Preferred Event Length', fontsize=18)
plt.tight_layout()

#plt.savefig('hmm_K_sweep_paper_results/preferred_event_length')
