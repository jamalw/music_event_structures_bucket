import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
from statsmodels.nonparametric.kernel_regression import KernelReg

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/hmm_K_sweep_paper_results/principled/'

jobNum = 20
nBoots = 1000
prec_data_list = []
a1_data_list = []
AG_data_list = []
mpfc_data_list = []

# load in each job (20 of which contain 50 bootstraps each which is 1000 boostraps total) for each ROI separately to be converted into one large matrix containing all bootstraps
for i in range(jobNum):
    prec_data_jobNum = np.load(datadir + 'lprec_wva_split_merge_01_' + str(i) + '.npy')
    prec_data_list.append(prec_data_jobNum)
    a1_data_jobNum = np.load(datadir + 'rA1_wva_split_merge_01_' + str(i) + '.npy')
    a1_data_list.append(a1_data_jobNum)
    AG_data_jobNum = np.load(datadir + 'lAG_wva_split_merge_01_' + str(i) + '.npy')
    AG_data_list.append(AG_data_jobNum)
    mpfc_data_jobNum = np.load(datadir + 'bil_mPFC_wva_split_merge_01_' + str(i) + '.npy')
    mpfc_data_list.append(mpfc_data_jobNum)


prec_data = np.dstack(prec_data_list) 
a1_data = np.dstack(a1_data_list)
AG_data = np.dstack(AG_data_list)
mpfc_data = np.dstack(mpfc_data_list)

sigma = '5'

durs_run1 = np.array([225,90,180,135,90,180,135,90,180,135,225,90,225,225,180,135])

durs_run1_new = durs_run1[:,np.newaxis]

fairK = np.array((3,5,9,15,20,25,30,35,40,45))

event_lengths = durs_run1_new/fairK

unique_event_lengths = np.unique(event_lengths)
x = event_lengths.ravel()

ROI_data = [a1_data, AG_data, prec_data, mpfc_data]
#ROI_data = [a1_data,AG_data,prec_data]

test_x = np.linspace(min(x), max(x), num=100)
smooth_wva = np.zeros((len(unique_event_lengths), len(ROI_data), nBoots))

for b in range(nBoots):
    # Optimize bandwidth
    opt_bw = 0
    for ROI in range(len(ROI_data)):
        y = ROI_data[ROI][:,:,b].ravel()
        KR = KernelReg(y,x,var_type='c')
        opt_bw += KR.bw/len(ROI_data)

    max_wva = np.zeros(len(ROI_data))
    for ROI in range(len(ROI_data)):
        y = ROI_data[ROI][:,:,b].ravel()
        KR = KernelReg(y,x,var_type='c', bw=opt_bw)
        max_wva[ROI] = np.argmax(KR.fit(test_x)[0])  # Find peak on fine grid
        smooth_wva[:, ROI, b] += KR.fit(unique_event_lengths)[0]

np.save(datadir + 'smooth_wva_split_merge_01_a1_prec_AG_bilmPFC',smooth_wva)


