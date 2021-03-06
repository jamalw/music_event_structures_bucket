import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/beh/'

df1 = pd.read_excel(datadir + 'recog_scores.xlsx',"Day 1")
df2 = pd.read_excel(datadir + 'recog_scores.xlsx',"Day 2")


data1 = df1.as_matrix(columns=df1.columns[0:])
data2 = df2.as_matrix(columns=df2.columns[0:])
 #
 #mean_data1 = np.mean(data1)
 #mean_data2 = np.mean(data2)
 #sem_data1 = stats.sem(np.reshape(data1,(32*25)))
 #sem_data2 = stats.sem(np.reshape(data2,(32*25)))
 #
 #all_data_means = np.array([mean_data1,mean_data2])
 #all_data_sems = np.array([sem_data1,sem_data2])
 #
data1_exp  = np.mean(data1[0:16,:])
data1_lure = np.mean(data1[16:32,:])
data2_exp  = np.mean(data2[0:16,:])
data2_lure = np.mean(data2[16:32,:])
 #
 #p_correct_data1_exp = np.sum(data1_exp)/80
 #p_correct_data1_lure = np.sum(data1_lure)/80
 #p_correct_data2_exp = np.sum(data2_exp)/80
 #p_correct_data2_lure = np.sum(data2_lure)/80 

all_p_correct = np.array([data1_exp,data1_lure,data2_exp,data2_lure])

N = 4
ind = np.arange(N)
width = 0.35

fig = plt.figure()
ax = fig.add_subplot(111)

rects1 = ax.bar(ind, all_p_correct, width, color='k', align = 'center')
#labels = ['Presented', 'Lure', 'Presented','Lure']
#plt.xticks(ind + width / 4.5, labels, fontsize = 15)
ax.get_children()
ax.get_children()[1].set_color('r')
ax.get_children()[3].set_color('r')
ax.set_xticklabels(('','Presented','\n\nDay1','Lure','','Presented','\n\nDay2','Lure'),ha='center', fontsize=15)
ax.set_title('Recognition Test Scores', fontweight='bold',fontsize=18)
ax.set_ylabel('Recognition Scores',fontsize=15)
ax.set_ylim([0,5])
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='x', bottom=False)
plt.tight_layout()

# compute t-test for difference between day1 discrimination and day 2 discrimination
d1 = np.mean(data1[0:16,:],axis=0)
d2 = np.mean(data2[0:16,:],axis=0)
l1 = np.mean(data1[16:32,:],axis=0)
l2 = np.mean(data2[16:32,:],axis=0)

diff1 = d1 - l1
diff2 = d2 - l2

stats.ttest_rel(diff1,diff2)

plt.savefig('plots/recog_fig.svg')


