import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

datadir = '/jukebox/norman/jamalw/MES/prototype/link/scripts/data/beh/'

df1 = pd.read_excel(datadir + 'familiarity_scores.xlsx',"Day 1")
df2 = pd.read_excel(datadir + 'familiarity_scores.xlsx',"Day 2")


data1 = df1.as_matrix(columns=df1.columns[0:])
data2 = df2.as_matrix(columns=df2.columns[0:])

# average over songs leaving one average score per participant
day1 = np.mean(data1,axis=0)
day2 = np.mean(data2,axis=0)

subj_avg = (day1 + day2)/2
grand_avg = np.mean(subj_avg)

# compute normalized responses
day1_new = day1 - subj_avg + grand_avg
day2_new = day2 - subj_avg + grand_avg

mean_data1 = np.mean(data1)
mean_data2 = np.mean(data2)
sem_data1 = stats.sem(day1_new)
sem_data2 = stats.sem(day2_new)

all_data_means = np.array([mean_data1,mean_data2])
all_data_sems = np.array([sem_data1,sem_data2])

N = 2
ind = np.arange(N)
width = 0.35

#plt.bar(ind, all_data_means, width, color='k')
plt.bar(ind, all_data_means, width, color='k', yerr = all_data_sems, error_kw=dict(ecolor='lightseagreen',lw=3, solid_capstyle='projecting',capsize=5,capthick=2))
plt.ylabel('Familiarity Ratings', fontsize=15)
plt.title('Familiarity Judgements',fontweight='bold',fontsize=18)
labels = ['Day 1', 'Day 2']
plt.xticks(ind + width / 4.5, labels, fontsize = 15)
axes = plt.gca()
axes.set_ylim([0,5])

plt.savefig('familiarity_fig.png')

t,p = stats.ttest_rel(day1, day2, axis=0)

