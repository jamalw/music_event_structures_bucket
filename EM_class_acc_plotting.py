import numpy as np
import matplotlib.pyplot as plt
import pathlib

datadir = "/jukebox/norman/jamalw/MES/prototype/link/scripts/EM_song_data/"

allKs = np.arange(2,100,5)
nSongs = 16
n_comparisons = ((nSongs**2) - nSongs) * 2

nllTrain = np.zeros((len(allKs),n_comparisons))
nllTest  = np.zeros((len(allKs),n_comparisons))
accTrain = np.zeros((len(allKs),n_comparisons))
accTest  = np.zeros((len(allKs),n_comparisons))

for k in range(len(allKs)):
    counter = 0
    for s1 in range(nSongs):
        for s2 in range(nSongs):
            fn = pathlib.Path(datadir + 'allACC_' + str(allKs[k]) + str(s1) + str(s2) + '.npy')
            if fn.exists ():
                #if counter == 71:
                    #print(fn)
                data = np.mean(np.load(fn),axis=1)
                nllTrain[k,counter] = data[0]
                nllTest[k,counter] = data[1]
                accTrain[k,counter] = data[2]
                accTest[k,counter]  = data[3]
                counter = counter + 1
            else:
                counter = counter + 1
                continue

# plot results
ind = np.arange(len(allKs))

plt.figure(1,figsize=(10,5))
nllTrainAvg = np.true_divide(nllTrain.sum(1),(nllTrain!=0).sum(1))
nllTestAvg = np.true_divide(nllTest.sum(1),(nllTest!=0).sum(1))
plt.plot(ind,nllTrainAvg,'g',label='Train',linewidth=3)
plt.plot(ind,nllTestAvg,'k',label='Test',linewidth=3)
plt.xticks(ind,(allKs),fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('SRM K',fontsize=17,fontweight='bold')
plt.ylabel('Average % Correct',fontsize=17, fontweight='bold')
plt.title('Song Classification Train/Test Accuracy (Neg-Log)',fontsize=20,fontweight='bold')
plt.legend(prop={'size': 15})
plt.tight_layout()
plt.savefig('plots/EM_plots/train_test_nll')
             
plt.figure(2,figsize=(10,5))
trainAvg = np.true_divide(accTrain.sum(1),(accTrain!=0).sum(1))
plt.plot(ind,trainAvg,'g',label='Train',linewidth=3)
testAvg = np.true_divide(accTest.sum(1),(accTest!=0).sum(1))
plt.plot(ind,testAvg,'k',label='Test',linewidth=3)
plt.xticks(ind,(allKs),fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('SRM K',fontsize=17,fontweight='bold')
plt.ylabel('Average % Correct',fontsize=17, fontweight='bold')
plt.title('Song Classification Train/Test Accuracy (% Correct)',fontsize=20,fontweight='bold')
plt.legend(prop={'size': 15})
plt.tight_layout()
plt.savefig('plots/EM_plots/train_test_perc_correct')

#plt.show()




