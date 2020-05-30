from DummyData import *
from DataSetWine import *
from ClassData import *
from config import *


import math
import pdb

from evaluation import *
from model_feedforward import *

np.set_printoptions(suppress=True)


#def __init__(self, dataset, delimiter, target, random, base_size=5, datetime=0):
WineData = BaseData('data/winequality-red.csv', ';', 11, 1, config["nr_base_columns"])
WineData.load()
GoogleData = BaseData('data/google-safe-browsing-transparency-report-data.csv', ',', 10, 5)
GoogleData.load()


xy = np.concatenate( (GoogleData.xy,WineData.xy),axis=0)
y_score = normalize(np.concatenate( (GoogleData.y_score,WineData.y_score),axis=0))
#y_data = normalize(np.concatenate( (GoogleData.y_data,WineData.y_data),axis=0))

'''
from scipy.stats import pearsonr
pearsons=[]
nfs=[]

for  i in np.arange(y_score.shape[0]):
    pearson = pearsonr(x[i][:,5], y_data[i])[0]
    _nfs = y_score[i]
    pearsons.append(pearson)
    nfs.append(_nfs)
'''





model, i, modelhistory = best_feedforward_model(xy,y_score,True)


evaluations=[]

print("i:")
print(i)

print("++++")
print("++++")
evaluations.append(evaluation(WineData.base_xy, np.random.normal(0, 10,  WineData.base_dataset[:, 9].shape[0]), WineData.base_dataset[:,11], model))
print("++++")
print("++++")
for c in [0,1,2,3,4,5,6,7,8,9,10]:
    evaluations.append(evaluation(WineData.base_xy, WineData.base_dataset[:, c], WineData.base_dataset[:,11], model))
print("++++")
print("++++")
evaluations.append(evaluation(WineData.base_xy, WineData.base_dataset[:, 11], WineData.base_dataset[:,11], model))
print("++++")
print("++++")

evaluations = np.array(evaluations)
print(evaluations)
plt.scatter(evaluations[:,0],evaluations[:,1])
plt.xlabel('NFS Score')
plt.ylabel('Pearson')
plt.xlim(-1,1)
plt.ylim(-1,1)
plt.show()