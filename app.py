from DummyData import *
from DataSetWine import *
from ClassData import *
from config import *

import math
import pdb

from evaluation import *
from model_feedforward import *
from functools import reduce



np.set_printoptions(suppress=True)


def merge_data_sets(datasets):
    print("inside")
    for d in datasets:
        d.load()


    y_score = []
    xy      = []

    for d in datasets:
        y_score.append(d.y_score)
        xy.append(d.xy)

    y_score = np.concatenate(y_score)
    xy = np.concatenate(xy)


    return xy, y_score


WineData = BaseData('data/winequality-red.csv', ';', 11, 1, config["nr_base_columns"])
GoogleData = BaseData('data/google-safe-browsing-transparency-report-data.csv', ',', 10, 5)

xy, y_score = merge_data_sets([WineData, GoogleData, GoogleData])




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

model, i, modelhistory = best_feedforward_model(xy, y_score, True)

evaluations = []

print("i:")
print(i)

print("++++")
print("++++")
evaluations.append(evaluation(WineData.base_xy, np.random.normal(0, 10, WineData.base_dataset[:, 9].shape[0]),
                              WineData.base_dataset[:, 11], model))
print("++++")
print("++++")
for c in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    evaluations.append(evaluation(WineData.base_xy, WineData.base_dataset[:, c], WineData.base_dataset[:, 11], model))
print("++++")
print("++++")
evaluations.append(evaluation(WineData.base_xy, WineData.base_dataset[:, 11], WineData.base_dataset[:, 11], model))
print("++++")
print("++++")

evaluations = np.array(evaluations)
print(evaluations)

plt.scatter(evaluations[:, 0], evaluations[:, 2])
plt.xlabel('NFS Score')
plt.ylabel('Pearson')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.show()


plot_performance(evaluations[:, 1],evaluations[:, 3])