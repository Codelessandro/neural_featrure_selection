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


x = np.concatenate( (GoogleData.x,WineData.x),axis=0)
y = normalize(np.concatenate( (GoogleData.y,WineData.y),axis=0))

model, i, modelhistory = best_feedforward_model(x,y,True)


print("i:")
print(i)

print("++++")
print("++++")
evaluation(WineData.base_x, np.random.normal(0, 10,  WineData.base_dataset[:, 9].shape[0]), WineData.base_dataset[:,11], model)
print("++++")
print("++++")
for c in [0,1,2,3,4,5,6,7,8,9,10]:
    evaluation(WineData.base_x, WineData.base_dataset[:, c], WineData.base_dataset[:,11], model)
print("++++")
print("++++")
evaluation(WineData.base_x, WineData.base_dataset[:, 11], WineData.base_dataset[:,11], model)
print("++++")
print("++++")