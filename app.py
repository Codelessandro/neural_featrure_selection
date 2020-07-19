from DummyData import *
from DataSetWine import *
from BaseData import *
from config import *

import math
import pdb

from evaluation import *
from model_feedforward import *
from functools import reduce
from load_data import *

print("a")
np.set_printoptions(suppress=True)
print("b")


xy, y_score = load_data(config["task"])
print("c1")

#y_score, best_score = normalize(y_score)
print("c2")

model, i, modelhistory = best_feedforward_model(xy, y_score, True)
print("d")



print("We take:")
print(i)

print("Best score for normalization scale:")
#print(best_score)

WineData = BaseData('data/winequality-red.csv', ';', 11, 10, config["nr_base_columns"], rifs=True)
WineData.load()


_=evaluation_wrapper(config["task"], model, 'data/winequality-red.csv', True, WineData)
