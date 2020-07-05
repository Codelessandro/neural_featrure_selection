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


np.set_printoptions(suppress=True)


xy, y_score = load_data(config["task"])
y_score = normalize(y_score)

model, i, modelhistory = best_feedforward_model(xy, y_score, True)

WineData = BaseData('data/winequality-red.csv', ';', 11, 10, config["nr_base_columns"], rifs=True)
WineData.load()
_=evaluation_wrapper(config["task"], model, 'data/winequality-red.csv', True, WineData)
