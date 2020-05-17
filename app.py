from DummyData import *
from DataSetWine import *
from ClassData import *
from config import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from evaluation import *
import math
import pdb

machine_learnings_tasks = [
    "regression", "classification"
]

wine = DataSetWine()
data = wine.data


x = wine.data[:, 0:-1]
y = wine.data[:, -1]




#def __init__(self, dataset, delimiter, target, random, base_size=5, datetime=0):
WineData = BaseData('data/winequality-red.csv', ';', 11, 1, config["nr_base_columns"])
WineData.load()
data = WineData.data


x = data[:,0:-1]
y = data[:,-1]

#training neural network
model = Sequential()

model.add(Dense(10, kernel_initializer='random_normal', bias_initializer='ones', input_dim=config["max_limit_dataset_rows"] * (config["nr_base_columns"] + 1)))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='rmsprop', learning_rate=0.000001, loss='binary_crossentropy')
model.fit(x, y, epochs=1, batch_size=2, validation_split=0.2)



print("--")

print(model.predict(x))
print(y)
print("--")

evaluation(WineData.base_x, WineData.dataset[:, 9], model)
evaluation(WineData.base_x, WineData.dataset[:, 11], model)
