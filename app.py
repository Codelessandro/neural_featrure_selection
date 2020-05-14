from DummyData import *
from DataSetWine import *
from ClassData import *
from config import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import math
import pdb

machine_learnings_tasks = [
    "regression", "classification"
]


def get_data():
    wine = DataSetWine()
    return wine.data



data = get_data()


#def __init__(self, dataset, delimiter, target, random, base_size=5, datetime=0):
WineData = BaseData('data/winequality-red.csv', ';', 11, 1, 5)
WineData.load()
import pdb; pdb.set_trace()

x = data[:,0:-1]
y = data[:,-1]


#training neural network
import pdb; pdb.set_trace()
model = Sequential()
model.add(Dense(32, input_dim=11193))
model.add(Dense(1))
model.add(Activation('relu'))

model.compile(optimizer='rmsprop',
              loss='mse')

model.fit(x,y, epochs=10, batch_size=2, validation_split=0.2)




