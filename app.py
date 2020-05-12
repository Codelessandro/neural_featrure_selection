from DummyData import *
from DataSetWine import *
from GoogleTransparency import *
from config import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import math
import pdb

machine_learnings_tasks = [
    "regression", "classification"
]


#def get_data():
    #wine = DataSetWine()
    #return wine.data

def get_data():
    transparency = GoogleTransparency()
    return transparency.data


#data = gen_dummy_data_sets()

data = get_data()

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
