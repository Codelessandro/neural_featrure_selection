from DummyData import *
from config import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import math
import pdb

machine_learnings_tasks = [
    "regression", "classification"
]


#build data
data = gen_dummy_data_sets()
x = data[:,0:20]
y = data[:,20]

import pdb; pdb.set_trace()


#training neural network
model = Sequential()
model.add(Dense(32, input_dim=20))
model.add(Dense(1))
model.add(Activation('relu'))

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

model.fit(x,y, epochs=10, batch_size=32, validation_split=0.2)




