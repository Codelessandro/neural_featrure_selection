from DummyData import *
from DataSetWine import *
from ClassData import *
from config import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from evaluation import *
import math
import pdb

np.set_printoptions(suppress=True)


#def __init__(self, dataset, delimiter, target, random, base_size=5, datetime=0):
WineData = BaseData('data/winequality-red.csv', ';', 11, 1, config["nr_base_columns"])
WineData.load()
GoogleData = BaseData('data/google-safe-browsing-transparency-report-data.csv', ',', 10, 5)
GoogleData.load()

data =  np.vstack((  GoogleData.data ,  WineData.data ))


x = data[:,0:-1]
y = data[:,-1]

#training neural network

for i in np.arange(5):
    model = Sequential()
    model.add(Dense(10, kernel_initializer='random_normal', bias_initializer='ones', input_dim=config["max_limit_dataset_rows"] * (config["nr_base_columns"] + 1)))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='rmsprop', learning_rate=0.001, loss='binary_crossentropy')
    model.fit(x, y, epochs=10, batch_size=2, validation_split=0.2)


evaluation(WineData.base_x, np.random.normal(0, 10,  WineData.dataset[:, 9].shape[0]), WineData.dataset[:,11], model)
evaluation(WineData.base_x, WineData.dataset[:, 9], WineData.dataset[:,11], model)
evaluation(WineData.base_x, WineData.dataset[:, 11], WineData.dataset[:,11], model)
