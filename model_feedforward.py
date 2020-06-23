from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten

import numpy as np
from config import *

from matplotlib import pyplot as plt


def best_feedforward_model(x,y,plot_batch_labels=False):


    def get_random_hyperparams():
        hyperparams = {
            "learning_rate" : np.random.choice([0.1,0.01,0.001,0.0001]),
            "epochs" : np.random.choice([5,10,20,30,50]),
            "batch_size" : np.random.choice([2,4,6,8,16,32]),
            "layers" : np.random.choice([2,3,4]),
            "nodes_on_layer": np.random.choice([4,5,6])
        }

        if config["prod"]==False:
            hyperparams["epochs"] = 1
        return hyperparams

    best_model=None
    best_i=None
    best_val_loss=np.Infinity

    for i in np.arange(config["nr_feedforward_iterations"]):
        hp = get_random_hyperparams()
        model = Sequential()

        if config["budget_join"]:
            input_shape = config["nr_base_columns"] + 1 + config["nr_add_columns_budget"]
        else:
            input_shape =  config["nr_base_columns"] + 1 + 1 #+1=target / +1 = add_coluim


        model.add(Flatten(input_shape=(config["batch_size"], input_shape))   )
        model.add(Dense(10, kernel_initializer='random_normal', bias_initializer='ones'))

        for layer in np.arange(hp["layers"]):
            model.add(Dense(hp["nodes_on_layer"]))

        if config["budget_join"]:
            model.add(Dense(config["nr_add_columns_budget"]))
        else:
            model.add(Dense(1))

        model.add(Activation('sigmoid'))
        model.compile(optimizer='rmsprop', learning_rate=hp["learning_rate"], loss='mse')
        _=model.fit(x, y, epochs=hp["epochs"], batch_size=hp["batch_size"], validation_split=0.2)

        mean_val_loss=np.mean(_.history["val_loss"][-10:])

        if mean_val_loss<best_val_loss:
            best_model=model
            best_val_loss = mean_val_loss
            best_i = i

        if plot_batch_labels:

            if  config["budget_join"]==False:
                plt.scatter(np.arange(y.shape[0]), y)
                plt.scatter(np.arange(y.shape[0]), model.predict(x))
                plt.savefig('./plots/' + str(i) + '.png')
                plt.close()

            if  config["budget_join"]==True:
                plt.scatter(np.arange(y.shape[0]*y.shape[1]), y.flatten())
                plt.scatter(np.arange(y.shape[0]*y.shape[1]), model.predict(x).flatten())
                plt.savefig('./plots/' + str(i) + '.png')
                plt.close()

    return best_model, best_i, _