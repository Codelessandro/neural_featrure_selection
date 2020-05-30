import numpy as np
from config import *
from scipy.stats import pearsonr

from utils import *


def pearson(base_dataset,add_column,y):
    return pearsonr(add_column,y)

def constant(base_dataset,add_column,y):
    return 0.5


def evaluation(base_dataset, add_column, y, neural_feature_model):
    evaluations=[]
    if config["machine_learning_task"]=="regression":
        evaluations.append( {
            "constant" : constant(base_dataset,add_column, y)
        })
        evaluations.append( {
            "pearson": pearson(base_dataset, add_column, y)[0]
        })


    extended_x =  np.concatenate((base_dataset,add_column.reshape(-1,1)),axis=1)  #.reshape(1,config["max_limit_dataset_rows"]*(config["nr_base_columns"]+1))
    x_batch = generate_batch(extended_x)
    scores=neural_feature_model.predict(x_batch)
    score = np.mean(scores)


    print("For this dataset with this column we have the following scores:")

    for e in evaluations:
        method = list(e.keys())[0]
        print("method:" + str(method) + str(e[method]))

    print("Neural Feature Selection:" + str(score))

    return score, pearson(base_dataset, add_column, y)[0]




