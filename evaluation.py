import numpy as np
from config import *

def method_a(base_dataset,add_column):
    return 0.8

def method_b(base_dataset,add_column):
    return 0.6


def evaluation(base_dataset, add_column, neural_feature_model):
    score_method_a = method_a(base_dataset, add_column)
    score_method_b = method_b(base_dataset,add_column)
    extended_x =  np.concatenate((base_dataset,add_column.reshape(-1,1)),axis=1).reshape(1,config["max_limit_dataset_rows"]*(config["nr_base_columns"]+1))

    score = neural_feature_model.predict(extended_x)
    print("For this dataset with this column we have the following scores:")
    print("Method A:" + str(score_method_a))
    print("Method B:" + str(score_method_b))
    print("Neural Feature Selection:" + str(score[0][0]))



