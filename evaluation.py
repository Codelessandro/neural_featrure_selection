import numpy as np
from config import *
from scipy.stats import pearsonr

from utils import *
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from BaseData import *
import time

def pearson(base_dataset,add_column,y):
    return pearsonr(add_column,y)

def constant(base_dataset,add_column,y):
    return 0.5



def evaluation_wrapper(task, model, _path, _print, eval_dataset, columns, target):


    if task == Task.multivariate_time_series:
        BirthDeaths3 = BaseData(_path, ';', 3, 1, base_size=config["nr_base_columns"])
        BirthDeaths3.load()


        evaluations = evaluation(
            BirthDeaths3.base_xy,
            add_eval_columns,
            eval_dataset.base_dataset[:, 11],
            model
        )


    if task == Task.regression:

        '''
        def get_wine_column():
            content = open('data/winequality-red.csv').read().split("\n")
            values = list(map(lambda line: line.split(";")[1], content))
            return np.array(values).astype(float)

        wc=get_wine_column()
        '''

        if config["budget_join"]:
            add_eval_columns = []
            for i in columns:
                add_eval_columns.append(eval_dataset.base_dataset[:, i])


            for i in np.arange(config["nr_add_columns_budget"] - len(add_eval_columns)):
                add_eval_columns.append(  np.random.normal(0,1,eval_dataset.base_dataset[:, 0].shape)  )
                #add_eval_columns.append(np.ones(eval_dataset.base_dataset[:, 0].shape))
        else:
            add_eval_columns = [
                eval_dataset.base_dataset[:, 0]
            ]

        evaluations = evaluation(
            eval_dataset.base_xy, #5
            add_eval_columns,   #1200
            eval_dataset.base_dataset[:, target],  #
            model #model
        )

    #plot_performance([evaluations[1]], [evaluations[3]])


    regr = linear_model.LinearRegression()
    regr.fit(eval_dataset.base_x,  eval_dataset.base_y)
    regression_score_base_data = regr.score(eval_dataset.base_x,  eval_dataset.base_y)
    indices=np.where(evaluations[0] > config["nfs_output_threshold"])



    regr = linear_model.LinearRegression()

    add_columns= np.array(list(map( lambda c: c[0] , eval_dataset.add_columns)))[indices[0],:]

    augmented_x=  np.transpose(np.concatenate( ( np.transpose(eval_dataset.base_x,(1,0)), add_columns ) ,axis=0),(1,0))
    regr.fit(augmented_x, eval_dataset.base_y)
    regression_score_augmented = regr.score(augmented_x, eval_dataset.base_y)
            
    output= ""
    output+="Path: " + str(_path)
    output+="\nBase DataSet size: " + str(eval_dataset.base_x.shape)
    output+="\nScore for Base DataSet: " + str( regression_score_base_data)

    output+="\nColumns presented to NFS: " + str(config["nr_add_columns_budget"])
    output+="\nColumns chosen by NFS by threshold: " + str(len(indices[0]))
    output+="\nAugmented DataSet size: " + str(augmented_x.shape)


    output+="\nScore for Augmented DataSet: " + str( regression_score_augmented)

    output+="\nNFS Time: " + str(  evaluations[1])
    output+="\nPearson Time: " + str(  evaluations[3])

    output+="\nConfig: " + str(  config )

    f = open("output/" + str(time.time()).split(".")[0] + ".txt", "a")
    f.write(output)
    f.close()

    return _path, evaluations, regression_score_base_data, regression_score_augmented


def evaluation(base_dataset, add_columns, y, neural_feature_model):

    evaluations=[]
    if config["machine_learning_task"]=="regression":




        p_start = timer()
        for c in add_columns:
            p = pearson(base_dataset, c, y)[0]
            evaluations.append( {
                "pearson": p
            })
        p_end = timer()


    extended_x =  np.concatenate((base_dataset,  np.vstack(add_columns).reshape(-1,len(add_columns))  ), axis=1) #.reshape(1,config["max_limit_dataset_rows"]*(config["nr_base_columns"]+1))
    x_batch = generate_batch(extended_x)


    nfs_start = timer()
    scores=neural_feature_model.predict(x_batch)
    scores = np.mean(scores,axis=0)
    nfs_end = timer()



    evaluations = list(map(lambda e: e['pearson'], evaluations))
    return  scores,  (nfs_end - nfs_start), evaluations, (p_end-p_start)




def plot_performance(nfs, method, columns):

    N=np.arange(len(method))

    width = 0.5
    p1 = plt.bar(N*2, nfs, width, align='edge')
    print(nfs)
    print(method)
    p2 = plt.bar(N*2+0.5, method, width, align='edge')

    plt.xlabel('additional Column on a Dataset')
    plt.ylabel('Time in seconds')
    #plt.yscale('log')
    plt.xticks(np.arange(len(columns))*2,list(map(lambda c: str(c) + " columns", columns)))

    plt.legend((p1[0], p2[0]), ('NFS', 'Pearson'))
    plt.show()
