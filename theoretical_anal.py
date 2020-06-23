''''
copy this is app.py for execution
'''

def theory_wrapper():
    evaluations=[]
    columns = [10,20,40,80,160,320,640,1280]
    columns = np.arange(50,1500,50)
    for c in columns:
        evaluation = theory(c)
        evaluations.append(evaluation)
    print(evaluations)

    plot_performance( list(map(lambda eval: eval[1] , evaluations)  ), list(map(lambda eval: eval[3] , evaluations)  ), columns)

def theory(columns):
    config["nr_add_columns_budget"]=columns

    xy, y_score = load_data(config["task"])
    model, i, modelhistory = best_feedforward_model(xy, y_score, True)
    evaluation = evaluation_wrapper(config["task"], model)
    return evaluation

theory_wrapper()
