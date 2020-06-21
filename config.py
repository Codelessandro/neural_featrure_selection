config = {
    "machine_learning_task" : "regression", #classification
    "nr_dummy_datasets" : 5,
    "max_limit_dataset_rows" : 1000,
    "dataset_rows" : 1000,
    "batch_size": 20, #nr_# rows
    "nr_base_columns" : 5,
    "nr_feedforward_iterations": 1,
    "budget_join" : True,
    "nr_add_columns_budget": 140 #if this number is biger than colums we can get from the dataset, bootstrapping is activated automatically
}
