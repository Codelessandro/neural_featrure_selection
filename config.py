config = {
    "machine_learning_task" : "regression", #classification
    "nr_dummy_datasets" : 5,
    "nr_additional_columns_per_dataset" : 4,
    "max_limit_dataset_rows" : 1000,
    "dataset_rows" : 1000,
    "batch_size": 20, #nr_# rows
    "nr_base_columns" : 5,
    "nr_add_columns_per_budget_group" : 5,
    "nr_feedforward_iterations": 1,
    "budget_join" : False,
    "bootstrapping": True,
    "nr_bootstraps": 10
}
