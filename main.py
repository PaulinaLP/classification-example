import os
import sys
import pandas as pd
import json
import pickle

from Code.ingest_data import download
from Code import preprocess
from Code import eda
from Code import training

script_path = os.path.abspath(os.path.dirname(sys.argv[0]))
dependencies_path = os.path.join(script_path, 'dependencies')
output_path = os.path.join(script_path, 'output')
input_path = os.path.join(script_path, 'input')

if __name__ == '__main__':
    with open(os.path.join(input_path, 'config.json')) as file:
        configuration = json.load(file)
        # get data from config file
        id_column = configuration['id']
        target = configuration['target']
    # downloading the data if not previously downloaded from sql server
    if configuration['download_sql'] == 1:
        server = configuration['server']
        database = configuration['database']
        df_train = download(os.path.join(input_path, 'ingest.sql'), server, database)
        df_train.to_csv(os.path.join(input_path, 'train.csv'))
    else:
        # if already downloaded get data from csv
        df_train = pd.read_csv(os.path.join(input_path, 'train.csv'))
    if configuration['eda'] == 1:
        eda.quantitative_analysis(df_train, target)
    if configuration['preprocessing'] == 1:
        preprocessing_pipeline, X_train_preprocessed, X_test_preprocessed, y_train, y_test = (
            preprocess.preprocess(df_train, target))
        with open(os.path.join(output_path, 'preprocessing_pipeline'), 'wb') as f:
            pickle.dump(preprocessing_pipeline, f)
        if configuration['train'] == 1:
            mlflow_uri = configuration['mlflow_uri']
            experiment_name = 'experiment_rf_1'
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            # training with param grid
            training.run_mlflow_experiment(param_grid, mlflow_uri, experiment_name, X_train_preprocessed,
                                           X_test_preprocessed, y_train, y_test)


