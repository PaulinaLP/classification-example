import os
import sys
import pandas as pd
import json
from Code.ingest_data import download
from Code import preprocessing

script_path = os.path.abspath(os.path.dirname(sys.argv[0]))
dependencies_path = os.path.join(script_path, 'dependencies')
output_path = os.path.join(script_path, 'output')
input_path = os.path.join(script_path, 'input')

if __name__ == '__main__':
    with open(os.path.join(input_path, 'config.json')) as file:
        configuration = json.load(file)
    if configuration['download_sql'] == 1:
        server = configuration['server']
        database = configuration['database']
        df_train = download(os.path.join(input_path, 'ingest.sql'), server, database)
        df_train.to_csv(os.path.join(input_path, 'train.csv'))
    else:
        df_train = pd.read_csv(os.path.join(input_path, 'train.csv'))
        id_column = configuration['id']
        target = configuration['target']
        outliers_fin, outliers_agg = preprocessing.outliers(df_train,id_column)

    if configuration['feature_engineering'] == 1:
        print("fe")
    if configuration['eda'] == 1:
        print("eda")
    if configuration['train'] == 1:
        print("train")

