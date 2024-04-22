import os
import sys
import pandas as pd
import json

script_path = os.path.abspath(os.path.dirname(sys.argv[0]))
dependencies_path = os.path.join(script_path, 'dependencies')
output_path = os.path.join(script_path, 'output')
input_path = os.path.join(script_path, 'input')


if __name__ == '__main__':
    with open(os.path.join(input_path, 'config.json')) as file:
        configuration = json.load(file)
    if configuration['download_sql'] == 1:
        print("1")
    df_train = pd.read_csv(os.path.join(input_path, 'train.csv'))
    print(df_train.head(10))
