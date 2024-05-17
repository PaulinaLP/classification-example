import pandas as pd


def safe_division(dividend, divisor):
    if divisor == 0:
        return 1
    else:
        return dividend / divisor


def get_first_character(value):
    if value[0].isalpha():
        return value[0]
    else:
        return '0'


def check_null(df):
    null_counts = df.isnull().sum()
    dtypes = df.dtypes
    result = pd.concat([null_counts, dtypes], axis=1, keys=['null_count', 'data_type'])
    return result, null_counts
