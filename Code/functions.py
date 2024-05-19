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


def get_feature_names(preprocessor, numeric_columns, categorical_columns):
    # Get numeric feature names
    numeric_features = numeric_columns
    # Get categorical feature names after one-hot encoding
    cat_pipeline = preprocessor.named_transformers_['cat']
    onehot_features = cat_pipeline.named_steps['onehot'].get_feature_names_out(categorical_columns)
    # Concatenate all feature names
    feature_names = numeric_features + onehot_features.tolist()
    return feature_names