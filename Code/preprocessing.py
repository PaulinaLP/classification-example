import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class OutliersPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_column):
        self.target_column = target_column
        self.upper_bounds = {}
        self.lower_bounds = {}

    def fit(self, X, y=None):
        # Check if input is a numpy array and convert to DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Calculate upper and lower bounds for each numeric column except for the target column
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()

        for column in numeric_columns:
            upper_bound = X[column].quantile(0.75) + 1.5 * (X[column].quantile(0.75) - X[column].quantile(0.25))
            lower_bound = X[column].quantile(0.25) - 1.5 * (X[column].quantile(0.75) - X[column].quantile(0.25))
            self.upper_bounds[column] = upper_bound
            self.lower_bounds[column] = lower_bound

        return self

    def transform(self, X):
        # Check if input is a numpy array and convert to DataFrame
        is_numpy_array = isinstance(X, np.ndarray)
        if is_numpy_array:
            X = pd.DataFrame(X)

        # Update values based on bounds
        for column in self.upper_bounds:
            upper_bound = self.upper_bounds[column]
            lower_bound = self.lower_bounds[column]
            # if the value is an outlier we clip it to min/max permitted value
            X[column] = X[column].clip(lower=lower_bound, upper=upper_bound)

        # If the original input was a numpy array, convert back to numpy array
        if is_numpy_array:
            X = X.values

        return X


class CategoryPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5, columns=[]):
        self.threshold = threshold
        self.columns_to_remove = []
        self.columns= columns

    def fit(self, X, y=None):
        # Check if input is a numpy array and convert to DataFrame
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        for column in self.columns:
            value_counts = X[column].value_counts(normalize=True)
            top_5_categories_sum = value_counts.iloc[:5].sum()
            if top_5_categories_sum < self.threshold:
                print(self.columns_to_remove)
                print(f'removing column {column}')
                self.columns_to_remove.append(column)


        return self

    def transform(self, X):
        # Check if input is a numpy array and convert to DataFrame
        is_numpy_array = isinstance(X, np.ndarray)
        if is_numpy_array:
            X = pd.DataFrame(X)

        X = X.drop(self.columns_to_remove,axis=1)

        # If the original input was a numpy array, convert back to numpy array
        if is_numpy_array:
            X = X.values

        return X


