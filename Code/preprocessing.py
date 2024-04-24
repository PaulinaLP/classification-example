import pandas as pd
import numpy as np


class OutliersPreprocessor:
    def __init__(self, target_column):
        self.target_column = target_column
        self.upper_bounds = {}
        self.lower_bounds = {}

    def fit(self, df):
        # Calculate upper and lower bounds for each numeric column except for the target column
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns.remove(self.target_column)

        for column in numeric_columns:
            upper_bound = df[column].quantile(0.75) + 1.5 * (
                        df[column].quantile(0.75) - df[column].quantile(0.25))
            lower_bound = df[column].quantile(0.25) - 1.5 * (
                        df[column].quantile(0.75) - df[column].quantile(0.25))
            self.upper_bounds[column] = upper_bound
            self.lower_bounds[column] = lower_bound

    def transform(self, df):
        # Update values based on bounds
        for column in self.upper_bounds:
            upper_bound = self.upper_bounds[column]
            lower_bound = self.lower_bounds[column]
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

        return df


class CustomPreprocessor:
    def __init__(self, target_column):
        self.target_column = target_column
        self.outliers_preprocessor = OutliersPreprocessor(target_column)

    def fit(self, df):
        # Fit OutliersPreprocessor
        self.outliers_preprocessor.fit(df)

    def transform(self, df):
        # Transform using OutliersPreprocessor
        df_transformed = self.outliers_preprocessor.transform(df)
        # Other preprocessing steps can be added here
        return df_transformed
