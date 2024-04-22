import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def outliers(df,id_column):
    num_columns = df.select_dtypes(['float', 'int', 'int64', 'float32', 'float64', 'uint8'])
    not_binary_columns = num_columns.loc[:, ~(df.isin([0, 1,np.nan]).all())]
    outliers_fin = pd.DataFrame(columns=['variable','upper_bound','lower_bound','value',id_column])
    for column in not_binary_columns.columns:
        q1 = np.nanpercentile(not_binary_columns[column], 25)
        q3 = np.nanpercentile(not_binary_columns[column], 75)
        iqr = q3 - q1
        lower_bound = q1 - (3 * iqr)
        upper_bound = q3 + (3 * iqr)+10
        print(f'for variable {column} upper bound is {upper_bound} and lower_bound is {lower_bound}')
        out = df[((df[column] > upper_bound) | (df[column] < lower_bound)) & (df[column].notnull())]
        out['variable'] = column
        out['value'] = out[column]
        out['upper_bound'] = upper_bound
        out['lower_bound'] = lower_bound
        out = out[['variable','upper_bound','lower_bound','value',id_column]]
        outliers_fin=pd.concat([outliers_fin,out])
    aggregations = {id_column: "count"}
    outliers_agg=outliers_fin.groupby(['variable','upper_bound','lower_bound'], as_index=False).agg(aggregations)
    outliers_agg.sort_values(by=id_column, ascending=False, inplace=True)
    return outliers_fin, outliers_agg


class OutlierPreprocessor:
    def __init__(self):
        self.outliers_df = None
        self.outliers_agg_df = None
        self.id_column = None

    def fit(self, df, id_column):
        self.id_column = id_column
        num_columns = df.select_dtypes(['float', 'int', 'int64', 'float32', 'float64', 'uint8'])
        not_binary_columns = num_columns.loc[:, ~(df.isin([0, 1, np.nan]).all())]
        outliers_fin = pd.DataFrame(columns=['variable', 'upper_bound', 'lower_bound', 'value', id_column])
        for column in not_binary_columns.columns:
            q1 = np.nanpercentile(not_binary_columns[column], 25)
            q3 = np.nanpercentile(not_binary_columns[column], 75)
            iqr = q3 - q1
            lower_bound = q1 - (3 * iqr)
            upper_bound = q3 + (3 * iqr) + 10
            print(f'for variable {column} upper bound is {upper_bound} and lower_bound is {lower_bound}')
            out = df[((df[column] > upper_bound) | (df[column] < lower_bound)) & (df[column].notnull())]
            out['variable'] = column
            out['value'] = out[column]
            out['upper_bound'] = upper_bound
            out['lower_bound'] = lower_bound
            out = out[['variable', 'upper_bound', 'lower_bound', 'value', id_column]]
            outliers_fin = pd.concat([outliers_fin, out])
        aggregations = {id_column: "count"}
        outliers_agg = outliers_fin.groupby(['variable', 'upper_bound', 'lower_bound'], as_index=False).agg(aggregations)
        outliers_agg.sort_values(by=id_column, ascending=False, inplace=True)
        self.outliers_df = outliers_fin
        self.outliers_agg_df = outliers_agg

    def transform(self, df):
        transformed_df = df.copy()
        for index, row in self.outliers_df.iterrows():
            variable = row['variable']
            upper_bound = row['upper_bound']
            lower_bound = row['lower_bound']
            column = row[self.id_column]
            transformed_df.loc[(transformed_df[self.id_column] == column) & (transformed_df[variable] > upper_bound), variable] = upper_bound
            transformed_df.loc[(transformed_df[self.id_column] == column) & (transformed_df[variable] < lower_bound), variable] = lower_bound
        return transformed_df


class CustomPreprocessor:
    def __init__(self):
        self.outlier_preprocessor = None
        self.scaler = None

    def fit(self, df, id_column):
        # Initialize outlier preprocessor
        self.outlier_preprocessor = OutlierPreprocessor()
        self.outlier_preprocessor.fit(df, id_column)

        # Initialize scaler for numerical features
        self.scaler = StandardScaler()
        numerical_columns = df.select_dtypes(include=['float', 'int'])
        self.scaler.fit(numerical_columns)

    def transform(self, df):
        transformed_df = df.copy()

        # Apply outlier treatment
        transformed_df = self.outlier_preprocessor.transform(transformed_df)

        # Apply scaling to numerical features
        numerical_columns = transformed_df.select_dtypes(include=['float', 'int'])
        transformed_df[numerical_columns.columns] = self.scaler.transform(numerical_columns)

        return transformed_df
