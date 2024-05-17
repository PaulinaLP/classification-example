import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class OutliersPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_column):
        self.target_column = target_column
        self.upper_bounds = {}
        self.lower_bounds = {}

    def fit(self, df, y=None):
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
        return self

    def transform(self, df):
        # Update values based on bounds
        for column in self.upper_bounds:
            upper_bound = self.upper_bounds[column]
            lower_bound = self.lower_bounds[column]
            # if the value is an outlier we clip it to min/max permitted value
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        return df


# standard MinMax Scaler excluding a list of columns passed as param
class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_columns=None):
        self.exclude_columns = exclude_columns if exclude_columns else []
        self.scalers = {}

    def fit(self, X, y=None):
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.columns_to_scale = [col for col in numeric_columns if col not in self.exclude_columns]
        for col in self.columns_to_scale:
            scaler = MinMaxScaler()
            scaler.fit(X[[col]])
            self.scalers[col] = (scaler.data_min_[0], scaler.data_max_[0])
        return self

    def transform(self, X):
        X_scaled = X.copy()
        for col, (min_val, max_val) in self.scalers.items():
            X_scaled[col] = (X[col] - min_val) / (max_val - min_val)
        return X_scaled


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, exclude_columns=None, max_classes=None):
        self.exclude_columns = exclude_columns if exclude_columns else []
        self.max_classes = max_classes
        self.columns = None
        self.category_mappings = {}

    def fit(self, x, y=None):
        self.columns = x.columns
        for col in x.columns:
            if col not in self.exclude_columns and x[col].dtype == 'object':
                value_counts = x[col].value_counts()
                if self.max_classes and len(value_counts) > self.max_classes:
                    most_common = value_counts.nlargest(self.max_classes).index
                    self.category_mappings[col] = most_common.tolist()
                else:
                    self.category_mappings[col] = value_counts.index.tolist()
        print(self.category_mappings)
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for col, categories in self.category_mappings.items():
            X_encoded[col] = np.where(X_encoded[col].isin(categories), X_encoded[col], 'other')
            dummies = pd.get_dummies(X_encoded[col], prefix=col)
            X_encoded = pd.concat([X_encoded, dummies], axis=1)
            X_encoded.drop(col, axis=1, inplace=True)
        return X_encoded


class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, max_ohe=5):
        self.target_column = target_column
        self.outliers_preprocessor = OutliersPreprocessor(target_column)
        self.min_max_scaler = CustomMinMaxScaler(exclude_columns=[target_column])
        self.ohe_encoder = CustomOneHotEncoder(exclude_columns=[target_column], max_classes=max_ohe)

    def fit(self, df, y=None):
        self.outliers_preprocessor.fit(df)
        self.min_max_scaler.fit(df)
        self.ohe_encoder.fit(df)
        return self

    def transform(self, df):
        df_transformed = self.outliers_preprocessor.transform(df)
        df_transformed = self.min_max_scaler.transform(df_transformed)
        df_transformed = self.ohe_encoder.transform(df_transformed)
        return df_transformed


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns1=None, columns0=None, columns_mode=None, columns_mean=None, columns_na=None):
        self.columns1 = columns1 if columns1 else []
        self.columns0 = columns0 if columns0 else []
        self.columns_mode = columns_mode if columns_mode else []
        self.columns_mean = columns_mean if columns_mean else []
        self.columns_na = columns_na if columns_na else []

        self.mode_values = {}
        self.mean_values = {}

    def fit(self, X, y=None):
        for column in self.columns_mode:
            if column in X.columns:
                self.mode_values[column] = X[column].mode()[0]
        for column in self.columns_mean:
            if column in X.columns:
                self.mean_values[column] = X[column].mean()
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in self.columns1:
            if column in X_transformed.columns:
                X_transformed[column].fillna(value=-1, inplace=True)
        for column in self.columns0:
            if column in X_transformed.columns:
                X_transformed[column].fillna(value=0, inplace=True)
        for column in self.columns_mode:
            if column in X_transformed.columns:
                X_transformed[column].fillna(value=self.mode_values[column], inplace=True)
        for column in self.columns_mean:
            if column in X_transformed.columns:
                X_transformed[column].fillna(value=self.mean_values[column], inplace=True)
        for column in self.columns_na:
            if column in X_transformed.columns:
                X_transformed[column].fillna(value="NA", inplace=True)
        return X_transformed
