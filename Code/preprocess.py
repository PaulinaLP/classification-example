from Code import preprocessing
from Code import functions
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd


def preprocess(df_train, target):
    numeric_columns = df_train.select_dtypes(include=['number']).columns.tolist()
    numeric_columns.remove(target)
    categorical_columns = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    # Shuffle the DataFrame
    df = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    # Split data into features and target
    X = df.drop(columns=[target])
    y = df[target]
    # Split data into train and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y,
                                                        shuffle=True)
    # dropping cat columns with too many categories
    column_dropper = preprocessing.CategoryPreprocessor(threshold=0.5, columns=categorical_columns)
    X_train = column_dropper.fit_transform(X_train)
    X_test = column_dropper.transform(X_test)
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('outliers', preprocessing.OutliersPreprocessor(target))
            ]), numeric_columns),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(max_categories=5, handle_unknown='infrequent_if_exist'))
            ]), categorical_columns),
        ]
    )
    # Pipeline to handle the entire process
    # to do: include column dropper into the pipeline
    preprocessing_pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    preprocessing_pipeline.fit(X_train, y_train)
    # Get the feature names
    feature_names = functions.get_feature_names(preprocessor, numeric_columns, categorical_columns)
    # Transform the data
    X_train_preprocessed = preprocessing_pipeline.transform(X_train)
    # to do: include oversampling
    X_test_preprocessed = preprocessing_pipeline.transform(X_test)
    # Print the first 5 rows of the preprocessed data with descriptive column names
    print(pd.DataFrame(X_train_preprocessed, columns=feature_names).head(5))
    print(pd.DataFrame(X_test_preprocessed, columns=feature_names).head(5))
    return preprocessing_pipeline,   X_train_preprocessed , X_test_preprocessed, y_train, y_test
