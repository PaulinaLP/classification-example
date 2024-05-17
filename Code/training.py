import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, roc_auc_score


def run_mlflow_experiment(param_grid, uri, experiment_name, df, target_name):
    # Set the MLflow tracking URI and experiment name
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    # Shuffle the DataFrame
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split data into features and target
    X = df.drop(columns=[target_name])
    y = df[target_name]

    # Split data into train and test sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

    # Initialize RandomForestClassifier
    rf = RandomForestClassifier(random_state=42)

    # Define scoring metrics
    scoring = {
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'roc_auc': make_scorer(roc_auc_score)
    }

    # Initialize GridSearchCV with 10-fold cross-validation
    grid_search = GridSearchCV(rf, param_grid, cv=10, n_jobs=-1, scoring=scoring, refit='roc_auc')

    # Start MLflow run
    with mlflow.start_run():

        # Perform grid search
        grid_search.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(grid_search.best_params_)

        # Log cross-validation metrics
        for metric_name, scores in grid_search.cv_results_.items():
            if 'mean_test_' in metric_name:
                mlflow.log_metric(metric_name, scores[grid_search.best_index_])

        # Log test set metrics
        y_pred_test = grid_search.best_estimator_.predict(X_test)
        mlflow.log_metric('test_precision', precision_score(y_test, y_pred_test))
        mlflow.log_metric('test_recall', recall_score(y_test, y_pred_test))
        mlflow.log_metric('test_f1', f1_score(y_test, y_pred_test))
        mlflow.log_metric('test_roc_auc', roc_auc_score(y_test, y_pred_test))

        # Log training set metrics
        y_pred_train = grid_search.best_estimator_.predict(X_train)
        mlflow.log_metric('train_precision', precision_score(y_train, y_pred_train))
        mlflow.log_metric('train_recall', recall_score(y_train, y_pred_train))
        mlflow.log_metric('train_f1', f1_score(y_train, y_pred_train))
        mlflow.log_metric('train_roc_auc', roc_auc_score(y_train, y_pred_train))

        # Log best model
        mlflow.sklearn.log_model(grid_search.best_estimator_, 'random_forest_model')