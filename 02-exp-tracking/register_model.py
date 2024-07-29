import os
import pickle
import click
import mlflow
import psutil
import scipy.sparse
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from typing import Dict, Tuple

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

def log_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024 ** 2)} MB")  # Convert bytes to MB

def load_pickle(filename: str) -> Tuple:
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def convert_to_dense_if_needed(data):
    if scipy.sparse.issparse(data):
        return data.toarray()
    return data

def train_and_log_model(data_path: str, params: Dict) -> None:
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    # Convert to dense if data is sparse
    X_train = convert_to_dense_if_needed(X_train)
    X_val = convert_to_dense_if_needed(X_val)
    X_test = convert_to_dense_if_needed(X_test)

    mlflow.sklearn.autolog()  # Enable autologging

    try:
        with mlflow.start_run():
            for param in RF_PARAMS:
                if param in params:
                    params[param] = int(params[param])

            rf = RandomForestRegressor(**params)
            log_memory_usage()
            rf.fit(X_train, y_train)
            log_memory_usage()

            # Evaluate model on the validation and test sets
            val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
            mlflow.log_metric("val_rmse", val_rmse)
            test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
            mlflow.log_metric("test_rmse", test_rmse)
            log_memory_usage()

    except Exception as e:
        print(f"Exception during training and logging: {e}")

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int) -> None:

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register the best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="rf-best-model")

if __name__ == '__main__':
    run_register_model()
