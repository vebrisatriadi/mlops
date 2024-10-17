import pandas as pd
import mlflow

def load_data(path):
    mlflow.log_param("data_path", path)
    return pd.read_csv(path)