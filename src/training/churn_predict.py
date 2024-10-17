import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Set up MLflow tracking server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("telco_churn_prediction")

def load_data(url):
    data = pd.read_csv(url)
    
    # Preprocessing
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data = data.dropna()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        data[col] = le.fit_transform(data[col])
    
    X = data.drop(['customerID', 'Churn'], axis=1)
    y = (data['Churn'] == 'Yes').astype(int)
    
    return train_test_split(X, y, test_size=0.4, random_state=42)

def train_model(X_train, y_train, params):
    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1

def register_model(run_id, model_name):
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    model_details = mlflow.register_model(model_uri, model_name)
    return model_details.version

def promote_model(model_name, version, stage):
    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )

def mlops_pipeline(data_url, model_name):
    # Load and split data
    X_train, X_test, y_train, y_test = load_data(data_url)
    
    # Define model parameters
    params = {
        "n_estimators": 1000,
        "max_depth": 8,
        "random_state": 42
    }
    
    # Make sure no run is active
    mlflow.end_run()
    
    # Train model and log to MLflow
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_params(params)
        
        # Train and evaluate model
        model = train_model(X_train, y_train, params)
        accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model", signature=infer_signature(X_train, y_train))
        
        print(f"Model performance: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        # Register model
        model_version = register_model(run.info.run_id, model_name)
        
        # Promote model to staging
        promote_model(model_name, model_version, "Staging")
        
        print(f"Model {model_name} version {model_version} promoted to Staging")

if __name__ == "__main__":
    data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    mlops_pipeline(data_url, "telco_churn_prediction_model")