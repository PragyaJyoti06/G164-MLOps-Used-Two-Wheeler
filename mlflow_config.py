
import mlflow

# Configure local SQLite tracking database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set experiment name
mlflow.set_experiment("MP233_Used_Two_Wheeler")

print("MLflow tracking URI:", mlflow.get_tracking_uri())
print("MLflow experiment configured successfully!")
