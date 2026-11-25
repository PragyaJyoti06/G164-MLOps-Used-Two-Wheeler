import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

MODEL_NAME = "mp233_used_two_wheeler"

# Set version 1 as champion
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="champion",
    version="1"
)

# Verify alias
champion = client.get_model_version_by_alias(
    name=MODEL_NAME,
    alias="champion"
)

print("Champion model registered successfully!")
print("Model name:", champion.name)
print("Model version:", champion.version)
print("Alias:", champion.aliases)