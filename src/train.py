import os
import pandas as pd
import mlflow
import joblib
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Set dynamic project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)  # Fixed typo: os.makedirs

# Tracking setup
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")  # Used dynamic path DB_PATH
experiment_name = "Advertising_Sale_Regression"
registered_model_name = "Sales_prediction_Model"
mlflow.set_experiment(experiment_name)

# Data preparation
df = pd.read_csv(DATA_PATH)
X, y = df[["TV", "Radio", "Newspaper"]], df["Sales"]
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

# Train candidate models
models = {
    "Linear_Regression": LinearRegression(),
    "Ridge_Regression": Ridge(alpha=1.0),
    "Random_Forest": RandomForestRegressor(max_depth=5, random_state=42)
}

batch_runs = []
for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(xtrain, ytrain)
        rmse = root_mean_squared_error(ytest, model.predict(xtest))

        mlflow.log_param("model_type", name)
        mlflow.log_metric("test_rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")
        batch_runs.append((run.info.run_id, rmse))

# Find the best model
batch_runs.sort(key=lambda x: x[1])  # Sort by lowest RMSE
best_run_id, best_rmse = batch_runs[0]

# Register winning model as challenger
client = MlflowClient()
challenger_model = mlflow.register_model(
    model_uri=f"runs:/{best_run_id}/model",
    name=registered_model_name
)
challenger_version = challenger_model.version

# Assign challenger alias
client.set_registered_model_alias(registered_model_name, "challenger", challenger_version)
print(f"Best batch run {best_run_id} registered as challenger (v{challenger_version}, RMSE: {best_rmse:.4f})")

# Challenger vs Champion evaluation gate
try:
    champion_info = client.get_model_version_by_alias(registered_model_name, "champion")
    champion_run = client.get_run(champion_info.run_id)
    champion_rmse = champion_run.data.metrics["test_rmse"]  # Fixed: metrics
    champion_version = champion_info.version

    print(f"Current Champion: Version {champion_version} (RMSE: {champion_rmse:.4f})")

    if best_rmse < champion_rmse:
        client.set_registered_model_alias(registered_model_name, "champion", challenger_version)  # Fixed space/typo
        print(f"Title Change! Challenger (v{challenger_version}) defeated Champion (v{champion_version})")
    else:
        print(f"Defended! Champion (v{champion_version}) retains its title.")
except Exception:
    # First time running
    client.set_registered_model_alias(registered_model_name, "champion", challenger_version)
    print(f"No existing champion found. Version {challenger_version} crowned as first Champion!")

# Load current champion from MLflow registry
champion_model_uri = f"models:/{registered_model_name}@champion"
champion_model = mlflow.sklearn.load_model(champion_model_uri)

# Save standalone champion artifact
champion_export_path = os.path.join(MODELS_DIR, "champion-model.pkl")
joblib.dump(champion_model, champion_export_path)

print(f"Exported registry champion model to {champion_export_path}")