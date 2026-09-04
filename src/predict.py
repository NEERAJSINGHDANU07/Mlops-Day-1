import pandas as pd
import mlflow
import numpy as np
import mlflow.sklearn

# MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load Champion model from MLflow Model Registry
model = mlflow.sklearn.load_model(
    "models:/Sales_prediction_Model@champion"
)

# New observation
new_data = pd.DataFrame({
    "TV": [35],
    "Radio": [50000],
    "Newspaper": [8]
})

# Prediction
prediction = model.predict(new_data)

print("Predicted Sales:", prediction[0])
