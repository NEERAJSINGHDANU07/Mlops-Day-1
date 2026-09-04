import pandas as pd
import mlflow
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# step tracking 
mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment_name = "Advertising_Sale_Regression"
registered_model_name = "Sales_prediction_Model"
mlflow.set_experiment(experiment_name)

# data preprocesssing 
df = pd.read_csv(r"D:\MLops Day 1\Data\data.csv")
X,y = df[["TV","Radio","Newspaper"]],df["Sales"]
xtrain, xtest, ytrain, ytest = train_test_split(X,y, test_size = 0.2, random_state = 42)
 
# Train candidate models (log only , do nor register yet )
modles = {
    "Linear_Regression" : LinearRegression(),
    "Ridge_Regression" : Ridge(alpha= 1.0),
    "Random_Forest" : RandomForestRegressor(max_depth = 5, random_state = 42)
}
batch_runs=[]
for name, model in modles.items():
    with mlflow.start_run(run_name= name) as run:
        model.fit(xtrain,ytrain)
        rmse = root_mean_squared_error(ytest, model.predict(xtest))

        mlflow.log_param("model_type", name)
        mlflow.log_metric("test_rmse", rmse)
        # NOTICS : NO REGISTERD_MODEL_NAME HERE 
        mlflow.sklearn.log_model(model,artifact_path="model")
        batch_runs.append((run.info.run_id,rmse))

#4 FIND THE SINGLE BEST MODEL FORM THIS BATCH 
batch_runs.sort(key = lambda x :x[1]) # short by lowest rmse
best_run_id, best_rmse = batch_runs[0]

#5 REGISTER ONLY TH EWINNING RUN AS THE CHALLENGER 
client = MlflowClient()
challenger_model = mlflow.register_model(
    model_uri = f"runs:/{best_run_id}/model",
    name= registered_model_name

)
challenger_version = challenger_model.version
 
 # ASSIGN CHALLENGER ALIAS
client.set_registered_model_alias(registered_model_name, "challenger", challenger_version)
print(f"Best bach run{best_run_id} registerd as challenger (v{challenger_version}, RMSE: {best_rmse:4f})")

# challenger vs champion evalution gate
try:
    champion_info = client.get_model_version_by_alias(registered_model_name,"champion")
    champion_run = client.get_run(champion_info.run_id)
    champion_rmse = champion_run.data.matrics["test_rmse"]
    champion_version = champion_info.version

    print(f"Current Champion: Version {champion_version} (RMSE: {champion_rmse :4f})")

    if best_rmse < champion_rmse:
        client.set_registered_model_alias(registered_model_name, " chmapion", challenger_version)
        print(f"Title Change:! Challenger (v{challenger_version}) defeted Champion (v{champion_version})")
    else:
        print(f"Defended ! Champion (v{champion_version}) retains its title.")
except Exception:
    # First time running 
    client.set_registered_model_alias(registered_model_name, "champion", challenger_version)
    print(f"No existing champion found. Version { challenger_version} crowned as first Champion!")
    

