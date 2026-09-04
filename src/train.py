import pandas as pd
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv(r"D:\\MLops Day 1\\Data\\data.csv")

# Train-test split
X = df.drop(columns=["Sales"])
y = df["Sales"]
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=67)

# Linear Regression model
model = LinearRegression()
model.fit(xtrain, ytrain)

# Model dump
joblib.dump(model, r"D:\MLops Day 1\Models\linear_reg_model.pkl") 