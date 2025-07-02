import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
import joblib

df = pd.read_csv("data/crop_yield.csv")
X = df[["Crop", "Temperature", "Rainfall", "Humidity"]]
y = df["Yield"]

categorical = ["Crop"]
numerical = ["Temperature", "Rainfall", "Humidity"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(), categorical)
], remainder='passthrough')

base_models = [
    ("lr", LinearRegression()),
    ("rf", RandomForestRegressor(n_estimators=100)),
    ("dt", DecisionTreeRegressor()),
    ("nn", MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500))
]

stack_model = StackingRegressor(
    estimators=base_models,
    final_estimator=RandomForestRegressor(n_estimators=50)
)

pipeline = Pipeline([
    ("pre", preprocessor),
    ("model", stack_model)
])

pipeline.fit(X, y)
joblib.dump(pipeline, "models/stack_model.pkl")