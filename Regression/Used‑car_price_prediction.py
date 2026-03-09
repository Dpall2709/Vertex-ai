## Predict used‑car selling price from make/model/year, km driven, fuel, transmission, owner type, etc.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
cols = [
    "symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors",
    "body-style","drive-wheels","engine-location","wheel-base","length","width","height",
    "curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore",
    "stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"
]
df = pd.read_csv(url, names=cols, na_values='?')

df = df.dropna(subset=["price"])
df["price"] = df["price"].astype(float)


num_cols = ["wheel-base","length","width","height","curb-weight","engine-size",
            "bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg"]
cat_cols = ["make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels",
            "engine-location","engine-type","num-of-cylinders","fuel-system"]


X = df[num_cols + cat_cols].copy()
y = df["price"].copy()


X[num_cols] = X[num_cols].astype(float)
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna("missing")


pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

model = Pipeline(steps=[
    ("pre", pre),
    ("rf", RandomForestRegressor(n_estimators=600, random_state=42))
])


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)


model.fit(X_tr, y_tr)
pred = model.predict(X_te)


mse = mean_squared_error(y_te, pred) 
rmse = np.sqrt(mse)               
r2 = r2_score(y_te, pred)
print(f"MSE: {mse:,.0f} | RMSE: {rmse:,.0f} | R²: {r2:.3f}")