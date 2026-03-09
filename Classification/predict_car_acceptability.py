## Goal: Classify a car as unacceptable / acceptable / good / very good from categorical attributes 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

cols = ["buying","maint","doors","persons","lug_boot","safety","class"]
df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
    names=cols
)

X = df.drop("class", axis=1)
y = df["class"]

pre = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)],
    remainder="drop"
)

clf = Pipeline(steps=[
    ("pre", pre),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42))
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf.fit(X_tr, y_tr)
y_pred = clf.predict(X_te)
print(classification_report(y_te, y_pred))