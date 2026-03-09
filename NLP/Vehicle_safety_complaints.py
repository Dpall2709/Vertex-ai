## Goal: Use complaint narratives to classify the component involved

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

df = pd.read_csv("complaints.csv")
df = df.dropna(subset=["summary","components"])

X_train, X_test, y_train, y_test = train_test_split(
    df["summary"], df["components"], test_size=0.2, random_state=42, stratify=df["components"]
)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(min_df=3, ngram_range=(1,2), stop_words="english")),
    ("lr", LogisticRegression(max_iter=200))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)
print(classification_report(y_test, pred))