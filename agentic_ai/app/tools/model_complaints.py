import pandas as pd,joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
def train_complaint_classifier(df_path,model_dir): 
    df=pd.read_parquet(df_path)
    X=df['summary'] 
    y=df['components']
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2,stratify=y)
    p=Pipeline([('tfidf',TfidfVectorizer()),('lr',LogisticRegression(max_iter=500))])
    p.fit(X_tr,y_tr) 
    model_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump(p,model_dir/'clf.pkl')