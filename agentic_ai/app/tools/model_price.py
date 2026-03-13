import pandas as pd, numpy as np, joblib, json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
NUM=['wheel-base','length','width','height','curb-weight','engine-size','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg']
CAT=['make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','engine-type','num-of-cylinders','fuel-system']
def train_price_model(df_path,model_dir):
    df=pd.read_parquet(df_path)
    X=df[NUM+CAT].copy()
    y=df['price']
    X[NUM]=X[NUM].astype(float).fillna(X[NUM].median())
    X[CAT]=X[CAT].fillna('missing')
    X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.2)
    m=CatBoostRegressor(iterations=300,verbose=False)
    m.fit(X_tr,y_tr,cat_features=[X.columns.get_loc(c) for c in CAT])
    pred=m.predict(X_te)
    rmse=float(np.sqrt(mean_squared_error(y_te,pred)))
    model_dir.mkdir(parents=True,exist_ok=True)
    m.save_model(str(model_dir/'model.cbm'))
    meta={'num':NUM,'cat':CAT,'med':X[NUM].median().to_dict()}
    joblib.dump(meta,model_dir/'meta.pkl')
    return rmse