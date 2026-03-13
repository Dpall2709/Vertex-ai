import pandas as pd
URL='https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
C=[ 'symboling','normalized-losses','make','fuel-type','aspiration','num-of-doors','body-style','drive-wheels','engine-location','wheel-base','length','width','height','curb-weight','engine-type','num-of-cylinders','engine-size','fuel-system','bore','stroke','compression-ratio','horsepower','peak-rpm','city-mpg','highway-mpg','price']
def fetch_uci_automobile(): 
    df=pd.read_csv(URL,names=C,na_values='?')
    df=df.dropna(subset=['price'])
    df['price']=df['price'].astype(float)
    return df