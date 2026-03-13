import pandas as pd,requests,zipfile,io
URL='https://www-odi.nhtsa.dot.gov/downloads/folders/Complaints/FLAT_CMPL.zip'
def fetch_nhtsa_complaints():
    r=requests.get(URL); 
    z=zipfile.ZipFile(io.BytesIO(r.content))
    n=[x for x in z.namelist() if x.endswith('.csv')][0]
    df=pd.read_csv(z.open(n),dtype=str)
    df=df.rename(columns=str.lower)
    df=df.dropna(subset=['summary','components'])
    return df[['summary','components']]