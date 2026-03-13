from fastapi import FastAPI
from app.tools.ingest_nhtsa import fetch_nhtsa_complaints
from app.tools.ingest_usedcars import fetch_uci_automobile
from app.tools.model_price import train_price_model
from app.tools.model_complaints import train_complaint_classifier
from app.tools.rag import load_or_build_index, query_index
from app.schemas import AskRequest, AskResponse
from pathlib import Path
app=FastAPI()
base=Path(__file__).resolve().parent.parent
@app.post('/ask')
def ask(req:AskRequest):
    df_path=base/'data/processed/complaints.parquet'
    model,index,texts=load_or_build_index(df_path, base/'index/faiss')
    hits=query_index(model,index,texts,req.query,top_k=req.top_k)
    ans=' '.join(t for _,_,t in hits)
    return AskResponse(answer=ans,sources=[{'idx':i,'text':t} for i,_,t in hits])