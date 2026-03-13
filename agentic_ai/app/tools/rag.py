import pandas as pd,faiss,joblib
from sentence_transformers import SentenceTransformer
def load_or_build_index(p,index_dir):
    df=pd.read_parquet(p)
    texts=df['summary'].astype(str).tolist()
    index_dir.mkdir(parents=True,exist_ok=True)
    m=SentenceTransformer('all-MiniLM-L6-v2')
    emb=m.encode(texts,normalize_embeddings=True)
    idx=faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb.astype('float32'))
    joblib.dump(texts,index_dir/'texts.pkl')
    faiss.write_index(idx,str(index_dir/'index.faiss'))
    return m,idx,texts

def query_index(m,idx,texts,q,top_k=5): 
    e=m.encode([q],normalize_embeddings=True)
    D,I=idx.search(e.astype('float32'),top_k)
    return [(int(i),float(d),texts[i]) for i,d in zip(I[0],D[0]) if i>=0]