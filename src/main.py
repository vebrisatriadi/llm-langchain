from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import pickle
from contextlib import asynccontextmanager

MODEL_DIR = "/Users/vebrisatriadi/Documents/Portfolio/llm/"
# HOME_DIR = os.getenv("HOME_DIR", "/app")
# MODEL_DIR = os.path.join(HOME_DIR, "llm/langchain/saved_rag_model")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_chain
    qa_chain = load_model("./langchain/saved_rag_model")
    yield

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

def load_model(load_dir="./langchain/saved_rag_model"):
    """Memuat model RAG yang telah disimpan"""
    # Load konfigurasi
    with open(f"{load_dir}/config.pkl", "rb") as f:
        config = pickle.load(f)
    
    # Setup embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Load FAISS index
    vectorstore = FAISS.load_local(
        load_dir + "/faiss_index",
        embeddings,
        allow_dangerous_deserialization = True
    )
    
    # Setup retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=config["search_kwargs"]
    )
    
    # Setup LLM
    llm = HuggingFaceHub(
        repo_id=config["model_config"]["repo_id"],
        model_kwargs=config["model_config"]["model_kwargs"]
    )
    
    # Buat chain baru
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    return qa_chain

def query_rag(qa_chain, question):
    """Fungsi untuk query"""
    response = qa_chain.run(question)
    return response

@app.post("/query", response_model=QueryResponse)
def query_model(request: QueryRequest):
    try:
        answer = query_rag(qa_chain, request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)