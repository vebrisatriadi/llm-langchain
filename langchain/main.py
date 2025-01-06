import json
import pickle
import numpy as np
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

def load_qa_data(file_path):
    """Load data dari file JSON QA"""
    documents = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data = json.loads(line)
                combined_text = f"Pertanyaan: {data['question']}\nJawaban: {data['answer']}"
                doc = Document(page_content=combined_text, metadata={"source": "qa_dataset"})
                documents.append(doc)
    return documents

def setup_rag():
    # Kode setup RAG yang sudah ada
    documents = load_qa_data("dokumen.txt")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    db = FAISS.from_documents(texts, embeddings)
    
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small", 
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    return qa_chain

def save_model(qa_chain, save_dir="./saved_model"):
    """Menyimpan model RAG"""
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Simpan FAISS index
    qa_chain.retriever.vectorstore.save_local(f"{save_dir}/faiss_index")
    
    # Simpan konfigurasi
    config = {
        "search_kwargs": qa_chain.retriever.search_kwargs,
        "model_config": {
            "repo_id": "google/flan-t5-small",
            "model_kwargs": {"temperature": 0.5, "max_length": 512}
        }
    }
    
    with open(f"{save_dir}/config.pkl", "wb") as f:
        pickle.dump(config, f)
    
    print(f"Model berhasil disimpan di: {save_dir}")

if __name__ == "__main__":
    # Training dan menyimpan model
    print("Training model...")
    qa_chain = setup_rag()
    
    print("Menyimpan model...")
    save_model(qa_chain, "./saved_rag_model")