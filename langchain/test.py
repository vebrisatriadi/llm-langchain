# test.py

import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import pickle

def load_model(load_dir="./saved_rag_model"):
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

if __name__ == "__main__":
    # Set Hugging Face token
    
    # Load model
    print("Loading model...")
    qa_chain = load_model("./saved_rag_model")
    print("Model loaded successfully!")
    
    # Test beberapa pertanyaan
    test_questions = [
        "apa itu hipertensi?",
        "gmana ya cara mengobati luka bakar berat?",
        "diabetes tipe 2 gejalanya apa?"
    ]
    
    print("\nMulai testing model...")
    print("=" * 50)
    
    for question in test_questions:
        print("\nQ:", question)
        answer = query_rag(qa_chain, question)
        print("A:", answer)
        print("-" * 50)