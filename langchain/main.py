import numpy as np
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

from langchain_community.llms import HuggingFaceHub
def setup_rag():
    # 1. Load dokumen
    loader = TextLoader("dokumen.txt")
    documents = loader.load()
    
    # 2. Split dokumen menjadi chunk
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    # 3. Buat embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # 4. Buat vector store
    db = FAISS.from_documents(texts, embeddings)
    
    # 5. Setup retriever
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # 6. Setup LLM dan chain
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

# Fungsi untuk query
def query_rag(qa_chain, question):
    response = qa_chain.run(question)
    return response

# Contoh penggunaan
if __name__ == "__main__":
    qa_chain = setup_rag()
    question = "Bagaimana cara melakukan pertolongan pertama pada luka bakar ringan?"
    answer = query_rag(qa_chain, question)
    print("================================================")
    print(answer)