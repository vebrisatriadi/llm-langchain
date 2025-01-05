import os
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import logging

# Setup logging buat debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_path():
    base_path = "/Users/vebrisatriadi/.llama/checkpoints"
    try:
        # Cek isi folder
        files = os.listdir(base_path)
        logger.info(f"Files in directory: {files}")
        
        # Cari file model
        model_files = [f for f in files if f.startswith("Llama3.2-3B")]
        if not model_files:
            logger.error("Model file tidak ditemukan!")
            return None
            
        return os.path.join(base_path, model_files[0])
    except Exception as e:
        logger.error(f"Error checking path: {str(e)}")
        return None

def initialize_llama():
    try:
        # Cek model path
        model_path = check_model_path()
        if not model_path:
            raise ValueError("Model path tidak valid!")
            
        # Initialize LlamaCpp
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.75,
            max_tokens=2000,
            n_ctx=2048,
            n_threads=4,  # Sesuaikan dengan CPU
            verbose=True
        )
        
        return llm
        
    except Exception as e:
        logger.error(f"Error initializing Llama: {str(e)}")
        raise

def test_prompt():
    try:
        llm = initialize_llama()
        
        # Test prompt template
        template = PromptTemplate(
            input_variables=["product"],
            template="Berikan review singkat tentang {product}."
        )
        
        chain = LLMChain(llm=llm, prompt=template)
        
        # Test sederhana
        response = chain.invoke({"product": "laptop gaming"})
        print(response['text'])
        
    except Exception as e:
        print(f"Error running prompt: {str(e)}")

if __name__ == "__main__":
    test_prompt()