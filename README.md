# LLM with Langchain with HuggingFace models

- Python 3.11.5
- Using data in file `dokumen.txt` with JSON format

## Description
- Project to build simple RAG model
- Building RAG Model with Langchain framework
- Datasets including medical information
- Using `google/flan-t5-small` from `HuggingFace` repo for smaller computation

## How to Run
1. Activate your virtual environment
2. Set your `HuggingFace` API Key Token:
```
export HUGGINGFACEHUB_API_TOKEN=
```
3. Go to `langchain` folder
4. Install Dependencies
```
pip install -r requirements.txt
```
5. Build and save the RAG model
```
python main.py
```
6. Test the saved RAG model
```
python test.py
```

## Run the Docker image
Set your `HuggingFace` API Key Token in Dockerfile
```
docker build -t llm-app .
```
```
docker run -p 5050:5050 llm-app
```