# AHD's Document Genie (Local RAG Chatbot)

A Streamlit chatbot that answers questions about an uploaded PDF using Retrieval-Augmented Generation (RAG).

## Features
- Upload PDF → extract text → chunk with overlap
- Create embeddings using `nomic-embed-text` (Ollama)
- Retrieve top-k relevant chunks via cosine similarity
- Inject retrieved context into the prompt for grounded answers
- Shows which chunks were used (Sources expander)

## Tech Stack
- Python
- Streamlit
- Ollama (`llama3.2:3b`)
- Embeddings: `nomic-embed-text`
- Retrieval: cosine similarity (top-k)

## How to Run (Local)
1. Install and run Ollama
2. Pull models:
   - `ollama pull llama3.2:3b`
   - `ollama pull nomic-embed-text`
3. Install dependencies:
   - `pip install -r requirements.txt`
4. Start the app:
   - `streamlit run app.py`

## Notes on Deployment
This app uses Ollama via `http://localhost:11434`, so it is designed to run locally.
To deploy online, replace the LLM + embeddings with a hosted API or host Ollama on a server.
