# Research Paper RAG Assistant

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions from research papers.

## Tech Stack

- Sentence Transformers (Embeddings)
- FAISS (Vector Database)
- TinyLlama / Llama3 (LLM via Ollama)
- LangChain
- FastAPI
- Streamlit

## Architecture

User Query → Embedding → FAISS Retrieval → LLM → Answer

## How to Run

1. Install dependencies
2. Run create_vector_db.py
3. Start FastAPI backend
4. Run Streamlit frontend