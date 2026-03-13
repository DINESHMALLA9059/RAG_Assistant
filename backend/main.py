from fastapi import FastAPI
from backend.rag_pipeline import ask_question

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Research Paper RAG API is running"}

@app.get("/ask")
def ask(query: str):
    answer = ask_question(query)
    return {"response": answer}