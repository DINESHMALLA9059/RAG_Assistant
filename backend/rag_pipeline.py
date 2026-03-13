import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

DB_PATH = "vector_db"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index(f"{DB_PATH}/index.faiss")

# Load chunks
with open(f"{DB_PATH}/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# LLM
llm = Ollama(model="tinyllama")


def retrieve_chunks(query, k=3):

    query_vector = model.encode([query])

    distances, indices = index.search(query_vector, k)

    results = [chunks[i] for i in indices[0]]

    return results


def generate_answer(query):

    docs = retrieve_chunks(query)

    context = "\n\n".join(docs)

    prompt = f"""
Answer the question ONLY using the context below.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response


def ask_question(query):

    docs = retrieve_chunks(query, 1)

    if len(docs) == 0:
        return "This question is not related to the research papers."

    return generate_answer(query)