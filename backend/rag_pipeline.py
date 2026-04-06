import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

DB_PATH = "backend/vector_db"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index(f"{DB_PATH}/index.faiss")

# Load chunks
with open(f"{DB_PATH}/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# LLM
llm = Ollama(model="tinyllama")


def retrieve_chunks(query, k=3, threshold=1.5):

    query_vector = model.encode([query])

    distances, indices = index.search(query_vector, k)

    results = []

    for i, dist in zip(indices[0], distances[0]):
        if dist < threshold:   # ✅ filter irrelevant chunks
            results.append(chunks[i])

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

    docs = retrieve_chunks(query, 3)

    if not docs:
        return "This question is not related to the research papers."

    return generate_answer(query)