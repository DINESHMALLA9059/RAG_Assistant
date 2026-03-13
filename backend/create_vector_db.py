import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data"
DB_PATH = "vector_db"

documents = []

# Load PDFs
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

texts = [chunk.page_content for chunk in chunks]

# Load sentence transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text → embeddings
embeddings = model.encode(texts)

dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, f"{DB_PATH}/index.faiss")

# Save text chunks
with open(f"{DB_PATH}/chunks.pkl", "wb") as f:
    pickle.dump(texts, f)

print("FAISS index created successfully")