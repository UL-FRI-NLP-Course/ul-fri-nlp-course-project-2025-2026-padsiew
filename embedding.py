import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ===== CONFIG =====
INPUT_FILE = "real_estate_chunks.jsonl"
INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.json"

# ===== LOAD DATA =====
texts = []
metadata = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["text"])
        metadata.append(obj)

print(f"Loaded {len(texts)} chunks")


# ===== LOAD EMBEDDING MODEL =====
print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# ===== CREATE EMBEDDINGS =====
print("Creating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

embeddings = np.array(embeddings).astype("float32")

print("Embeddings shape:", embeddings.shape)


# ===== BUILD FAISS INDEX =====
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index built")


# ===== SAVE INDEX =====
faiss.write_index(index, INDEX_FILE)

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False)

print("Saved index and metadata")