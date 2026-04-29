import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input chunks JSONL")
    parser.add_argument("--index-out", required=True, help="Output FAISS index path")
    parser.add_argument("--metadata-out", required=True, help="Output metadata pickle path")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Embedding model")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    input_path = Path(args.input)
    index_path = Path(args.index_out)
    metadata_path = Path(args.metadata_out)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    records = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            # Include metadata in embedding text so retrieval can match article/source names too.
            text_for_embedding = (
                f"{obj.get('source', '')}. "
                f"{obj.get('article', '')}. člen. "
                f"{obj.get('article_title', '')}. "
                f"{obj.get('text', '')}"
            )

            obj["text_for_embedding"] = text_for_embedding
            records.append(obj)

    print(f"Loaded chunks: {len(records)}", flush=True)

    if not records:
        raise ValueError("No records loaded. Check input path or filtering.")

    print(f"Loading embedding model: {args.model}", flush=True)
    model = SentenceTransformer(args.model)

    texts = [r["text_for_embedding"] for r in records]

    print("Creating embeddings...", flush=True)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")
    print(f"Embeddings shape: {embeddings.shape}", flush=True)

    # Because embeddings are normalized, inner product = cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_path))

    with metadata_path.open("wb") as f:
        pickle.dump(records, f)

    print(f"Saved FAISS index: {index_path}", flush=True)
    print(f"Saved metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
