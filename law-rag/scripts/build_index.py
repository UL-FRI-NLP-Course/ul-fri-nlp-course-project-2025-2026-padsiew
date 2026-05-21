import re
import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def format_passages(texts, model_name):
    if "e5" in model_name.lower():
        return [f"passage: {t}" for t in texts]
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input chunks JSONL")
    parser.add_argument("--index-output", required=True, help="Output FAISS index")
    parser.add_argument("--metadata-output", required=True, help="Output metadata pickle")
    parser.add_argument("--embed-model", default="BAAI/bge-m3")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-embed-chars", type=int, default=3000)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args()

    input_path = Path(args.input)
    index_output = Path(args.index_output)
    metadata_output = Path(args.metadata_output)

    index_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    records = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            raw_text = obj.get("text", "")
            raw_text = re.sub(r"\s+", " ", raw_text).strip()
            raw_text = raw_text[:args.max_embed_chars]

            text_for_embedding = (
                f"Vir: {obj.get('source', '')}. "
                f"Člen: {obj.get('article', '')}. "
                f"Naslov člena: {obj.get('article_title', '')}. "
                f"Besedilo: {raw_text}"
            )
            obj["text_for_embedding"] = text_for_embedding
            records.append(obj)

    print(f"Loaded chunks: {len(records)}", flush=True)
    print(f"Loading embedder: {args.embed_model}", flush=True)
    embedder = SentenceTransformer(args.embed_model)
    embedder.max_seq_length = args.max_seq_length
    print(f"Embedder max_seq_length: {embedder.max_seq_length}", flush=True)

    texts = [r["text_for_embedding"] for r in records]
    texts = format_passages(texts, args.embed_model)

    print("Encoding chunks...", flush=True)
    embeddings = embedder.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    print(f"Embeddings shape: {embeddings.shape}", flush=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print(f"Writing FAISS index: {index_output}", flush=True)
    faiss.write_index(index, str(index_output))

    print(f"Writing metadata: {metadata_output}", flush=True)
    with metadata_output.open("wb") as f:
        pickle.dump(records, f)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()