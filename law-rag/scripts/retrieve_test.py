import argparse
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_QUESTIONS = [
    "Kako se vpiše pogodbena komasacija?",
    "Kaj ureja kataster nepremičnin?",
    "Kaj pomeni služnost?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="FAISS index path")
    parser.add_argument("--metadata", required=True, help="Metadata pickle path")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Embedding model")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--question", action="append", help="Custom question; can be repeated")
    args = parser.parse_args()

    questions = args.question if args.question else DEFAULT_QUESTIONS

    print(f"Loading FAISS index: {args.index}", flush=True)
    index = faiss.read_index(str(args.index))

    print(f"Loading metadata: {args.metadata}", flush=True)
    with Path(args.metadata).open("rb") as f:
        records = pickle.load(f)

    print(f"Loaded records: {len(records)}", flush=True)

    print(f"Loading embedding model: {args.model}", flush=True)
    model = SentenceTransformer(args.model)

    for question in questions:
        print("\n" + "=" * 100)
        print("QUESTION:", question)

        q_emb = model.encode(
            [question],
            normalize_embeddings=True,
        )
        q_emb = np.asarray(q_emb, dtype="float32")

        scores, ids = index.search(q_emb, args.top_k)

        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            if idx < 0:
                continue

            r = records[idx]

            print("\n" + "-" * 80)
            print(f"Rank: {rank}")
            print(f"Score: {float(score):.4f}")
            print(f"Chunk ID: {r.get('chunk_id', '')}")
            print(f"Source: {r.get('source', '')}")
            print(f"Article: {r.get('article', '')}")
            print(f"Article title: {r.get('article_title', '')}")
            print(f"URL: {r.get('source_url', '')}")
            print("Text preview:")
            print(r.get("text", "")[:900])


if __name__ == "__main__":
    main()
