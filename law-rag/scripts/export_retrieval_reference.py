import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT = Path("/d/hpc/projects/onj_fri/pad-siew/law-rag")

EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_INDEX = PROJECT / "index/real_estate_full.faiss"
DEFAULT_METADATA = PROJECT / "index/real_estate_full_metadata.pkl"
DEFAULT_QUESTIONS = PROJECT / "data/eval/questions_30.jsonl"
DEFAULT_OUTPUT = PROJECT / "data/outputs/retrieval_reference_top10_questions30.jsonl"


def load_questions(path: Path):
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def normalize_text(value: str) -> str:
    return (value or "").strip().lower()


def expected_match(chunk: dict, expected_source: str, expected_article: str) -> bool:
    expected_source = normalize_text(expected_source)
    expected_article = normalize_text(str(expected_article)).replace(".", "")

    source = normalize_text(chunk.get("source", ""))
    article = normalize_text(str(chunk.get("article", ""))).replace(".", "")

    source_ok = True
    article_ok = True

    if expected_source:
        source_ok = expected_source in source or source in expected_source

    if expected_article:
        article_ok = article == expected_article

    return source_ok and article_ok


def retrieve(question: str, embedder, index, metadata, top_k: int):
    q_emb = embedder.encode([question], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, ids = index.search(q_emb, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx < 0:
            continue

        chunk = dict(metadata[idx])
        chunk["rank"] = rank
        chunk["score"] = float(score)
        results.append(chunk)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--index", default=str(DEFAULT_INDEX))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    questions_path = Path(args.questions)
    index_path = Path(args.index)
    metadata_path = Path(args.metadata)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Questions: {questions_path}", flush=True)
    print(f"Index: {index_path}", flush=True)
    print(f"Metadata: {metadata_path}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Top-k: {args.top_k}", flush=True)

    questions = load_questions(questions_path)

    print("Loading index...", flush=True)
    index = faiss.read_index(str(index_path))

    print("Loading metadata...", flush=True)
    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)

    print(f"Chunks: {len(metadata)}", flush=True)

    print("Loading embedder...", flush=True)
    embedder = SentenceTransformer(EMBED_MODEL)

    total_with_expected = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    top10_hits = 0

    with output_path.open("w", encoding="utf-8") as out:
        for q in questions:
            qid = q.get("id", "")
            question = q["question"]
            expected_source = q.get("expected_source", "")
            expected_article = str(q.get("expected_article", ""))

            retrieved = retrieve(question, embedder, index, metadata, args.top_k)

            has_expected = bool(expected_source or expected_article)
            matches = []

            if has_expected:
                total_with_expected += 1
                matches = [
                    expected_match(chunk, expected_source, expected_article)
                    for chunk in retrieved
                ]

                top1_hits += int(any(matches[:1]))
                top3_hits += int(any(matches[:3]))
                top5_hits += int(any(matches[:5]))
                top10_hits += int(any(matches[:10]))

            record = {
                "id": qid,
                "question": question,
                "topic": q.get("topic", ""),
                "type": q.get("type", ""),
                "expected_source": expected_source,
                "expected_article": expected_article,
                "expected_in_top1": bool(any(matches[:1])) if has_expected else None,
                "expected_in_top3": bool(any(matches[:3])) if has_expected else None,
                "expected_in_top5": bool(any(matches[:5])) if has_expected else None,
                "expected_in_top10": bool(any(matches[:10])) if has_expected else None,
                "retrieved": [
                    {
                        "rank": c.get("rank"),
                        "score": c.get("score"),
                        "chunk_id": c.get("chunk_id") or c.get("id"),
                        "source": c.get("source"),
                        "article": c.get("article"),
                        "article_title": c.get("article_title") or c.get("title", ""),
                        "text_preview": c.get("text", "")[:500],
                        "text": c.get("text", ""),
                    }
                    for c in retrieved
                ],
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            print(f"{qid}: top1={record['expected_in_top1']} top3={record['expected_in_top3']} top5={record['expected_in_top5']} top10={record['expected_in_top10']}", flush=True)

    print("Done.", flush=True)
    print(f"Saved: {output_path}", flush=True)

    if total_with_expected:
        print("Retrieval metrics:", flush=True)
        print(f"Top-1: {top1_hits}/{total_with_expected} = {top1_hits / total_with_expected:.2%}", flush=True)
        print(f"Top-3: {top3_hits}/{total_with_expected} = {top3_hits / total_with_expected:.2%}", flush=True)
        print(f"Top-5: {top5_hits}/{total_with_expected} = {top5_hits / total_with_expected:.2%}", flush=True)
        print(f"Top-10: {top10_hits}/{total_with_expected} = {top10_hits / total_with_expected:.2%}", flush=True)


if __name__ == "__main__":
    main()
