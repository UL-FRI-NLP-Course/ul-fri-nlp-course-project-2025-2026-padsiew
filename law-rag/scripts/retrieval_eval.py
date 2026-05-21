import argparse
import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def norm(x):
    return str(x or "").strip().lower().replace(".", "")


def format_query(q, model_name):
    if "e5" in model_name.lower():
        return "query: " + q
    return q


def expected_match(chunk, q):
    # single-source legacy fields
    exp_source = q.get("expected_source", "")
    exp_article = str(q.get("expected_article", ""))

    # multi/new fields
    exp_sources = q.get("expected_sources") or ([exp_source] if exp_source else [])
    exp_articles = q.get("expected_articles") or ([exp_article] if exp_article else [])
    gold_chunk_ids = q.get("gold_chunk_ids") or []

    chunk_source = norm(chunk.get("source", ""))
    chunk_article = norm(chunk.get("article", ""))
    chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")

    if gold_chunk_ids and chunk_id in set(map(str, gold_chunk_ids)):
        return True

    for s in exp_sources:
        for a in exp_articles:
            source_ok = True if not s else norm(s) in chunk_source or chunk_source in norm(s)
            article_ok = True if not a else chunk_article == norm(a)
            if source_ok and article_ok:
                return True

    return False


def retrieve(question, model, model_name, index, metadata, top_k):
    qq = format_query(question, model_name)
    emb = model.encode([qq], normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")
    scores, ids = index.search(emb, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx < 0:
            continue
        c = dict(metadata[idx])
        c["rank"] = rank
        c["score"] = float(score)
        results.append(c)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--embed-model", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    index = faiss.read_index(args.index)
    with open(args.metadata, "rb") as f:
        metadata = pickle.load(f)

    model = SentenceTransformer(args.embed_model)

    totals = {"top1": 0, "top3": 0, "top5": 0, "top10": 0}
    n = 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out:
        for q in questions:
            retrieved = retrieve(q["question"], model, args.embed_model, index, metadata, args.top_k)
            matches = [expected_match(c, q) for c in retrieved]

            hits = {
                "top1": any(matches[:1]),
                "top3": any(matches[:3]),
                "top5": any(matches[:5]),
                "top10": any(matches[:10]),
            }

            for k in totals:
                totals[k] += int(hits[k])
            n += 1

            out.write(json.dumps({
                "id": q.get("id"),
                "question": q["question"],
                "type": q.get("type"),
                "source_mode": q.get("source_mode"),
                "embed_model": args.embed_model,
                "hits": hits,
                "retrieved": [
                    {
                        "rank": c["rank"],
                        "score": c["score"],
                        "chunk_id": c.get("chunk_id"),
                        "source": c.get("source"),
                        "article": c.get("article"),
                        "article_title": c.get("article_title", ""),
                    }
                    for c in retrieved
                ],
            }, ensure_ascii=False) + "\n")

    print("MODEL:", args.embed_model)
    print("QUESTIONS:", args.questions)
    print("N:", n)
    for k, v in totals.items():
        print(f"{k}: {v}/{n} = {v/n:.2%}")


if __name__ == "__main__":
    main()
