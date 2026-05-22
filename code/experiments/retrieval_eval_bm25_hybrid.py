import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


TOKEN_RE = re.compile(r"[a-zA-ZčšžćđČŠŽĆĐ0-9]+")


def tokenize(text: str):
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 1]


def norm(x):
    return str(x or "").strip().lower().replace(".", "")


def format_query(q, model_name):
    if model_name and "e5" in model_name.lower():
        return "query: " + q
    return q


def expected_match(chunk, q):
    exp_source = q.get("expected_source", "")
    exp_article = str(q.get("expected_article", ""))

    exp_sources = q.get("expected_sources") or ([exp_source] if exp_source else [])
    exp_articles = q.get("expected_articles") or ([exp_article] if exp_article else [])
    gold_chunk_ids = set(map(str, q.get("gold_chunk_ids") or []))

    chunk_source = norm(chunk.get("source", ""))
    chunk_article = norm(chunk.get("article", ""))
    chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "")

    if gold_chunk_ids and chunk_id in gold_chunk_ids:
        return True

    for s in exp_sources:
        for a in exp_articles:
            source_ok = True if not s else norm(s) in chunk_source or chunk_source in norm(s)
            article_ok = True if not a else chunk_article == norm(a)
            if source_ok and article_ok:
                return True

    return False


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bm25_search(query, bm25, top_k=10, k1=1.5, b=0.75):
    scores = defaultdict(float)
    terms = tokenize(query)

    doc_lens = bm25["doc_lens"]
    avgdl = bm25["avgdl"]
    idf = bm25["idf"]
    inverted = bm25["inverted"]

    for term in terms:
        postings = inverted.get(term)
        if not postings:
            continue

        term_idf = idf.get(term, 0.0)

        for doc_id, tf in postings.items():
            dl = doc_lens[int(doc_id)]
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            scores[int(doc_id)] += term_idf * (tf * (k1 + 1)) / denom

    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return best


def dense_search(query, model, model_name, index, top_k):
    q = format_query(query, model_name)
    emb = model.encode([q], normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")
    scores, ids = index.search(emb, top_k)

    out = []
    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            out.append((int(idx), float(score)))
    return out


def normalize_scores(items):
    if not items:
        return {}
    max_score = max(score for _, score in items)
    if max_score <= 0:
        return {idx: 0.0 for idx, _ in items}
    return {idx: score / max_score for idx, score in items}


def make_results(items, metadata):
    results = []
    for rank, (idx, score) in enumerate(items, start=1):
        c = dict(metadata[idx])
        c["rank"] = rank
        c["score"] = float(score)
        results.append(c)
    return results


def calc_hits(retrieved, q):
    matches = [expected_match(c, q) for c in retrieved]
    return {
        "top1": any(matches[:1]),
        "top3": any(matches[:3]),
        "top5": any(matches[:5]),
        "top10": any(matches[:10]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--bm25-index", required=True)
    ap.add_argument("--mode", choices=["bm25", "dense", "hybrid"], required=True)
    ap.add_argument("--faiss-index", default="")
    ap.add_argument("--faiss-metadata", default="")
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    ap.add_argument("--candidate-k", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5, help="dense weight in hybrid")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    questions = load_questions(args.questions)

    with open(args.bm25_index, "rb") as f:
        bm25 = pickle.load(f)

    bm25_metadata = bm25["metadata"]

    model = None
    faiss_index = None
    faiss_metadata = None

    if args.mode in {"dense", "hybrid"}:
        model = SentenceTransformer(args.embed_model)
        faiss_index = faiss.read_index(args.faiss_index)
        with open(args.faiss_metadata, "rb") as f:
            faiss_metadata = pickle.load(f)

    totals = {"top1": 0, "top3": 0, "top5": 0, "top10": 0}
    n = 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out:
        for q in questions:
            query = q["question"]

            if args.mode == "bm25":
                bm25_items = bm25_search(query, bm25, top_k=args.top_k)
                retrieved = make_results(bm25_items, bm25_metadata)

            elif args.mode == "dense":
                dense_items = dense_search(query, model, args.embed_model, faiss_index, args.top_k)
                retrieved = make_results(dense_items, faiss_metadata)

            else:
                dense_items = dense_search(query, model, args.embed_model, faiss_index, args.candidate_k)
                bm25_items = bm25_search(query, bm25, top_k=args.candidate_k)

                dense_norm = normalize_scores(dense_items)
                bm25_norm = normalize_scores(bm25_items)

                union_ids = set(dense_norm) | set(bm25_norm)
                hybrid = []
                for idx in union_ids:
                    score = args.alpha * dense_norm.get(idx, 0.0) + (1 - args.alpha) * bm25_norm.get(idx, 0.0)
                    hybrid.append((idx, score))

                hybrid = sorted(hybrid, key=lambda x: x[1], reverse=True)[:args.top_k]
                retrieved = make_results(hybrid, faiss_metadata)

            hits = calc_hits(retrieved, q)
            for k in totals:
                totals[k] += int(hits[k])
            n += 1

            out.write(json.dumps({
                "id": q.get("id"),
                "question": query,
                "type": q.get("type"),
                "source_mode": q.get("source_mode"),
                "mode": args.mode,
                "alpha": args.alpha,
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

    print("MODE:", args.mode)
    print("QUESTIONS:", args.questions)
    print("ALPHA:", args.alpha)
    print("N:", n)
    for k, v in totals.items():
        print(f"{k}: {v}/{n} = {v/n:.2%}")


if __name__ == "__main__":
    main()
