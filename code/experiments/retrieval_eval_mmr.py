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


def mmr_select(candidate_embs, relevance_scores, top_k=10, lambda_mult=0.7):
    """
    candidate_embs are normalized vectors.
    relevance_scores are query similarity scores from FAISS.
    """
    selected = []
    remaining = list(range(len(candidate_embs)))

    while remaining and len(selected) < top_k:
        if not selected:
            best = max(remaining, key=lambda i: relevance_scores[i])
            selected.append(best)
            remaining.remove(best)
            continue

        selected_embs = candidate_embs[selected]
        mmr_scores = []

        for i in remaining:
            relevance = relevance_scores[i]
            diversity_penalty = np.max(candidate_embs[i] @ selected_embs.T)
            score = lambda_mult * relevance - (1.0 - lambda_mult) * diversity_penalty
            mmr_scores.append((score, i))

        best = max(mmr_scores, key=lambda x: x[0])[1]
        selected.append(best)
        remaining.remove(best)

    return selected


def retrieve_dense(question, model, model_name, index, metadata, top_k):
    qq = format_query(question, model_name)
    q_emb = model.encode([qq], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, ids = index.search(q_emb, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx < 0:
            continue
        c = dict(metadata[idx])
        c["rank"] = rank
        c["score"] = float(score)
        c["dense_rank"] = rank
        c["dense_score"] = float(score)
        results.append(c)

    return results


def retrieve_mmr(question, model, model_name, index, metadata, candidate_k, final_k, lambda_mult):
    qq = format_query(question, model_name)
    q_emb = model.encode([qq], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, ids = index.search(q_emb, candidate_k)

    valid_ids = []
    valid_scores = []

    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            valid_ids.append(int(idx))
            valid_scores.append(float(score))

    candidate_chunks = [metadata[i] for i in valid_ids]
    candidate_texts = [c.get("text", "") for c in candidate_chunks]

    # Re-embed candidate chunks for pairwise similarity.
    # For E5, documents need passage prefix.
    if "e5" in model_name.lower():
        candidate_texts_for_emb = ["passage: " + t for t in candidate_texts]
    else:
        candidate_texts_for_emb = candidate_texts

    candidate_embs = model.encode(candidate_texts_for_emb, normalize_embeddings=True)
    candidate_embs = np.asarray(candidate_embs, dtype="float32")
    relevance_scores = np.asarray(valid_scores, dtype="float32")

    selected_local_ids = mmr_select(
        candidate_embs=candidate_embs,
        relevance_scores=relevance_scores,
        top_k=final_k,
        lambda_mult=lambda_mult,
    )

    results = []
    for rank, local_i in enumerate(selected_local_ids, start=1):
        c = dict(candidate_chunks[local_i])
        c["rank"] = rank
        c["score"] = float(valid_scores[local_i])
        c["dense_rank"] = local_i + 1
        c["dense_score"] = float(valid_scores[local_i])
        c["mmr_rank"] = rank
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
    ap.add_argument("--embed-model", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--mode", choices=["dense", "mmr"], default="mmr")
    ap.add_argument("--candidate-k", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--lambda-mult", type=float, default=0.7)
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
            if args.mode == "dense":
                retrieved = retrieve_dense(
                    q["question"], model, args.embed_model, index, metadata, args.top_k
                )
            else:
                retrieved = retrieve_mmr(
                    q["question"], model, args.embed_model, index, metadata,
                    args.candidate_k, args.top_k, args.lambda_mult
                )

            hits = calc_hits(retrieved, q)
            for k in totals:
                totals[k] += int(hits[k])
            n += 1

            out.write(json.dumps({
                "id": q.get("id"),
                "question": q["question"],
                "type": q.get("type"),
                "source_mode": q.get("source_mode"),
                "embed_model": args.embed_model,
                "mode": args.mode,
                "candidate_k": args.candidate_k,
                "top_k": args.top_k,
                "lambda_mult": args.lambda_mult,
                "hits": hits,
                "retrieved": [
                    {
                        "rank": c.get("rank"),
                        "score": c.get("score"),
                        "dense_rank": c.get("dense_rank"),
                        "dense_score": c.get("dense_score"),
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
    print("MODE:", args.mode)
    print("candidate_k:", args.candidate_k)
    print("top_k:", args.top_k)
    print("lambda_mult:", args.lambda_mult)
    print("N:", n)

    for k, v in totals.items():
        print(f"{k}: {v}/{n} = {v/n:.2%}")


if __name__ == "__main__":
    main()
