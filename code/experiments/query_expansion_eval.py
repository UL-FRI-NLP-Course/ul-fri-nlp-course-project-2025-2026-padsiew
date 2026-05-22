import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT = Path("/d/hpc/projects/onj_fri/pad-siew/law-rag")

QUESTIONS_PATH = PROJECT / "data/eval/questions_30.jsonl"
INDEX_PATH = PROJECT / "index/real_estate_full.faiss"
METADATA_PATH = PROJECT / "index/real_estate_full_metadata.pkl"
OUTPUT_PATH = PROJECT / "data/outputs/query_expansion_eval.jsonl"

EMBED_MODEL = "BAAI/bge-m3"
TOP_K = 10


LEGAL_TERMS = {
    "kataster": "kataster nepremičnin zemljiški kataster Geodetska uprava ZKN",
    "zemljiška knjiga": "zemljiška knjiga zemljiškoknjižni predlog vpis ZZK-1",
    "služnost": "služnost stvarna služnost osebna služnost SPZ",
    "hipoteka": "hipoteka zastavna pravica nepremičnina SPZ",
    "etažna lastnina": "etažna lastnina posamezni del skupni deli SPZ",
    "stavbna pravica": "stavbna pravica zgradba nad pod tujo nepremičnino SPZ",
    "posest": "posest neposredna dejanska oblast stvar SPZ",
    "lastninska pravica": "lastninska pravica uporabljati uživati razpolagati SPZ",
    "gradbeno dovoljenje": "gradbeno dovoljenje pogoji za izdajo gradbenega dovoljenja GZ-1",
    "komasacija": "pogodbena komasacija komasacijsko soglasje vpis kataster nepremičnin ZUreP-3",
    "urejanje prostora": "urejanje prostora trajnostni prostorski razvoj ZUreP-3",
}


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(x):
    return (x or "").lower().strip()


def expand_query(question, topic=""):
    q = question
    haystack = normalize(question + " " + topic)

    additions = []
    for key, terms in LEGAL_TERMS.items():
        if key in haystack:
            additions.append(terms)

    if additions:
        return q + " " + " ".join(additions)

    return q


def expected_match(chunk, expected_source, expected_article):
    expected_source = normalize(expected_source)
    expected_article = normalize(str(expected_article)).replace(".", "")

    source = normalize(chunk.get("source", ""))
    article = normalize(str(chunk.get("article", ""))).replace(".", "")

    source_ok = True
    article_ok = True

    if expected_source:
        source_ok = expected_source in source or source in expected_source
    if expected_article:
        article_ok = article == expected_article

    return source_ok and article_ok


def retrieve(query, embedder, index, metadata):
    emb = embedder.encode([query], normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")
    scores, ids = index.search(emb, TOP_K)

    out = []
    for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
        if idx < 0:
            continue
        c = dict(metadata[idx])
        c["rank"] = rank
        c["score"] = float(score)
        out.append(c)
    return out


def calc_hits(retrieved, expected_source, expected_article):
    matches = [expected_match(c, expected_source, expected_article) for c in retrieved]
    return {
        "top1": any(matches[:1]),
        "top3": any(matches[:3]),
        "top5": any(matches[:5]),
        "top10": any(matches[:10]),
    }


def main():
    questions = load_questions(QUESTIONS_PATH)

    index = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    embedder = SentenceTransformer(EMBED_MODEL)

    totals = {
        "original": {"top1": 0, "top3": 0, "top5": 0, "top10": 0},
        "expanded": {"top1": 0, "top3": 0, "top5": 0, "top10": 0},
    }
    n = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for q in questions:
            question = q["question"]
            topic = q.get("topic", "")
            expected_source = q.get("expected_source", "")
            expected_article = str(q.get("expected_article", ""))

            if not expected_source and not expected_article:
                continue

            n += 1

            expanded = expand_query(question, topic)

            original_ret = retrieve(question, embedder, index, metadata)
            expanded_ret = retrieve(expanded, embedder, index, metadata)

            original_hits = calc_hits(original_ret, expected_source, expected_article)
            expanded_hits = calc_hits(expanded_ret, expected_source, expected_article)

            for k in totals["original"]:
                totals["original"][k] += int(original_hits[k])
                totals["expanded"][k] += int(expanded_hits[k])

            record = {
                "id": q.get("id"),
                "question": question,
                "topic": topic,
                "expected_source": expected_source,
                "expected_article": expected_article,
                "expanded_query": expanded,
                "original_hits": original_hits,
                "expanded_hits": expanded_hits,
                "original_retrieved": [
                    {
                        "rank": c["rank"],
                        "score": c["score"],
                        "source": c.get("source"),
                        "article": c.get("article"),
                        "article_title": c.get("article_title", ""),
                    }
                    for c in original_ret
                ],
                "expanded_retrieved": [
                    {
                        "rank": c["rank"],
                        "score": c["score"],
                        "source": c.get("source"),
                        "article": c.get("article"),
                        "article_title": c.get("article_title", ""),
                    }
                    for c in expanded_ret
                ],
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(q.get("id"), original_hits, "=>", expanded_hits, flush=True)

    print("N =", n)
    for mode in ["original", "expanded"]:
        print(mode)
        for k, v in totals[mode].items():
            print(f"  {k}: {v}/{n} = {v/n:.2%}")


if __name__ == "__main__":
    main()
