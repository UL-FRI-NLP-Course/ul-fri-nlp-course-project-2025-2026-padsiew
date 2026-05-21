import argparse
import json
import random
from pathlib import Path


BAD_TITLE_KEYWORDS = [
    "prenehanje",
    "prehodne",
    "končne",
    "kazenske določbe",
    "začetek veljavnosti",
    "uskladitev",
    "rok za",
]

GOOD_TOPIC_KEYWORDS = [
    "kataster",
    "zemljiška knjiga",
    "lastninska pravica",
    "etažna lastnina",
    "služnost",
    "hipoteka",
    "stavbna pravica",
    "posest",
    "gradbeno dovoljenje",
    "urejanje prostora",
    "komasacija",
    "parcel",
    "nepremičnin",
]


def load_chunks(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def is_good_chunk(c):
    text = c.get("text", "")
    title = (c.get("article_title") or "").lower()
    source = (c.get("source") or "").lower()
    joined = f"{title} {source} {text[:1000]}".lower()

    if len(text) < 300:
        return False
    if len(text) > 3500:
        return False
    if any(bad in joined for bad in BAD_TITLE_KEYWORDS):
        return False
    if not any(good in joined for good in GOOD_TOPIC_KEYWORDS):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/real_estate_chunks_full.jsonl")
    parser.add_argument("--output-single", default="data/eval/sample_single_source_chunks.jsonl")
    parser.add_argument("--output-multi", default="data/eval/sample_multi_source_groups.jsonl")
    parser.add_argument("--n-single", type=int, default=60)
    parser.add_argument("--n-multi-groups", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    chunks = load_chunks(args.input)
    good = [c for c in chunks if is_good_chunk(c)]

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Good chunks: {len(good)}")

    random.shuffle(good)

    out_single = Path(args.output_single)
    out_multi = Path(args.output_multi)
    out_single.parent.mkdir(parents=True, exist_ok=True)

    # single-source candidates
    singles = good[: args.n_single]

    with open(out_single, "w", encoding="utf-8") as f:
        for c in singles:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # multi-source groups: group by source/topic-ish words, then sample pairs/triples
    by_source = {}
    for c in good:
        source = c.get("source", "")
        by_source.setdefault(source, []).append(c)

    groups = []

    # 10 groups with 2 chunks, 10 groups with 3 chunks if possible
    sources = [s for s, cs in by_source.items() if len(cs) >= 3]
    random.shuffle(sources)

    for source in sources:
        if len(groups) >= args.n_multi_groups:
            break

        cs = by_source[source]
        random.shuffle(cs)

        size = 2 if len(groups) < 10 else 3
        group = cs[:size]

        groups.append({
            "group_id": f"multi_{len(groups)+1:03d}",
            "source": source,
            "chunks": group,
        })

    with open(out_multi, "w", encoding="utf-8") as f:
        for g in groups:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    print(f"Wrote singles: {out_single} ({len(singles)})")
    print(f"Wrote multi groups: {out_multi} ({len(groups)})")


if __name__ == "__main__":
    main()
