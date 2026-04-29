import argparse
import json
import re
import unicodedata
from pathlib import Path


REAL_ESTATE_KEYWORDS = [
    "nepremičnin",
    "nepremičnine",
    "zemljišč",
    "zemljišče",
    "lastnina",
    "stvarnopravni",
    "stvarno pravo",
    "etažna",
    "gradnja",
    "gradbeni",
    "prostor",
    "prostorski",
    "urbanizem",
    "stavb",
    "parcela",
    "parcel",
    "kataster",
    "zemljiška knjiga",
    "vpis",
    "objekt",
    "stanovanje",
    "hiša",
    "posest",
    "hipoteka",
    "služnost",
    "gradbeno dovoljenje",
    "urejanje prostora",
    "komasacija",
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = text.replace("\u00ad", "")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # Remove common PISRS warning lines.
    text = re.sub(r"Opozorilo:.*?(?=\n)", "", text, flags=re.IGNORECASE)

    # Fix missing space in headings like "3.2Upravna komasacija"
    text = re.sub(r"(\d+\.\d+)([A-ZČŠŽ])", r"\1 \2", text)

    # Normalize spaces but keep newlines for splitting.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_real_estate_law(title: str, text: str = "") -> bool:
    haystack = f"{title} {text[:3000]}".lower()
    return any(keyword in haystack for keyword in REAL_ESTATE_KEYWORDS)


def is_good_text(text: str) -> bool:
    return bool(text) and len(text) > 300 and "člen" in text.lower()


def extract_article_number(article_text: str) -> str:
    match = re.match(r"^\s*(\d+)\.\s*člen", article_text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def extract_article_title(article_text: str) -> str:
    # Matches: 170. člen (vpis pogodbene komasacije)
    match = re.match(
        r"^\s*\d+\.\s*člen\s*\(([^)]+)\)",
        article_text,
        flags=re.IGNORECASE,
    )
    return match.group(1).strip() if match else ""


def split_by_article(text: str) -> list[str]:
    """
    Split by article headings like:
      170. člen
      170. člen (title)

    Keeps article heading inside each chunk.
    """
    text = normalize_text(text)

    # Ensure article headings are on their own split boundary where possible.
    # Then split before lines starting with "number. člen".
    parts = re.split(r"\n(?=\s*\d+\.\s*člen\b)", text, flags=re.IGNORECASE)

    chunks = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if not re.match(r"^\s*\d+\.\s*člen\b", part, flags=re.IGNORECASE):
            continue

        # Flatten newlines after article splitting.
        part = part.replace("\n", " ")
        part = re.sub(r"\s+", " ", part).strip()

        # Remove very short false positives.
        if len(part) < 100:
            continue

        chunks.append(part)

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input PISRS JSONL file")
    parser.add_argument("--output", required=True, help="Output chunks JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Optional max documents")
    parser.add_argument(
        "--filter-real-estate",
        action="store_true",
        help="Keep only real-estate-related laws",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    kept_docs = 0
    total_chunks = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if args.limit is not None and i >= args.limit:
                break

            if not line.strip():
                continue

            total_docs += 1

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = doc.get("text", "") or ""
            title = doc.get("naziv", "") or doc.get("title", "") or doc.get("source", "") or ""
            doc_id = doc.get("id", None)
            source_url = doc.get("url", "") or doc.get("source_url", "")

            if not is_good_text(text):
                continue

            if args.filter_real_estate and not is_real_estate_law(title, text):
                continue

            text = normalize_text(text)
            article_chunks = split_by_article(text)

            if not article_chunks:
                continue

            kept_docs += 1

            for chunk in article_chunks:
                article = extract_article_number(chunk)
                article_title = extract_article_title(chunk)

                record = {
                    "chunk_id": f"{doc_id}-{article}" if doc_id is not None and article else f"{doc_id}-{total_chunks}",
                    "doc_id": doc_id,
                    "source": title,
                    "source_url": source_url,
                    "article": article,
                    "article_title": article_title,
                    "text": chunk,
                }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

            if total_docs % 500 == 0:
                print(f"Processed docs: {total_docs}, kept docs: {kept_docs}, chunks: {total_chunks}", flush=True)

    print("Done.", flush=True)
    print(f"Total docs read: {total_docs}", flush=True)
    print(f"Kept docs: {kept_docs}", flush=True)
    print(f"Chunks written: {total_chunks}", flush=True)
    print(f"Output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
