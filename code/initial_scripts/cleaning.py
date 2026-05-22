import json
import re

# ===== CONFIG =====
INPUT_FILE = r"COLESLAW 1.0/PISRS/register-predpisov.jsonl"
OUTPUT_FILE = "real_estate_chunks.jsonl"
LIMIT = 20000


# ===== REAL ESTATE FILTER =====
def is_real_estate_law(title):
    keywords = [
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
        "urejanje prostora"
    ]

    title = title.lower()
    return any(k in title for k in keywords)


# ===== CLEAN TEXT =====
def clean_text(text):
    text = text.replace("\xa0", " ")

    # remove "Opozorilo..." lines
    text = re.sub(r"Opozorilo:.*?\n", "", text)

    # normalize spaces BUT keep newlines for splitting
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


# ===== FILTER GOOD TEXT =====
def is_good_text(text):
    return (
        text is not None and
        len(text) > 500 and
        "člen" in text.lower()
    )


# ===== SPLIT BY ČLEN =====
def split_by_clen(text):
    parts = re.split(r"\n(?=\d+\.\s*člen)", text)

    chunks = []
    for part in parts:
        part = part.strip()

        if len(part) > 100:

            # ensure it's a real article
            if not re.match(r"^\d+\.\s*člen", part.lower()):
                continue

            # remove newlines AFTER splitting
            part = part.replace("\n", " ")
            part = re.sub(r"\s+", " ", part)

            chunks.append(part)

    return chunks


# ===== MAIN =====
def process():
    chunks = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= LIMIT:
                break

            try:
                doc = json.loads(line)
            except:
                continue

            text = doc.get("text", "")
            title = doc.get("naziv", "")
            doc_id = doc.get("id", None)

            # ===== FILTER REAL ESTATE LAWS =====
            if not is_real_estate_law(title):
                continue

            if not is_good_text(text):
                continue

            text = clean_text(text)
            article_chunks = split_by_clen(text)

            for chunk in article_chunks:
                chunks.append({
                    "text": chunk,
                    "source": title,
                    "id": doc_id,
                    "article": chunk.split(" ")[0]  # e.g. "1."
                })

            if i % 100 == 0:
                print(f"Processed {i} documents...")

    print(f"\nTotal chunks created: {len(chunks)}")

    # ===== SAVE =====
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            json.dump(c, f, ensure_ascii=False)
            f.write("\n")


# ===== RUN =====
if __name__ == "__main__":
    process()