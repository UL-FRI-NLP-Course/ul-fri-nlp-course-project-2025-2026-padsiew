import argparse
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path


TOKEN_RE = re.compile(r"[a-zA-ZčšžćđČŠŽĆĐ0-9]+")


def tokenize(text: str):
    return [t.lower() for t in TOKEN_RE.findall(text or "") if len(t) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    metadata = []
    doc_lens = []
    inverted = defaultdict(dict)

    with open(args.input, encoding="utf-8") as f:
        for doc_id, line in enumerate(f):
            if not line.strip():
                continue
            c = json.loads(line)
            text = c.get("text", "")
            toks = tokenize(text)
            tf = Counter(toks)

            metadata.append(c)
            doc_lens.append(len(toks))

            for term, freq in tf.items():
                inverted[term][doc_id] = freq

            if (doc_id + 1) % 5000 == 0:
                print(f"Processed {doc_id+1}", flush=True)

    n_docs = len(metadata)
    avgdl = sum(doc_lens) / max(1, n_docs)

    idf = {}
    for term, postings in inverted.items():
        df = len(postings)
        idf[term] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    obj = {
        "metadata": metadata,
        "doc_lens": doc_lens,
        "avgdl": avgdl,
        "idf": idf,
        "inverted": dict(inverted),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(obj, f)

    print("Done.")
    print("Docs:", n_docs)
    print("Terms:", len(idf))
    print("Output:", args.output)


if __name__ == "__main__":
    main()
