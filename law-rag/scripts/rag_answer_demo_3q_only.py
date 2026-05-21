import argparse
import os
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


PROJECT = Path("/d/hpc/projects/onj_fri/pad-siew/law-rag")

DEFAULT_INDEX_PATH = PROJECT / "index/real_estate.faiss"
DEFAULT_METADATA_PATH = PROJECT / "index/real_estate_metadata.pkl"

EMBED_MODEL = "BAAI/bge-m3"
LLM_MODEL = "cjvt/GaMS-2B-Instruct"


DEFAULT_QUESTIONS = [
    "Kako se vpiše pogodbena komasacija?",
    "Kaj ureja kataster nepremičnin?",
    "Kaj pomeni služnost?",
]


def load_retriever(index_path: Path, metadata_path: Path):
    print(f"Loading FAISS index: {index_path}", flush=True)
    index = faiss.read_index(str(index_path))

    print(f"Loading metadata: {metadata_path}", flush=True)
    with metadata_path.open("rb") as f:
        records = pickle.load(f)

    print(f"Loaded records: {len(records)}", flush=True)

    print(f"Loading embedding model: {EMBED_MODEL}", flush=True)
    embedder = SentenceTransformer(EMBED_MODEL)

    return index, records, embedder


def retrieve(question: str, index, records, embedder, top_k: int):
    q_emb = embedder.encode(
        [question],
        normalize_embeddings=True,
    )
    q_emb = np.asarray(q_emb, dtype="float32")

    scores, ids = index.search(q_emb, top_k)

    results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue

        item = dict(records[idx])
        item["score"] = float(score)
        results.append(item)

    return results


def format_sources(contexts):
    parts = []

    for i, c in enumerate(contexts, start=1):
        source = c.get("source", "")
        article = c.get("article", "")
        title = c.get("article_title", "")
        text = c.get("text", "")

        # Keep each context reasonably short for the 2B model.
        # Later we can tune this.
        max_chars = 700
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."

        header = f"[{i}] {source}"
        if article:
            header += f", {article}. člen"
        if title:
            header += f" ({title})"

        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


def make_prompt(question: str, contexts):
    sources_text = format_sources(contexts)
    return f"""Si pravni asistent za slovensko pravo.

    Odgovori v slovenščini, največ v dveh stavkih.
    Uporabi samo spodnje vire.
    Uporabi samo vire, ki so neposredno povezani z vprašanjem.
    Če so med viri različna pravna področja, izberi najrelevantnejše vire in prezri nerelevantne.
    Ne dodajaj informacij, ki niso razvidne iz virov.
    Ne dodajaj novih vprašanj, primerov ali seznamov, razen če uporabnik to izrecno zahteva.
    Če odgovor ni razviden iz virov, napiši: "Iz navedenih virov tega ni mogoče zanesljivo ugotoviti."
    Na koncu stavkov dodaj sklice na uporabljene vire, npr. [1] ali [2].

    VIRI:
    {sources_text}

    VPRAŠANJE:
    {question}

    ODGOVOR:
    """

def load_gams():
    print(f"HF_HOME: {os.environ.get('HF_HOME')}", flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    print(f"Loading tokenizer: {LLM_MODEL}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL,
        local_files_only=True,
    )
    print(f"Tokenizer loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"Loading model: {LLM_MODEL}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        local_files_only=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    if torch.cuda.is_available():
        print(
            "GPU memory allocated GB:",
            round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            flush=True,
        )
        print(
            "GPU memory reserved GB:",
            round(torch.cuda.memory_reserved(0) / 1024**3, 2),
            flush=True,
        )

    return tokenizer, model

def clean_generated_answer(answer: str) -> str:
    import re

    answer = answer.strip()

    # Cut off any extra sections the model invents after the actual answer.
    patterns = [
        r"\s*VPRAŠANJE\s*:",
        r"\s*QUESTION\s*:",
        r"\s*ODGOVOR\s*:",
        r"\s*KRATEK ODGOVOR\s*:",
        r"\s*DOLGI ODGOV[OA]R\s*:",
        r"\s*VIRI\s*:",
        r"\s*VIR\s*:",
        r"\s*SOURCES\s*:",
        r"\s*SOURCE\s*:",
        r"\s*ODGOVORI NA VIR[EIO]*\s*:",
        r"\s*Pravila\s*:",
        r"\s*Naloga\s*:",
    ]

    for pattern in patterns:
        match = re.search(pattern, answer, flags=re.IGNORECASE)
        if match:
            answer = answer[:match.start()].strip()

    return answer.strip()

def generate_answer(question: str, contexts, tokenizer, model, max_new_tokens: int):
    prompt = make_prompt(question, contexts)

    print("\n" + "=" * 100, flush=True)
    print("QUESTION:", question, flush=True)

    print("\nRETRIEVED SOURCES:", flush=True)
    for i, c in enumerate(contexts, start=1):
        print(
            f"[{i}] score={c['score']:.4f} | "
            f"{c.get('source', '')}, {c.get('article', '')}. člen "
            f"({c.get('article_title', '')})",
            flush=True,
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=7000,
    ).to(model.device)

    print("Generating answer...", flush=True)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    generated_tokens = output[0][input_len:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    answer = clean_generated_answer(decoded)

    if "[" not in answer and contexts:
        answer = answer.rstrip(".") + ". [1]"

    print("\nANSWER:", flush=True)
    print(answer, flush=True)
    print("\nSOURCES:", flush=True)
    for i, c in enumerate(contexts, start=1):
        print(
            f"[{i}] {c.get('source', '')}, "
            f"{c.get('article', '')}. člen "
            f"({c.get('article_title', '')})",
            flush=True,
        )
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=str(DEFAULT_INDEX_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--question", action="append")
    args = parser.parse_args()

    questions = args.question if args.question else DEFAULT_QUESTIONS

    index, records, embedder = load_retriever(
        Path(args.index),
        Path(args.metadata),
    )

    tokenizer, model = load_gams()

    for question in questions:
        contexts = retrieve(
            question=question,
            index=index,
            records=records,
            embedder=embedder,
            top_k=args.top_k,
        )

        generate_answer(
            question=question,
            contexts=contexts,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
