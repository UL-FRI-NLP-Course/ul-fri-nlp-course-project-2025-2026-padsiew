import argparse
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import CrossEncoder

PROJECT = Path("/d/hpc/projects/onj_fri/pad-siew/law-rag")

QUESTIONS_PATH = PROJECT / "data/eval/questions_50_article_derived.jsonl"
# QUESTIONS_PATH = PROJECT / "data/eval/questions_30.jsonl"
OUTPUT_DIR = PROJECT / "data/outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = PROJECT / "index/real_estate_full.faiss"
METADATA_PATH = PROJECT / "index/real_estate_full_metadata.pkl"

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

TOP_K = 10
# CONTEXT_N = 2        # MOVED TO ARG PARSER!!!
# print("CONTEXT_N", CONTEXT_N)
MAX_CONTEXT_CHARS_PER_CHUNK = 1200 # 700
MAX_NEW_TOKENS = 220 # 160


def is_gams3_model(model_name: str) -> bool:
    return "gams3" in model_name.lower()


def model_dtype(model_name: str):
    if not torch.cuda.is_available():
        return torch.float32
    if is_gams3_model(model_name):
        return torch.bfloat16
    return torch.float16
def load_questions(path: Path):
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def normalize_text(value: str) -> str:
    return (value or "").strip().lower()

def compact_chunk_text(chunk: dict, max_chars: int = 1200) -> str:
    text = chunk.get("text", "")
    text = re.sub(r"\s+", " ", text).strip()

    source = chunk.get("source", "")
    article = str(chunk.get("article", "")).replace(".", "")
    title = chunk.get("article_title") or chunk.get("title") or ""

    prefix = f"Vir: {source}. Člen: {article}. Naslov: {title}. "
    return (prefix + text)[:max_chars]


def rerank(question: str, candidates: list[dict], reranker) -> list[dict]:
    pairs = [
        (question, compact_chunk_text(c, max_chars=1200))
        for c in candidates
    ]

    scores = reranker.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    reranked = []
    for c, score in zip(candidates, scores):
        item = dict(c)
        item["dense_rank"] = item.get("rank")
        item["dense_score"] = item.get("score")
        item["rerank_score"] = float(score)
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    for i, item in enumerate(reranked, start=1):
        item["rank"] = i
        item["rerank_rank"] = i

    return reranked

def expected_match(chunk: dict, expected_source: str, expected_article: str) -> bool:
    expected_source = normalize_text(expected_source)
    expected_article = normalize_text(expected_article).replace(".", "")

    source = normalize_text(chunk.get("source", ""))
    article = normalize_text(str(chunk.get("article", ""))).replace(".", "")

    source_ok = True
    article_ok = True

    if expected_source:
        source_ok = expected_source in source or source in expected_source

    if expected_article:
        article_ok = article == expected_article

    return source_ok and article_ok

def make_prompt(
    question: str,
    contexts: list[dict],
    prompt_id: str = "p1",
    max_context_chars: int = 700,
) -> str:
    source_blocks = []

    for i, c in enumerate(contexts, start=1):
        text = c.get("text", "")
        text = re.sub(r"\s+", " ", text).strip()
        text = text[:max_context_chars]

        source = c.get("source", "")
        article = str(c.get("article", "")).replace(".", "")
        title = c.get("article_title") or c.get("title") or ""

        source_blocks.append(
            f"[{i}] Vir: {source}, {article}. člen"
            + (f" ({title})" if title else "")
            + f"\nBesedilo: {text}"
        )

    sources_text = "\n\n".join(source_blocks)

    prompts = {
        "p1": f"""Si pravni asistent za slovensko pravo.

Naloga: odgovori na vprašanje samo na podlagi navedenih virov.

Pravila:
- Odgovori v slovenščini.
- Odgovor naj ima največ dva stavka.
- Ne dodajaj naslovov, seznamov, razdelkov, novih vprašanj ali dodatnih primerov.
- Ne dodajaj razdelka "Viri", "References" ali "###".
- Uporabi samo vire, ki so neposredno povezani z vprašanjem.
- Če je med viri očitno nerelevanten vir, ga prezri.
- Če odgovor ni razviden iz virov, napiši samo: "Iz navedenih virov tega ni mogoče zanesljivo ugotoviti."
- Na koncu odgovora dodaj sklic na uporabljeni vir, npr. [1].

VIRI:
{sources_text}

VPRAŠANJE:
{question}

ODGOVOR:
""",

        "p2": f"""Odgovori kot pravni asistent za slovensko pravo.

Uporabi izključno spodnje vire.
Odgovor naj bo zelo kratek: največ en ali dva stavka.
Ne dodajaj pojasnil, naslovov ali seznama virov.
Če odgovora ni v virih, napiši samo: "Iz navedenih virov tega ni mogoče zanesljivo ugotoviti."
Na koncu dodaj sklic v obliki [1] ali [2].

VIRI:
{sources_text}

VPRAŠANJE:
{question}

ODGOVOR:
""",

        "p3": f"""Si pravni asistent. Tvoja naloga je odgovoriti z natančnim sklicem na pravni vir.

Pravila:
- Odgovori samo na podlagi spodnjih virov.
- Vsaka vsebinska trditev mora biti podprta s sklicem, npr. [1].
- Ne uporabljaj virov, ki niso neposredno relevantni.
- Ne izmišljaj zakonov, členov ali pogojev.
- Ne dodajaj posebnega razdelka z viri.
- Odgovor naj ima največ dva stavka.
- Če vira ni dovolj, napiši samo: "Iz navedenih virov tega ni mogoče zanesljivo ugotoviti."

VIRI:
{sources_text}

VPRAŠANJE:
{question}

ODGOVOR:
""",

        "p4": f"""Si previden pravni asistent za slovensko pravo.

Odgovori samo, če je odgovor jasno razviden iz navedenih virov.
Če so viri nerelevantni, dvoumni ali ne zadostujejo za odgovor, napiši samo:
"Iz navedenih virov tega ni mogoče zanesljivo ugotoviti."

Pravila:
- Ne ugibaj.
- Ne uporabljaj splošnega znanja.
- Ne dodajaj novih vprašanj, primerov, naslovov ali seznama virov.
- Odgovor naj ima največ dva stavka.
- Na koncu dodaj sklic na uporabljeni vir, npr. [1].

VIRI:
{sources_text}

VPRAŠANJE:
{question}

ODGOVOR:
""",

        "p5": f"""Si pravni asistent za slovensko pravo.

Odgovori samo na podlagi spodnjih virov.
Če vprašanje zahteva primerjavo, presojo trditve ali praktičen primer, odgovori v največ štirih stavkih.
Sicer odgovori v največ dveh stavkih.
Pri primerjavi jasno povej razliko med pojmi.
Pri trditvah "drži/ne drži" najprej napiši "Drži" ali "Ne drži".
Ne dodajaj razdelkov, seznama virov ali dodatnih primerov.
Če odgovor ni razviden iz virov, napiši samo: "Iz navedenih virov tega ni mogoče zanesljivo ugotoviti."
Na koncu dodaj sklic na uporabljeni vir, npr. [1].

VIRI:
{sources_text}

VPRAŠANJE:
{question}

ODGOVOR:
""",
    }

    return prompts[prompt_id]


def clean_answer(text: str) -> str:
    text = text.strip()

    # Remove accidental continuation into new examples/questions.
    stop_markers = [
        "\nVprašanje:",
        "\nVPRAŠANJE:",
        "\nQuestion:",
        "\nVIRI:",
        "\nViri:",
        "\nSOURCES:",
        "\nSources:",
        "\nReferences",
        "\nREFERENCE",
        "\nODGOVOR:",
        "\nOdgovor:",
        "\nSklic",
        "\n###",
        "\n---",
        "<eos>",
        "<end_of_turn>",
    ]

    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()

    # Remove accidental fallback phrase if it appears after a real answer.
    fallback = "Iz navedenih virov tega ni mogoče zanesljivo ugotoviti."
    if fallback in text and len(text.replace(fallback, "").strip()) > 40:
        text = text.replace(fallback, "").strip()

    # Remove unfinished trailing source-section starts.
    text = re.sub(r"\s*VIRI:\s*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*Sklic na vir:\s*$", "", text, flags=re.IGNORECASE).strip()

    # Keep answer compact.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def get_eos_token_ids(tokenizer):
    eos_ids = []

    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)

    end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if (
        isinstance(end_of_turn_id, int)
        and end_of_turn_id >= 0
        and tokenizer.convert_ids_to_tokens(end_of_turn_id) == "<end_of_turn>"
    ):
        eos_ids.append(end_of_turn_id)

    return list(dict.fromkeys(eos_ids))


def get_generation_kwargs(tokenizer, model_name: str, max_new_tokens: int):
    eos_token_ids = get_eos_token_ids(tokenizer)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    kwargs = {
        "max_new_tokens": max_new_tokens,
        "eos_token_id": eos_token_ids[0] if len(eos_token_ids) == 1 else eos_token_ids,
        "pad_token_id": pad_token_id,
        "repetition_penalty": 1.05,
        "remove_invalid_values": True,
        "renormalize_logits": True,
    }

    if pad_token_id is not None and pad_token_id not in eos_token_ids:
        kwargs["bad_words_ids"] = [[pad_token_id]]

    if is_gams3_model(model_name):
        kwargs.update(
            {
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 0.9,
            }
        )
    else:
        kwargs["do_sample"] = False

    return kwargs


def normalize_generation_config(model):
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None


def retrieve(question: str, embedder, index, metadata, top_k: int = TOP_K):
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


def generate_answer(
    question: str,
    contexts: list[dict],
    tokenizer,
    model,
    model_name: str,
    max_new_tokens: int,
    prompt_id: str,
    max_context_chars: int,
):
    prompt = make_prompt(question, contexts, prompt_id, max_context_chars)

    if is_gams3_model(model_name) and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6500).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    generation_kwargs = get_generation_kwargs(tokenizer, model_name, max_new_tokens)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            **generation_kwargs,
        )

    generated_ids = output[0][input_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    # answer = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return clean_answer(answer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    parser.add_argument("--model", default="cjvt/GaMS-2B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--run-name", default="rag_eval")
    parser.add_argument(
        "--prompt-id",
        choices=["p1", "p2", "p3", "p4", "p5"],
        default="p1",
    )
    parser.add_argument("--context-n", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=700)
    args = parser.parse_args()

    questions_path = Path(args.questions)
    print(f"Using LLM_MODEL: {args.model}", flush=True)
    print(f"Questions: {questions_path}", flush=True)

    print("Loading questions...", flush=True)
    questions = load_questions(questions_path)
    print(f"Loaded questions: {len(questions)}", flush=True)

    print("Loading FAISS index and metadata...", flush=True)
    index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("rb") as f:
        metadata = pickle.load(f)

    print(f"Loaded chunks: {len(metadata)}", flush=True)

    print("Loading embedder...", flush=True)
    embedder = SentenceTransformer(EMBED_MODEL)

    # print("Loading reranker...", flush=True)
    # reranker = CrossEncoder(
    #     RERANK_MODEL,
    #     device="cuda" if torch.cuda.is_available() else "cpu",
    # )
    # print("Reranker loaded.", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "__")
    output_path = OUTPUT_DIR / f"{args.run_name}_{args.prompt_id}_{safe_model}_{timestamp}.jsonl"

    total_with_expected = 0
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    prepared = []

    print("Retrieving and reranking contexts...", flush=True)
    for q in questions:
        qid = q.get("id", "")
        question = q["question"]
        expected_source = q.get("expected_source", "")
        expected_article = str(q.get("expected_article", ""))

        retrieved = retrieve(question, embedder, index, metadata, TOP_K)
        # retrieved_dense = retrieve(question, embedder, index, metadata, TOP_K)
        # retrieved = rerank(question, retrieved_dense, reranker)

        top1 = False
        top3 = False
        top5 = False

        has_expected = bool(expected_source or expected_article)

        if has_expected:
            total_with_expected += 1
            matches = [
                expected_match(chunk, expected_source, expected_article)
                for chunk in retrieved
            ]

            top1 = any(matches[:1])
            top3 = any(matches[:3])
            top5 = any(matches[:5])

            top1_hits += int(top1)
            top3_hits += int(top3)
            top5_hits += int(top5)

        prepared.append(
            {
                "question_record": q,
                "retrieved": retrieved,
                "expected_source": expected_source,
                "expected_article": expected_article,
                "top1": top1,
                "top3": top3,
                "top5": top5,
            }
        )

        print(f"Prepared {qid}: top3_expected={top3}", flush=True)

    del embedder
   # del reranker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    print("Loading GaMS model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype(args.model),
        device_map="auto",
        local_files_only=True,
        attn_implementation="eager",
    )
    normalize_generation_config(model)

    print("Model loaded.", flush=True)

    with output_path.open("w", encoding="utf-8") as out:
        for item in prepared:
            q = item["question_record"]
            retrieved = item["retrieved"]
            qid = q.get("id", "")
            question = q["question"]

            print("=" * 80, flush=True)
            print(f"{qid}: {question}", flush=True)

            answer = generate_answer(
                question=question,
                # contexts=retrieved[:RERANK_TOP_N],
                contexts=retrieved[:args.context_n],
                tokenizer=tokenizer,
                model=model,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                prompt_id=args.prompt_id,
                max_context_chars=args.max_context_chars,
            )

            record = {
                "id": qid,
                "question": question,
                "topic": q.get("topic", ""),
                "expected_source": item["expected_source"],
                "expected_article": item["expected_article"],
                "expected_in_top1": item["top1"],
                "expected_in_top3": item["top3"],
                "expected_in_top5": item["top5"],
                "model": args.model,
                "mode": "rag",
                "retrieved": [
                    {
                        "rank": c.get("rank"),
                        "score": c.get("score"),
                        "chunk_id": c.get("chunk_id") or c.get("id"),
                        "source": c.get("source"),
                        "article": c.get("article"),
                        "article_title": c.get("article_title") or c.get("title", ""),
                        "text": c.get("text", ""),
                        #"text_preview": c.get("text", "")[:500],
                    }
                    for c in retrieved
                ],
                "answer": answer,
                "prompt_id": args.prompt_id,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            print("ANSWER:", answer, flush=True)
            print("TOP RETRIEVED:", flush=True)
            for c in retrieved[:args.context_n]:
                print(
                    f"  rank={c.get('rank')} score={c.get('score'):.4f} | "
                    f"{c.get('source')} | {c.get('article')}. člen",
                    flush=True,
                )
            # for c in retrieved[:5]:
            #     print(
            #         f"  rerank_rank={c.get('rerank_rank')} dense_rank={c.get('dense_rank')} "
            #         f"dense_score={c.get('dense_score'):.4f}" # rerank_score={c.get('rerank_score'):.4f} | "
            #         f"{c.get('source')} | {c.get('article')}. člen",
            #         flush=True,
            #     )

    print("=" * 80, flush=True)
    print(f"Saved results to: {output_path}", flush=True)

    if total_with_expected:
        print("Retrieval metrics for questions with expected source/article:", flush=True)
        print(f"Top-1: {top1_hits}/{total_with_expected} = {top1_hits / total_with_expected:.2%}", flush=True)
        print(f"Top-3: {top3_hits}/{total_with_expected} = {top3_hits / total_with_expected:.2%}", flush=True)
        print(f"Top-5: {top5_hits}/{total_with_expected} = {top5_hits / total_with_expected:.2%}", flush=True)
    else:
        print("No expected labels provided; qualitative review only.", flush=True)


if __name__ == "__main__":
    main()
