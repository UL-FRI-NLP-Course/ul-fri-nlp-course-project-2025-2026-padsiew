import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


PROJECT = Path("/d/hpc/projects/onj_fri/pad-siew/law-rag")
OUTPUT_DIR = PROJECT / "data/outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_questions(path: Path):
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def make_prompt(question: str) -> str:
    return f"""Si pravni asistent za slovensko pravo.

Odgovori na vprašanje v slovenščini.
Odgovor naj bo kratek in jasen.
Če nisi prepričan, to jasno povej.
Ne izmišljaj si številk členov ali pravnih virov.

VPRAŠANJE:
{question}

ODGOVOR:
"""


def clean_answer(text: str) -> str:
    text = text.strip()

    stop_markers = [
        "<end_of_turn>",
        "\nVprašanje:",
        "\nVPRAŠANJE:",
        "\nQuestion:",
        "\nVIRI:",
        "\nViri:",
        "\nSOURCES:",
        "\nReferences",
        "\nREFERENCE",
        "\nODGOVOR:",
        "\nSklic",
        "\n###",
        "\n---",
    ]

    for marker in stop_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].strip()

    text = text.replace("<pad>", "").replace("```", "").strip()
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


def is_gams3_model(model_name: str) -> bool:
    return "gams3" in model_name.lower()


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


def normalize_generation_config(model, model_name: str):
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None


def model_dtype(model_name: str):
    if not torch.cuda.is_available():
        return torch.float32
    if is_gams3_model(model_name):
        return torch.bfloat16
    return torch.float16


def generate_answer(question, tokenizer, model, max_new_tokens, model_name):
    if is_gams3_model(model_name):
        user_prompt = f"Odgovori kratko v slovenščini: {question}"
    else:
        user_prompt = f"""Odgovori na vprašanje v slovenščini.
Odgovor naj bo kratek in jasen.
Če nisi prepričan, to jasno povej.
Ne izmišljaj si številk členov ali pravnih virov.

VPRAŠANJE:
{question}
"""

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = f"""Si pravni asistent za slovensko pravo.

{user_prompt}

ODGOVOR:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]
    generation_kwargs = get_generation_kwargs(
        tokenizer=tokenizer,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            **generation_kwargs,
        )
    generated_ids = output[0][input_len:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return clean_answer(answer)
# def generate_answer(question, tokenizer, model, max_new_tokens):
#     prompt = make_prompt(question)

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True,
#         max_length=4096,
#     ).to(model.device)

#     input_len = inputs["input_ids"].shape[1]

#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#         )

#     generated_ids = output[0][input_len:]
#     answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
#     return clean_answer(answer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default=str(PROJECT / "data/eval/questions_30.jsonl"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--run-name", default="no_rag")
    args = parser.parse_args()

    questions_path = Path(args.questions)

    print(f"Using model: {args.model}", flush=True)
    print(f"Questions: {questions_path}", flush=True)

    questions = load_questions(questions_path)
    print(f"Loaded questions: {len(questions)}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=model_dtype(args.model),
        device_map="auto",
        local_files_only=True,
    )
    normalize_generation_config(model, args.model)
    print("Model loaded.", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace("/", "__")
    output_path = OUTPUT_DIR / f"{args.run_name}_{safe_model}_{timestamp}.jsonl"

    with output_path.open("w", encoding="utf-8") as out:
        for q in questions:
            qid = q.get("id", "")
            question = q["question"]

            print("=" * 80, flush=True)
            print(f"{qid}: {question}", flush=True)

            answer = generate_answer(
                question=question,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_new_tokens,
                model_name=args.model,
            )

            record = {
                "id": qid,
                "question": question,
                "topic": q.get("topic", ""),
                "type": q.get("type", ""),
                "expected_source": q.get("expected_source", ""),
                "expected_article": q.get("expected_article", ""),
                "model": args.model,
                "mode": "no_rag",
                "answer": answer,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            print("ANSWER:", answer, flush=True)

    print("=" * 80, flush=True)
    print(f"Saved results to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
