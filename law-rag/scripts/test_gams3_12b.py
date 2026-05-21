import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = "cjvt/GaMS3-12B-Instruct"

print(f"Testing model: {MODEL_ID}", flush=True)
print("Import OK", flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

print("Loading tokenizer...", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    local_files_only=True,
)
print(f"Tokenizer loaded in {time.time() - t0:.1f}s", flush=True)

print("Loading model...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    local_files_only=True,
)
print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

if torch.cuda.is_available():
    print(
        "GPU memory allocated GB:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        flush=True,
    )
    print(
        "GPU memory reserved GB:",
        round(torch.cuda.memory_reserved() / 1024**3, 2),
        flush=True,
    )

prompt = """Si pravni asistent za slovensko pravo.

Odgovori kratko in jasno.

VPRAŠANJE:
Kaj pomeni služnost?

ODGOVOR:
"""

print("Tokenizing prompt...", flush=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
input_len = inputs["input_ids"].shape[1]

print("Generating...", flush=True)
t0 = time.time()
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_ids = output[0][input_len:]
answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"Generated in {time.time() - t0:.1f}s", flush=True)
print("=" * 80, flush=True)
print("ANSWER:", flush=True)
print(answer, flush=True)
print("=" * 80, flush=True)
print("DONE", flush=True)
